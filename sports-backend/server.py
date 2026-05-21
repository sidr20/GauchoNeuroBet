# Save this file as server.py
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv3
from datetime import datetime
import warnings

# --- Important backend setup ---
app = Flask(__name__)
CORS(app)

# --- In-Memory Cache for Models and Scalers ---
SESSION_MODELS = {}
SESSION_SCALERS = {}

print("--- Server is running with XGBoost, Advanced Features, and Live Context. ---")

# --- Load shared static data ---
print("Loading static API and CSV data...")
all_players_list = players.get_players()
player_dict = {player['full_name'].upper(): player['id'] for player in all_players_list}

all_teams_list = teams.get_teams()
team_abbrev_to_name = {team["abbreviation"]: team["full_name"] for team in all_teams_list}

try:
    all_defensive_ratings = pd.read_csv("estimated_defensive_ratings_since_2003.csv")
    all_team_stats = pd.read_csv("nba_team_stats_since_2003.csv")
except FileNotFoundError:
    print("FATAL ERROR: Make sure CSV files are in the same directory as server.py")
    exit()

print("All static data loaded.")

CORRECT_COLUMNS = ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE",
                   "MATCHUP", "WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                   "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PLUS_MINUS"]

# UPDATED: Added new advanced features, removed basic 'Back_to_Back' in favor of fatigue metrics
FEATURES_LIST = ["Rolling_PTS", "Rolling_AST", "Rolling_REB", 
                 "Rolling_MIN", "Rolling_FGM", "Rolling_FGA", "Rolling_FG3M", "Rolling_FG3A",
                 "Rolling_FTM", "Rolling_FTA", "Rolling_STL", "Rolling_BLK", "Rolling_TO", 
                 "PTS_lag1", "AST_lag1", "REB_lag1", "MIN_lag1", "FGA_lag1",
                 "HOME_GAME", "OPP_E_DEF_RATING", "OPP_TEAM_STL", "OPP_TEAM_BLK", "OPP_TEAM_WIN_PCT",
                 "Weighted_PTS_Form", "PTS_Variance_5g", "Days_Rest", "Is_Fatigued", "Is_Well_Rested", "Home_Advantage_Multiplier"]


# --- Helper: Get Tonight's Game Context ---
# --- Helper: Get Tonight's Game Context ---
def get_tonights_context(team_id, last_game_date):
    today = datetime.today()
    last_game = pd.to_datetime(last_game_date)
    days_rest = (today - last_game).days
    
    # Use ScoreboardV3 with explicit game date
    board = scoreboardv3.ScoreboardV3(game_date=today.strftime('%Y-%m-%d'))
    games = board.get_dict()['scoreboard']['games']
    
    is_home_game = 0
    opponent_id = None
    
    for game in games:
        home_team_id = game['homeTeam']['teamId']
        away_team_id = game['awayTeam']['teamId']
        
        if team_id == home_team_id:
            is_home_game = 1
            opponent_id = away_team_id
            break
        elif team_id == away_team_id:
            is_home_game = 0
            opponent_id = home_team_id
            break
            
    if not opponent_id:
        return {"error": "Player's team does not have a game scheduled for today."}
        
    opponent_name = next((t['full_name'] for t in all_teams_list if t['id'] == opponent_id), None)
    
    return {
        "OPPONENT": opponent_name,
        "HOME_GAME": is_home_game,
        "DAYS_REST": days_rest
    }

# --- Endpoint to provide player list to front-end ---
@app.route('/players', methods=['GET'])
def get_players():
    formatted_players = [{'id': str(p['id']), 'fullName': p['full_name']} for p in all_players_list if p['is_active']]
    return jsonify(formatted_players)

# --- Main Prediction Logic ---
def get_player_prediction(player_name, stat_to_check):
    team_name_replacements = {
        "Charlotte Bobcats": "Charlotte Hornets",
        "Seattle SuperSonics": "Oklahoma City Thunder",
        "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
        "New Jersey Nets": "Brooklyn Nets",
        "New Orleans Hornets": "New Orleans Pelicans"
    }
    
    def feature_engineer(df):
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(by="GAME_DATE").reset_index(drop=True)
        
        # Base Lags and Rolling
        df['PTS_lag1'] = df['PTS'].shift(1)
        df['AST_lag1'] = df['AST'].shift(1)
        df['REB_lag1'] = df['REB'].shift(1)
        df['MIN_lag1'] = df['MIN'].shift(1)
        df['FGA_lag1'] = df['FGA'].shift(1)

        for col in ['PTS', 'AST', 'REB', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'STL', 'BLK', 'TO']:
            df[f'Rolling_{col}'] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
        
        df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if isinstance(x, str) and "vs." in x else 0)
        df['Weighted_PTS_Form'] = df['PTS'].shift(1).ewm(span=5, adjust=False).mean()
        df['PTS_Variance_5g'] = df['PTS'].shift(1).rolling(window=5, min_periods=1).std().fillna(0)
        
        df['Home_Advantage_Multiplier'] = df['HOME_GAME'] * df['Rolling_PTS']
        
        # Advanced Fatigue
        df['Days_Rest'] = df['GAME_DATE'].diff().dt.days.fillna(2)
        df['Is_Fatigued'] = df['Days_Rest'].apply(lambda x: 1 if x <= 1 else 0)
        df['Is_Well_Rested'] = df['Days_Rest'].apply(lambda x: 1 if x >= 3 else 0)

        def extract_opponent(matchup):
            if not isinstance(matchup, str): return None
            try:
                part = matchup.split("vs. ")[1] if "vs." in matchup else matchup.split("@ ")[1]
                return team_abbrev_to_name.get(part, part)
            except IndexError: return None
        df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)

        def convert_season_id(season_id):
            year = int(season_id[-4:])
            return f"{year}-{str(year + 1)[-2:]}"
        df["SEASON_ID"] = df["SEASON_ID"].apply(convert_season_id)

        # Merge Opponent Defensive Ratings
        def_ratings = all_defensive_ratings.copy()
        def_ratings.rename(columns={"SEASON": "SEASON_ID"}, inplace=True)
        def_stats = def_ratings[['TEAM_NAME', 'SEASON_ID', 'E_DEF_RATING']]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            def_stats.loc[:, "TEAM_NAME"] = def_stats["TEAM_NAME"].replace(team_name_replacements)
        df = df.merge(def_stats, left_on=["OPPONENT", "SEASON_ID"], right_on=["TEAM_NAME", "SEASON_ID"], how="left")
        df.rename(columns={"E_DEF_RATING": "OPP_E_DEF_RATING"}, inplace=True)
        if "TEAM_NAME_y" in df.columns: df = df.drop(columns=["TEAM_NAME_y"])

        # Merge Opponent Team Stats
        team_stats_df = all_team_stats.copy()
        team_stats_df.rename(columns={"YEAR": "SEASON_ID", "STL": "TEAM_STL", "BLK": "TEAM_BLK", "WIN_PCT": "TEAM_WIN_PCT"}, inplace=True)
        opp_team_stats = team_stats_df[['TEAM_NAME', 'SEASON_ID', 'TEAM_STL', 'TEAM_BLK', 'TEAM_WIN_PCT']]
        df["OPPONENT"] = df["OPPONENT"].replace({"Los Angeles Clippers": "LA Clippers"})
        df = df.merge(opp_team_stats, left_on=["OPPONENT", "SEASON_ID"], right_on=["TEAM_NAME", "SEASON_ID"], how="left")
        df.rename(columns={"TEAM_STL": "OPP_TEAM_STL", "TEAM_BLK": "OPP_TEAM_BLK", "TEAM_WIN_PCT": "OPP_TEAM_WIN_PCT"}, inplace=True)
        if "TEAM_NAME_y" in df.columns: df = df.drop(columns=["TEAM_NAME_y"])
        if "TEAM_NAME_x" in df.columns: df = df.rename(columns={'TEAM_NAME_x': 'TEAM_NAME'})
        
        df = df.ffill().bfill()
        return df

    player_id = player_dict.get(player_name.upper())
    if not player_id: return {"error": "Player not found."}
    
    # FIXED: Unique Cache Key
    cache_key = f"{player_id}_{stat_to_check}"

    if cache_key in SESSION_MODELS:
        model = SESSION_MODELS[cache_key]
        scaler = SESSION_SCALERS[cache_key]
    else:
        print(f"No cached model for {player_name} - {stat_to_check}. Training XGBoost...")
            
        gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        games = gamefinder.get_dict()['resultSets'][0]['rowSet']
        if len(games) < 50: return {"error": f"Not enough data ({len(games)} games) to train."}
        
        data = pd.DataFrame(games, columns=CORRECT_COLUMNS)
        data = feature_engineer(data)
        data = data.dropna(subset=FEATURES_LIST)
        
        if data.empty: return {"error": "Not enough complete data for training."}

        target = stat_to_check.upper()
        if target not in data.columns: return {"error": f"Stat '{target}' not found in data."}
        
        X_train = data[FEATURES_LIST]
        y_train = data[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
        
        print(f"Training in progress...")
        model.fit(X_train_scaled, y_train, verbose=False)
        
        print("Training complete. Caching model.")
        SESSION_MODELS[cache_key] = model
        SESSION_SCALERS[cache_key] = scaler

    # --- Prediction Step ---
    gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
    games = gamefinder.get_dict()['resultSets'][0]['rowSet']
    if not games: return {"error": "No recent game data found."}
    
    prediction_data = pd.DataFrame(games, columns=CORRECT_COLUMNS)
    prediction_data = feature_engineer(prediction_data)
    
    # Fetch Tonight's Reality
    player_team_id = prediction_data['TEAM_ID'].iloc[-1]
    last_played_date = prediction_data['GAME_DATE'].iloc[-1]
    
    tonight = get_tonights_context(player_team_id, last_played_date)
    if "error" in tonight:
        return {"error": tonight["error"]}
        
    # Grab current form from their LAST game
    current_form = prediction_data.iloc[-1].copy()
    
    # Overwrite historical context with TONIGHT'S context
    current_form['HOME_GAME'] = tonight['HOME_GAME']
    current_form['Days_Rest'] = tonight['DAYS_REST']
    current_form['Is_Fatigued'] = 1 if tonight['DAYS_REST'] <= 1 else 0
    current_form['Is_Well_Rested'] = 1 if tonight['DAYS_REST'] >= 3 else 0
    current_form['Home_Advantage_Multiplier'] = tonight['HOME_GAME'] * current_form['Rolling_PTS']
    
    opp_def = all_defensive_ratings[all_defensive_ratings['TEAM_NAME'] == tonight['OPPONENT']]
    if not opp_def.empty:
        latest_opp_def = opp_def.iloc[-1]
        current_form['OPP_E_DEF_RATING'] = latest_opp_def['E_DEF_RATING']

    opp_stats = all_team_stats[all_team_stats['TEAM_NAME'] == tonight['OPPONENT']]
    if not opp_stats.empty:
        latest_opp_stats = opp_stats.iloc[-1]
        current_form['OPP_TEAM_STL'] = latest_opp_stats['STL']
        current_form['OPP_TEAM_BLK'] = latest_opp_stats['BLK']
        current_form['OPP_TEAM_WIN_PCT'] = latest_opp_stats['WIN_PCT']
    
    # Format, Scale, and Predict
    next_game_features = current_form[FEATURES_LIST].values.reshape(1, -1)
    next_game_features = np.nan_to_num(next_game_features)
    next_game_features_scaled = scaler.transform(next_game_features)
    
    predicted_stat = float(model.predict(next_game_features_scaled)[0])
    
    # Floor negative predictions at 0 (can't score -2 points)
    predicted_stat = max(0, predicted_stat)

    return {
        "playerName": player_name, 
        "statName": stat_to_check, 
        "predictedValue": f"{predicted_stat:.2f}",
        "opponent": tonight['OPPONENT'],
        "homeGame": bool(tonight['HOME_GAME'])
    }

@app.route('/predict', methods=['GET'])
def predict():
    player_name = request.args.get('player')
    stat_to_check = request.args.get('stat')
    if not player_name or not stat_to_check: return jsonify({"error": "Missing 'player' or 'stat' parameter"}), 400
    print(f"\nReceived prediction request for {player_name} for stat {stat_to_check}...")
    prediction_result = get_player_prediction(player_name, stat_to_check)
    print(f"Prediction result: {prediction_result}")
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)