# Save this file as server.py
# Final version with scoping fix for team_name_replacements.
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguegamefinder
import os
import warnings

# --- Important backend setup ---
app = Flask(__name__)
CORS(app)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- In-Memory Cache for Models and Scalers ---
SESSION_MODELS = {}
SESSION_SCALERS = {}

print("--- Server is running with Full Feature Engineering and Scoping Fix. ---")

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

FEATURES_LIST = ["Rolling_PTS", "Rolling_AST", "Rolling_REB", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                 "FTM", "FTA", "FT_PCT", "OREB", "DREB", "STL", "BLK", "TO", "PF", "PLUS_MINUS", "PTS_lag1", "AST_lag1", "REB_lag1", 
                 "HOME_GAME", "Back_to_Back", "OPP_E_DEF_RATING", "OPP_TEAM_STL", "OPP_TEAM_BLK", "OPP_TEAM_WIN_PCT"]


# --- Endpoint to provide player list to front-end ---
@app.route('/players', methods=['GET'])
def get_players():
    formatted_players = [{'id': str(p['id']), 'fullName': p['full_name']} for p in all_players_list if p['is_active']]
    return jsonify(formatted_players)


def get_player_prediction(player_name, stat_to_check):
    # --- FIX: Define this dictionary at the top level of the function ---
    team_name_replacements = {
        "Charlotte Bobcats": "Charlotte Hornets",
        "Seattle SuperSonics": "Oklahoma City Thunder",
        "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
        "New Jersey Nets": "Brooklyn Nets",
        "New Orleans Hornets": "New Orleans Pelicans"
    }
    
    # This function handles all feature engineering.
    def feature_engineer(df):
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(by="GAME_DATE")
        df['PTS_lag1'] = df['PTS'].shift(1)
        df['AST_lag1'] = df['AST'].shift(1)
        df['REB_lag1'] = df['REB'].shift(1)
        df["Rolling_PTS"] = df["PTS"].rolling(window=3, min_periods=1).mean()
        df["Rolling_AST"] = df["AST"].rolling(window=3, min_periods=1).mean()
        df["Rolling_REB"] = df["REB"].rolling(window=3, min_periods=1).mean()
        df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if isinstance(x, str) and "vs." in x else 0)
        df['Back_to_Back'] = df['GAME_DATE'].diff().dt.days.fillna(0).apply(lambda x: 1 if x == 1 else 0)

        def extract_opponent(matchup):
            if not isinstance(matchup, str):
                return None
            try:
                part = matchup.split("vs. ")[1] if "vs." in matchup else matchup.split("@ ")[1]
                return team_abbrev_to_name.get(part, part)
            except IndexError: return None
        df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)

        def convert_season_id(season_id):
            year = int(season_id[-4:])
            return f"{year}-{str(year + 1)[-2:]}"
        df["SEASON_ID"] = df["SEASON_ID"].apply(convert_season_id)

        def_ratings = all_defensive_ratings.copy()
        def_ratings.rename(columns={"SEASON": "SEASON_ID"}, inplace=True)
        def_stats = def_ratings[['TEAM_NAME', 'SEASON_ID', 'E_DEF_RATING']]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            def_stats.loc[:, "TEAM_NAME"] = def_stats["TEAM_NAME"].replace(team_name_replacements)
        df = df.merge(def_stats, left_on=["OPPONENT", "SEASON_ID"], right_on=["TEAM_NAME", "SEASON_ID"], how="left")
        df.rename(columns={"E_DEF_RATING": "OPP_E_DEF_RATING"}, inplace=True)
        if "TEAM_NAME_y" in df.columns: df = df.drop(columns=["TEAM_NAME_y"])

        team_stats_df = all_team_stats.copy()
        team_stats_df.rename(columns={"YEAR": "SEASON_ID", "STL": "TEAM_STL", "BLK": "TEAM_BLK", "WIN_PCT": "TEAM_WIN_PCT"}, inplace=True)
        opp_team_stats = team_stats_df[['TEAM_NAME', 'SEASON_ID', 'TEAM_STL', 'TEAM_BLK', 'TEAM_WIN_PCT']]
        df["OPPONENT"] = df["OPPONENT"].replace({"Los Angeles Clippers": "LA Clippers"})
        df = df.merge(opp_team_stats, left_on=["OPPONENT", "SEASON_ID"], right_on=["TEAM_NAME", "SEASON_ID"], how="left")
        df.rename(columns={"TEAM_STL": "OPP_TEAM_STL", "TEAM_BLK": "OPP_TEAM_BLK", "TEAM_WIN_PCT": "OPP_TEAM_WIN_PCT"}, inplace=True)
        if "TEAM_NAME_y" in df.columns: df = df.drop(columns=["TEAM_NAME_y"])
        if "TEAM_NAME_x" in df.columns: df = df.rename(columns={'TEAM_NAME_x': 'TEAM_NAME'})
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    # Check Cache First
    if stat_to_check in SESSION_MODELS:
        model = SESSION_MODELS[stat_to_check]
        scaler = SESSION_SCALERS[stat_to_check]
    else:
        print(f"No cached model for {stat_to_check}. Training a new one...")
        player_id = player_dict.get(player_name.upper())
        if not player_id: return {"error": "Player not found."}
            
        gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
        games = gamefinder.get_dict()['resultSets'][0]['rowSet']
        if len(games) < 50: return {"error": f"Not enough data ({len(games)} games) to train."}
        
        data = pd.DataFrame(games, columns=CORRECT_COLUMNS)
        data = feature_engineer(data)
        data = data.dropna(subset=FEATURES_LIST)
        
        if data.empty: return {"error": "Not enough complete data for training."}

        target = stat_to_check.upper()
        X_train = data[FEATURES_LIST]
        y_train = data[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
        from tensorflow.keras.regularizers import l2
        
        model = Sequential([
            Dense(128, input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.01)), BatchNormalization(), LeakyReLU(), Dropout(0.3),
            Dense(64, kernel_regularizer=l2(0.01)), BatchNormalization(), LeakyReLU(), Dropout(0.3),
            Dense(32, kernel_regularizer=l2(0.01)), BatchNormalization(), LeakyReLU(), Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae')
        
        print(f"Training in progress...")
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, verbose=0)
        print("Training complete. Caching model.")
        SESSION_MODELS[stat_to_check] = model
        SESSION_SCALERS[stat_to_check] = scaler

    # --- Prediction Step ---
    player_id = player_dict.get(player_name.upper())
    if not player_id: return {"error": "Player not found."}

    gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id)
    games = gamefinder.get_dict()['resultSets'][0]['rowSet']
    if not games: return {"error": "No recent game data found."}
    
    prediction_data = pd.DataFrame(games, columns=CORRECT_COLUMNS)
    prediction_data = feature_engineer(prediction_data)
    
    next_game_features = prediction_data[FEATURES_LIST].iloc[-10:].mean().values.reshape(1, -1)
    next_game_features = np.nan_to_num(next_game_features)
    next_game_features_scaled = scaler.transform(next_game_features)
    predicted_stat = model.predict(next_game_features_scaled)[0][0]

    return {"playerName": player_name, "statName": stat_to_check, "predictedValue": f"{predicted_stat:.2f}"}

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
