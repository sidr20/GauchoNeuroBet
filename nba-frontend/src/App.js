import React, { useState, useEffect } from 'react';
import { Search, BarChart2, Zap, User, Loader, Target } from 'lucide-react';
import logo from './logo.svg';

// This function is the CORE of the front-end to back-end connection
const getPrediction = async (player, stat) => {
    const endpoint = `http://127.0.0.1:5000/predict?player=${encodeURIComponent(player.fullName)}&stat=${stat}`;
    console.log(`Fetching from: ${endpoint}`);
    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    return data;
};

export default function App() {
    // --- STATE MANAGEMENT ---
    const [allPlayers, setAllPlayers] = useState([]); // Will hold the full list of players from the backend
    const [playerName, setPlayerName] = useState('');
    const [statToPredict, setStatToPredict] = useState('PTS');
    const [suggestions, setSuggestions] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const statOptions = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FTM'];

    // --- FETCH ALL PLAYERS ON INITIAL LOAD ---
    // This useEffect runs only once when the component first mounts
    useEffect(() => {
        const fetchAllPlayers = async () => {
            try {
                const response = await fetch('/players');
                const playersList = await response.json();
                setAllPlayers(playersList);
                console.log(`Loaded ${playersList.length} active players.`);
            } catch (err) {
                console.error("Failed to fetch players list from backend:", err);
                setError("Could not load player list. Is the backend server running?");
            }
        };
        fetchAllPlayers();
    }, []); // Empty dependency array means it runs once on mount

    // --- HANDLE SEARCH INPUT CHANGES ---
    // This now filters the full list of players from the state
    useEffect(() => {
        if (playerName.trim() === '') {
            setSuggestions([]);
            return;
        }
        // Filter the full list of players based on the input
        const filteredSuggestions = allPlayers
            .filter(p => p.fullName.toLowerCase().includes(playerName.toLowerCase()))
            .slice(0, 7); // Limit to the top 7 matches to keep the list clean
        setSuggestions(filteredSuggestions);

    }, [playerName, allPlayers]); // Rerun whenever the input or the full player list changes

    // --- HANDLE ACTIONS ---
    const handleSearch = async (player, stat) => {
        if (!player) {
            setError('Please select a player.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setPrediction(null);
        setPlayerName(player.fullName);
        setSuggestions([]);
        try {
            const pred = await getPrediction(player, stat);
            if(pred.error) {
                setError(pred.error);
                setPrediction(null);
            } else {
                setPrediction(pred);
            }
        } catch (err) {
            setError('Could not connect to the prediction backend. Is it running?');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handlePlayerSelect = (player) => {
        setPlayerName(player.fullName);
        setSuggestions([]);
        handleSearch(player, statToPredict);
    }

    // --- RENDER THE COMPONENT ---
    return (
        <div className="bg-gray-900 min-h-screen text-white font-sans p-4 sm:p-6 md:p-8">
            <div className="max-w-4xl mx-auto">
                <header className="text-center mb-8">
                    <div className="flex items-center justify-center mb-2 gap-3">
                        <img 
                        src={logo} 
                        alt="Gaucho NeuroBet logo" 
                        className="w-12 h-12" 
                        />
                        <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-300">
                            Gaucho NeuroBet
                        </h1>
                    </div>
                </header>

                <div className="bg-gray-800/50 p-4 rounded-lg shadow-lg mb-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="relative md:col-span-2">
                            <input
                                type="text"
                                value={playerName}
                                onChange={(e) => setPlayerName(e.target.value)}
                                placeholder="Search for any active NBA player..."
                                className="w-full p-4 pl-12 bg-gray-800 border-2 border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                            {suggestions.length > 0 && (
                                <ul className="absolute z-10 w-full bg-gray-800 border border-gray-700 rounded-lg mt-1 max-h-60 overflow-y-auto">
                                    {suggestions.map((player) => (
                                        <li
                                            key={player.id}
                                            className="p-3 hover:bg-gray-700 cursor-pointer flex items-center"
                                            onClick={() => handlePlayerSelect(player)}
                                        >
                                            <User className="w-5 h-5 mr-3 text-gray-400"/>
                                            {player.fullName}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                        <div className="relative">
                             <select 
                                value={statToPredict}
                                onChange={(e) => setStatToPredict(e.target.value)}
                                className="w-full p-4 pl-12 bg-gray-800 border-2 border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none"
                             >
                               {statOptions.map(stat => <option key={stat} value={stat}>{stat}</option>)}
                             </select>
                             <Target className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400"/>
                        </div>
                    </div>
                </div>

                <main>
                    {isLoading && (
                        <div className="flex flex-col items-center justify-center p-8 text-center">
                            <Loader className="w-12 h-12 animate-spin text-blue-400 mb-4" />
                            <p className="text-lg">Contacting server & running model...</p>
                        </div>
                    )}
                    {error && (
                        <div className="bg-red-900/50 border border-red-700 text-red-300 p-4 rounded-lg text-center">
                            <strong>Error:</strong> {error}
                        </div>
                    )}
                    {!isLoading && !prediction && !error && (
                         <div className="text-center text-gray-500 p-8">
                            {allPlayers.length === 0 && !error ? 
                                <><Loader className="w-8 h-8 animate-spin text-gray-600 mb-4 mx-auto"/> <p>Loading player list from server...</p></>
                                :
                                <><Zap size={48} className="mx-auto mb-4" /> <h2 className="text-2xl font-semibold">Ready for an absolute lock?</h2></>
                            }
                        </div>
                    )}
                    {prediction && (
                        <div className="bg-gray-800/50 p-6 rounded-xl shadow-2xl animate-fade-in">
                            <h2 className="text-3xl font-bold text-center mb-2">
                                Prediction for <span className="text-blue-400">{prediction.playerName}</span>
                            </h2>
                            <p className="text-center text-gray-400 mb-6">Next Game Projected {prediction.statName}</p>
                            <div className="flex flex-col items-center">
                                <div className="text-7xl font-bold text-white">{prediction.predictedValue}</div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
             <style jsx global>{` @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } } .animate-fade-in { animation: fade-in 0.5s ease-out forwards; } `}</style>
        </div>
    );
}
