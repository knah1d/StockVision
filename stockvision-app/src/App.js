import React, { useState, useEffect } from "react";
import "./App.css";
import "./utils/chartConfig"; // Import global Chart.js configuration
import Dashboard from "./components/Dashboard";
import StockAnalysis from "./components/StockAnalysis";
import StockComparison from "./components/StockComparison";
import SectorAnalysis from "./components/SectorAnalysis";
import StockPrediction from "./components/StockPrediction";
import ApiService from "./services/apiService";

function App() {
    const [currentView, setCurrentView] = useState("dashboard");
    const [isLoading, setIsLoading] = useState(true);
    const [apiHealth, setApiHealth] = useState(false);

    useEffect(() => {
        checkApiHealth();
    }, []);

    const checkApiHealth = async () => {
        try {
            await ApiService.healthCheck();
            setApiHealth(true);
        } catch (error) {
            setApiHealth(false);
            console.error("API not available:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const renderView = () => {
        switch (currentView) {
            case "dashboard":
                return <Dashboard />;
            case "analysis":
                return <StockAnalysis />;
            case "comparison":
                return <StockComparison />;
            case "sector":
                return <SectorAnalysis />;
            case "prediction":
                return <StockPrediction />;
            default:
                return <Dashboard />;
        }
    };

    if (isLoading) {
        return (
            <div className="loading-container">
                <div className="loading-spinner"></div>
                <p>Loading StockVision...</p>
            </div>
        );
    }

    if (!apiHealth) {
        return (
            <div className="error-container">
                <h2>⚠️ API Connection Error</h2>
                <p>Cannot connect to StockVision API backend.</p>
                <p>
                    Please make sure the backend server is running on port 8001.
                </p>
                <button onClick={checkApiHealth} className="retry-button">
                    🔄 Retry Connection
                </button>
            </div>
        );
    }

    return (
        <div className="App">
            <header className="app-header">
                <div className="header-content">
                    <h1 className="app-title">
                        📈 StockVision
                        <span className="app-subtitle">
                            Dhaka Stock Exchange Analysis
                        </span>
                    </h1>
                    <nav className="app-nav">
                        <button
                            className={
                                currentView === "dashboard"
                                    ? "nav-button active"
                                    : "nav-button"
                            }
                            onClick={() => setCurrentView("dashboard")}
                        >
                            📊 Dashboard
                        </button>
                        <button
                            className={
                                currentView === "analysis"
                                    ? "nav-button active"
                                    : "nav-button"
                            }
                            onClick={() => setCurrentView("analysis")}
                        >
                            🔍 Analysis
                        </button>
                        <button
                            className={
                                currentView === "comparison"
                                    ? "nav-button active"
                                    : "nav-button"
                            }
                            onClick={() => setCurrentView("comparison")}
                        >
                            🔄 Compare
                        </button>
                        <button
                            className={
                                currentView === "sector"
                                    ? "nav-button active"
                                    : "nav-button"
                            }
                            onClick={() => setCurrentView("sector")}
                        >
                            🏢 Sectors
                        </button>
                        <button
                            className={
                                currentView === "prediction"
                                    ? "nav-button active"
                                    : "nav-button"
                            }
                            onClick={() => setCurrentView("prediction")}
                        >
                            🤖 Prediction
                        </button>
                    </nav>
                </div>
            </header>

            <main className="app-main">{renderView()}</main>

            <footer className="app-footer">
                <p>
                    © 2025 StockVision - Dhaka Stock Exchange Analysis Platform
                </p>
            </footer>
        </div>
    );
}

export default App;
