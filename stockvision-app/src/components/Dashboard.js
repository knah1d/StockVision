import React, { useState, useEffect } from "react";
import ApiService from "../services/apiService";
import ExplainButton from "./ExplainButton";

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [tickers, setTickers] = useState([]);
    const [sectors, setSectors] = useState([]);
    const [volatileStocks, setVolatileStocks] = useState([]);
    const [predictionHealth, setPredictionHealth] = useState(null);
    const [quickPredictions, setQuickPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        try {
            setLoading(true);
            setError(null);

            // Load API stats
            const statsData = await ApiService.getStats();
            setStats(statsData);

            // Load available tickers and sectors
            const tickersData = await ApiService.getTickers(null, 20);
            setTickers(tickersData.tickers);
            setSectors(tickersData.sectors);

            // Load volatile stocks
            const volatileData = await ApiService.findVolatileStocks(
                null,
                30,
                5
            );
            setVolatileStocks(volatileData.volatile_stocks);

            // Check prediction service health
            try {
                const healthData = await ApiService.getPredictionHealth();
                setPredictionHealth(healthData);

                // Load quick predictions for top 3 stocks if service is healthy
                if (
                    healthData.status === "healthy" &&
                    tickersData.tickers.length > 0
                ) {
                    const topStocks = tickersData.tickers.slice(0, 3);
                    const predictions = [];

                    for (const ticker of topStocks) {
                        try {
                            const prediction = await ApiService.predictStock(
                                ticker,
                                30,
                                "linear_regression"
                            );
                            predictions.push(prediction);
                        } catch (predError) {
                            console.log(
                                `Prediction failed for ${ticker}:`,
                                predError
                            );
                        }
                    }
                    setQuickPredictions(predictions);
                }
            } catch (predHealthError) {
                console.log(
                    "Prediction service not available:",
                    predHealthError
                );
            }
        } catch (err) {
            setError("Failed to load dashboard data");
            console.error("Dashboard error:", err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="text-center">
                <div className="loading-spinner"></div>
                <p>Loading market overview...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="card">
                <div className="text-center">
                    <h3 className="text-danger">⚠️ {error}</h3>
                    <button
                        onClick={loadDashboardData}
                        className="button-primary"
                    >
                        🔄 Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard">
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">📊 Market Overview</h2>
                </div>

                {stats && (
                    <div className="stats-grid">
                        <div className="stat-item">
                            <div className="stat-value">
                                {stats.total_tickers}
                            </div>
                            <div className="stat-label">Total Stocks</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">
                                {stats.total_sectors}
                            </div>
                            <div className="stat-label">Sectors</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">
                                {stats.date_range?.start}
                            </div>
                            <div className="stat-label">Data From</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">
                                {stats.date_range?.end}
                            </div>
                            <div className="stat-label">Data Until</div>
                        </div>
                    </div>
                )}
            </div>

            <div className="dashboard-grid">
                <div className="card">
                    <div className="card-header">
                        <div
                            style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                            }}
                        >
                            <h3 className="card-title">
                                🌪️ Most Volatile Stocks (30 days)
                            </h3>
                            <ExplainButton
                                chartData={null}
                                chartRef={null}
                                defaultQuestion="What does stock volatility mean? How should beginners understand volatile stocks?"
                                contextInfo={`These are the most volatile stocks in the market over the last 30 days. Volatility measures how much a stock's price moves up and down. High volatility can mean higher potential gains but also higher risk of losses.`}
                                size="small"
                            />
                        </div>
                    </div>
                    {volatileStocks.length > 0 ? (
                        <div className="volatile-stocks">
                            {volatileStocks.map((stock, index) => (
                                <div
                                    key={stock.ticker}
                                    className="volatile-stock-item"
                                >
                                    <div className="stock-rank">
                                        #{index + 1}
                                    </div>
                                    <div className="stock-info">
                                        <div className="stock-ticker">
                                            {stock.ticker}
                                        </div>
                                        <div className="stock-sector">
                                            {stock.sector}
                                        </div>
                                    </div>
                                    <div className="stock-metrics">
                                        <div className="volatility">
                                            Volatility:{" "}
                                            {(stock.volatility * 100).toFixed(
                                                2
                                            )}
                                            %
                                        </div>
                                        <div
                                            className={`price-change ${
                                                stock.price_change_pct >= 0
                                                    ? "positive"
                                                    : "negative"
                                            }`}
                                        >
                                            {stock.price_change_pct >= 0
                                                ? "+"
                                                : ""}
                                            {stock.price_change_pct.toFixed(2)}%
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p>No volatile stocks data available</p>
                    )}
                </div>

                {/* ML Prediction Service Status */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">🤖 ML Prediction Service</h3>
                    </div>
                    {predictionHealth ? (
                        <div className="prediction-status">
                            <div
                                className={`status-indicator ${
                                    predictionHealth.status === "healthy"
                                        ? "healthy"
                                        : "warning"
                                }`}
                            >
                                <span className="status-icon">
                                    {predictionHealth.status === "healthy"
                                        ? "✅"
                                        : "⚠️"}
                                </span>
                                <span className="status-text">
                                    {predictionHealth.status === "healthy"
                                        ? "Service Active"
                                        : "Service Issues"}
                                </span>
                            </div>
                            <div className="prediction-stats">
                                <div className="pred-stat">
                                    <span className="pred-label">Models:</span>
                                    <span className="pred-value">
                                        {predictionHealth.models_loaded}
                                    </span>
                                </div>
                                <div className="pred-stat">
                                    <span className="pred-label">
                                        Features:
                                    </span>
                                    <span className="pred-value">
                                        {predictionHealth
                                            .preprocessing_components
                                            .feature_names
                                            ? "✅"
                                            : "❌"}
                                    </span>
                                </div>
                                <div className="pred-stat">
                                    <span className="pred-label">Scaler:</span>
                                    <span className="pred-value">
                                        {predictionHealth
                                            .preprocessing_components.scaler
                                            ? "✅"
                                            : "❌"}
                                    </span>
                                </div>
                            </div>

                            {/* Quick Predictions */}
                            {quickPredictions.length > 0 && (
                                <div className="quick-predictions">
                                    <h4 className="quick-pred-title">
                                        🔮 Quick Predictions (30 days)
                                    </h4>
                                    <div className="predictions-grid">
                                        {quickPredictions.map((pred) => (
                                            <div
                                                key={pred.ticker}
                                                className="prediction-card"
                                            >
                                                <div className="pred-ticker">
                                                    {pred.ticker}
                                                </div>
                                                <div className="pred-prices">
                                                    <span className="current-price">
                                                        ৳
                                                        {pred.ticker_info.current_price.toFixed(
                                                            2
                                                        )}
                                                    </span>
                                                    <span className="arrow">
                                                        →
                                                    </span>
                                                    <span className="predicted-price">
                                                        ৳
                                                        {pred.ticker_info.predicted_price.toFixed(
                                                            2
                                                        )}
                                                    </span>
                                                </div>
                                                <div
                                                    className={`pred-change ${
                                                        pred.ticker_info
                                                            .price_difference_pct >=
                                                        0
                                                            ? "positive"
                                                            : "negative"
                                                    }`}
                                                >
                                                    {pred.ticker_info
                                                        .price_difference_pct >=
                                                    0
                                                        ? "+"
                                                        : ""}
                                                    {pred.ticker_info.price_difference_pct.toFixed(
                                                        1
                                                    )}
                                                    %
                                                </div>
                                                <div className="pred-trend">
                                                    {pred.trend_analysis.trend}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                    <div className="prediction-note">
                                        <small>
                                            Using Linear Regression model. Visit
                                            the Prediction tab for detailed
                                            analysis.
                                        </small>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="prediction-unavailable">
                            <span className="unavailable-icon">🚫</span>
                            <span>ML Prediction service unavailable</span>
                            <small>
                                Backend prediction service is not running
                            </small>
                        </div>
                    )}
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">🏢 Available Sectors</h3>
                    </div>
                    <div className="sectors-list">
                        {sectors.slice(0, 10).map((sector, index) => (
                            <div key={sector} className="sector-item">
                                <span className="sector-number">
                                    {index + 1}.
                                </span>
                                <span className="sector-name">{sector}</span>
                            </div>
                        ))}
                        {sectors.length > 10 && (
                            <div className="sector-item">
                                <span className="sector-more">
                                    ...and {sectors.length - 10} more
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">📈 Sample Stocks</h3>
                </div>
                <div className="tickers-grid">
                    {tickers.map((ticker) => (
                        <div key={ticker} className="ticker-badge">
                            {ticker}
                        </div>
                    ))}
                </div>
                <div className="text-center mb-2">
                    <p className="text-muted">
                        Showing sample of {tickers.length} stocks. Use the
                        Analysis tab to explore individual stocks.
                    </p>
                </div>
            </div>

            <style jsx>{`
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-bottom: 1rem;
                }

                .stat-item {
                    text-align: center;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                }

                .stat-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #667eea;
                    margin-bottom: 0.5rem;
                }

                .stat-label {
                    font-size: 0.9rem;
                    color: #6c757d;
                    font-weight: 500;
                }

                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 1.5rem;
                }

                .volatile-stocks {
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                }

                .volatile-stock-item {
                    display: flex;
                    align-items: center;
                    padding: 0.75rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                    border-left: 4px solid #667eea;
                }

                .stock-rank {
                    font-weight: 700;
                    font-size: 1.1rem;
                    color: #667eea;
                    margin-right: 1rem;
                    min-width: 30px;
                }

                .stock-info {
                    flex: 1;
                }

                .stock-ticker {
                    font-weight: 600;
                    font-size: 1rem;
                    color: #2c3e50;
                }

                .stock-sector {
                    font-size: 0.8rem;
                    color: #6c757d;
                }

                .stock-metrics {
                    text-align: right;
                }

                .volatility {
                    font-size: 0.8rem;
                    color: #6c757d;
                }

                .price-change {
                    font-weight: 600;
                    font-size: 0.9rem;
                }

                .price-change.positive {
                    color: #27ae60;
                }

                .price-change.negative {
                    color: #e74c3c;
                }

                .sectors-list {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }

                .sector-item {
                    display: flex;
                    align-items: center;
                    padding: 0.5rem;
                    background: #f8f9fa;
                    border-radius: 4px;
                }

                .sector-number {
                    color: #667eea;
                    font-weight: 600;
                    margin-right: 0.5rem;
                    min-width: 25px;
                }

                .sector-name {
                    color: #2c3e50;
                }

                .sector-more {
                    color: #6c757d;
                    font-style: italic;
                }

                .tickers-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-bottom: 1rem;
                }

                .ticker-badge {
                    background: #667eea;
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 15px;
                    font-size: 0.8rem;
                    font-weight: 500;
                }

                .text-muted {
                    color: #6c757d;
                    font-size: 0.9rem;
                }

                .prediction-status {
                    padding: 1rem;
                }

                .status-indicator {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    margin-bottom: 1rem;
                    font-weight: 600;
                }

                .status-indicator.healthy {
                    color: #27ae60;
                }

                .status-indicator.warning {
                    color: #f39c12;
                }

                .prediction-stats {
                    display: flex;
                    gap: 2rem;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                }

                .pred-stat {
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                    text-align: center;
                }

                .pred-label {
                    font-size: 0.8rem;
                    color: #6c757d;
                }

                .pred-value {
                    font-weight: 600;
                    color: #2c3e50;
                }

                .quick-predictions {
                    border-top: 1px solid #e9ecef;
                    padding-top: 1rem;
                }

                .quick-pred-title {
                    margin-bottom: 1rem;
                    color: #2c3e50;
                    font-size: 1.1rem;
                }

                .predictions-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-bottom: 1rem;
                }

                .prediction-card {
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                    text-align: center;
                }

                .pred-ticker {
                    font-weight: bold;
                    font-size: 1.1rem;
                    color: #2c3e50;
                    margin-bottom: 0.5rem;
                }

                .pred-prices {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                    margin-bottom: 0.5rem;
                    font-size: 0.9rem;
                }

                .current-price {
                    color: #6c757d;
                }

                .arrow {
                    color: #667eea;
                    font-weight: bold;
                }

                .predicted-price {
                    color: #2c3e50;
                    font-weight: 600;
                }

                .pred-change {
                    font-weight: 600;
                    font-size: 1rem;
                    margin-bottom: 0.25rem;
                }

                .pred-change.positive {
                    color: #27ae60;
                }

                .pred-change.negative {
                    color: #e74c3c;
                }

                .pred-trend {
                    font-size: 0.8rem;
                    color: #6c757d;
                    text-transform: uppercase;
                }

                .prediction-note {
                    text-align: center;
                    color: #6c757d;
                    padding: 0.5rem;
                    background: #f1f3f4;
                    border-radius: 4px;
                }

                .prediction-unavailable {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 2rem;
                    color: #6c757d;
                    text-align: center;
                }

                .unavailable-icon {
                    font-size: 2rem;
                }

                @media (max-width: 768px) {
                    .dashboard-grid {
                        grid-template-columns: 1fr;
                    }

                    .stats-grid {
                        grid-template-columns: repeat(2, 1fr);
                    }

                    .volatile-stock-item {
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 0.5rem;
                    }

                    .stock-metrics {
                        text-align: left;
                        width: 100%;
                    }
                }
            `}</style>
        </div>
    );
};

export default Dashboard;
