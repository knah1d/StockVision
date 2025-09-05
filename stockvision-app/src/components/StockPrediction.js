import React, { useState, useEffect, useRef } from "react";
import { Line, Bar } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from "chart.js";
import ApiService from "../services/apiService";
import ExplainButton from "./ExplainButton";

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const StockPrediction = () => {
    const [selectedTicker, setSelectedTicker] = useState("");
    const [days, setDays] = useState(180);
    const [selectedModel, setSelectedModel] = useState("linear_regression");
    const [tickers, setTickers] = useState([]);
    const [predictionData, setPredictionData] = useState(null);
    const [modelComparison, setModelComparison] = useState(null);
    const [recommendations, setRecommendations] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [predictionHealth, setPredictionHealth] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Chart refs
    const predictionChartRef = useRef(null);
    const comparisonChartRef = useRef(null);

    useEffect(() => {
        loadTickers();
        checkPredictionHealth();
        loadModelInfo();
    }, []);

    const loadTickers = async () => {
        try {
            const data = await ApiService.getTickers();
            setTickers(data.tickers);
            if (data.tickers.length > 0) {
                setSelectedTicker(data.tickers[0]);
            }
        } catch (err) {
            console.error("Error loading tickers:", err);
        }
    };

    const checkPredictionHealth = async () => {
        try {
            const health = await ApiService.getPredictionHealth();
            setPredictionHealth(health);
        } catch (err) {
            console.error("Error checking prediction health:", err);
        }
    };

    const loadModelInfo = async () => {
        try {
            const info = await ApiService.getModelInfo();
            setModelInfo(info);
            if (info.available_models.length > 0) {
                setSelectedModel(info.available_models[0]);
            }
        } catch (err) {
            console.error("Error loading model info:", err);
        }
    };

    const predictStock = async () => {
        if (!selectedTicker) return;

        try {
            setLoading(true);
            setError(null);

            // Get prediction
            const prediction = await ApiService.predictStock(
                selectedTicker,
                days,
                selectedModel
            );
            setPredictionData(prediction);

            // Get model comparison
            const comparison = await ApiService.compareModels(
                selectedTicker,
                days
            );
            setModelComparison(comparison);

            // Get recommendations
            const recs = await ApiService.getRecommendations(
                selectedTicker,
                days
            );
            setRecommendations(recs);
        } catch (err) {
            console.error("Prediction error:", err);

            // Check if it's an insufficient data error
            if (err.response && err.response.data && err.response.data.detail) {
                const errorDetail = err.response.data.detail;
                if (errorDetail.includes("Insufficient valid data")) {
                    setError(
                        "Not enough historical data for this stock with the selected time period. Try selecting a longer period (6 months or 1 year)."
                    );
                } else {
                    setError(`Prediction failed: ${errorDetail}`);
                }
            } else {
                setError(
                    "Failed to get stock prediction. Please try again or select a different stock."
                );
            }
        } finally {
            setLoading(false);
        }
    };

    // Chart data for prediction vs actual
    const predictionChartData = predictionData
        ? {
              labels: predictionData.chart_data.dates,
              datasets: [
                  {
                      label: "Actual Price",
                      data: predictionData.chart_data.actual_prices,
                      borderColor: "#667eea",
                      backgroundColor: "rgba(102, 126, 234, 0.1)",
                      tension: 0.1,
                  },
                  {
                      label: "Predicted Price",
                      data: predictionData.chart_data.predicted_prices,
                      borderColor: "#f093fb",
                      backgroundColor: "rgba(240, 147, 251, 0.1)",
                      borderDash: [5, 5],
                      tension: 0.1,
                  },
              ],
          }
        : null;

    // Chart data for model comparison
    const comparisonChartData = modelComparison
        ? {
              labels: Object.keys(modelComparison.comparison),
              datasets: [
                  {
                      label: "RMSE (Lower is Better)",
                      data: Object.values(modelComparison.comparison).map(
                          (m) => m.rmse || 0
                      ),
                      backgroundColor: [
                          "#667eea",
                          "#f093fb",
                          "#f39c12",
                          "#27ae60",
                          "#e74c3c",
                          "#9b59b6",
                      ],
                  },
              ],
          }
        : null;

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: "top",
            },
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: "Date",
                },
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: "Price (৳)",
                },
            },
        },
    };

    const barChartOptions = {
        responsive: true,
        plugins: {
            legend: {
                display: false,
            },
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: "Models",
                },
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: "RMSE",
                },
            },
        },
    };

    return (
        <div className="stock-prediction">
            {/* Service Status */}
            {predictionHealth && (
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">🤖 ML Prediction Service</h2>
                    </div>
                    <div className="service-status">
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
                                    ? "Service Healthy"
                                    : "Service Degraded"}
                            </span>
                        </div>
                        <div className="status-details">
                            <span>
                                Models Loaded: {predictionHealth.models_loaded}
                            </span>
                            <span>
                                Scaler:{" "}
                                {predictionHealth.preprocessing_components
                                    .scaler
                                    ? "✅"
                                    : "❌"}
                            </span>
                            <span>
                                Features:{" "}
                                {predictionHealth.preprocessing_components
                                    .feature_names
                                    ? "✅"
                                    : "❌"}
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* Prediction Controls */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">🔮 Stock Price Prediction</h2>
                </div>

                <div className="prediction-controls">
                    <div className="form-group">
                        <label className="form-label">Select Stock:</label>
                        <select
                            className="form-select"
                            value={selectedTicker}
                            onChange={(e) => setSelectedTicker(e.target.value)}
                        >
                            <option value="">Choose a stock...</option>
                            {tickers.map((ticker) => (
                                <option key={ticker} value={ticker}>
                                    {ticker}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Analysis Period:</label>
                        <select
                            className="form-select"
                            value={days}
                            onChange={(e) => setDays(parseInt(e.target.value))}
                        >
                            <option value={90}>Last 3 months</option>
                            <option value={180}>
                                Last 6 months (Recommended)
                            </option>
                            <option value={365}>Last 1 year</option>
                            <option value={500}>Last 500 days</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label className="form-label">ML Model:</label>
                        <select
                            className="form-select"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={true}
                        >
                            <option value="linear_regression">
                                LINEAR REGRESSION
                            </option>
                        </select>
                        <small className="form-help">
                            Using Linear Regression for fast and reliable
                            predictions
                        </small>
                    </div>

                    <button
                        className="button-primary"
                        onClick={predictStock}
                        disabled={
                            !selectedTicker ||
                            loading ||
                            predictionHealth?.status !== "healthy"
                        }
                    >
                        {loading ? (
                            <span>
                                <span className="loading-spinner-inline"></span>
                                🔄 Predicting...
                            </span>
                        ) : (
                            "🔮 Predict Price"
                        )}
                    </button>
                </div>
            </div>

            {error && (
                <div className="card">
                    <div className="error-container">
                        <div className="error-message">
                            <h3>⚠️ Prediction Error</h3>
                            <p>{error}</p>
                            {error.includes("Not enough historical data") && (
                                <div className="error-suggestions">
                                    <h4>💡 Suggestions:</h4>
                                    <ul>
                                        <li>
                                            Try selecting "Last 6 months" or
                                            "Last 1 year" from the analysis
                                            period
                                        </li>
                                        <li>
                                            Some stocks have limited data
                                            availability in our dataset
                                        </li>
                                        <li>
                                            Consider trying stocks like:
                                            SQURPHARMA, GP, BEXIMCO, or BATBC
                                        </li>
                                    </ul>
                                </div>
                            )}
                            <div className="error-actions">
                                <button
                                    onClick={() => {
                                        setError(null);
                                        setDays(365);
                                    }}
                                    className="button-secondary"
                                >
                                    📅 Try 1 Year Period
                                </button>
                                <button
                                    onClick={() => setError(null)}
                                    className="button-primary"
                                >
                                    🔄 Retry
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {!loading && !error && !predictionData && selectedTicker && (
                <div className="card">
                    <div className="help-container">
                        <h3>📈 Ready to Predict!</h3>
                        <p>
                            Click the <strong>"🔮 Predict Price"</strong> button
                            above to get ML-powered predictions for{" "}
                            <strong>{selectedTicker}</strong>.
                        </p>
                        <div className="help-tips">
                            <h4>💡 Tips for best results:</h4>
                            <ul>
                                <li>
                                    Use 6 months or 1 year periods for more
                                    reliable predictions
                                </li>
                                <li>
                                    Popular stocks like SQURPHARMA, GP, BEXIMCO
                                    have good data coverage
                                </li>
                                <li>
                                    Check that the ML service status shows
                                    "Service Active" above
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            )}

            {predictionData && (
                <>
                    {/* Prediction Results */}
                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title">
                                📊 {predictionData.ticker} Prediction Results
                            </h3>
                        </div>

                        <div className="prediction-explanation">
                            <h4>🤖 How Predictions Work</h4>
                            <div className="explanation-grid">
                                <div className="explanation-item">
                                    <span className="explanation-icon">📊</span>
                                    <div>
                                        <strong>Train-Test Split:</strong> Model
                                        trains on 80% historical data, predicts
                                        on recent 20%
                                    </div>
                                </div>
                                <div className="explanation-item">
                                    <span className="explanation-icon">🔮</span>
                                    <div>
                                        <strong>Future Prediction:</strong> Uses
                                        trend analysis to forecast next period
                                        price movement
                                    </div>
                                </div>
                                <div className="explanation-item">
                                    <span className="explanation-icon">⚖️</span>
                                    <div>
                                        <strong>Risk Control:</strong>{" "}
                                        Predictions limited to ±20% of current
                                        price for safety
                                    </div>
                                </div>
                                <div className="explanation-item">
                                    <span className="explanation-icon">📈</span>
                                    <div>
                                        <strong>Trend Analysis:</strong>{" "}
                                        {predictionData.trend_analysis.trend}{" "}
                                        trend detected with{" "}
                                        {predictionData.trend_analysis.trend_strength.toFixed(
                                            2
                                        )}
                                        % strength
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="prediction-info-grid">
                            <div className="info-item">
                                <div className="info-label">Current Price</div>
                                <div className="info-value">
                                    ৳
                                    {predictionData.ticker_info.current_price.toFixed(
                                        2
                                    )}
                                </div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">
                                    Predicted Price
                                </div>
                                <div className="info-value predicted">
                                    ৳
                                    {predictionData.ticker_info.predicted_price.toFixed(
                                        2
                                    )}
                                </div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">
                                    Expected Change
                                </div>
                                <div
                                    className={`info-value ${
                                        predictionData.ticker_info
                                            .price_difference_pct >= 0
                                            ? "positive"
                                            : "negative"
                                    }`}
                                >
                                    {predictionData.ticker_info
                                        .price_difference_pct >= 0
                                        ? "+"
                                        : ""}
                                    {predictionData.ticker_info.price_difference_pct.toFixed(
                                        2
                                    )}
                                    %
                                </div>
                                <div className="info-description">
                                    {Math.abs(
                                        predictionData.ticker_info
                                            .price_difference_pct
                                    ) < 0.5
                                        ? "Minimal movement expected"
                                        : Math.abs(
                                              predictionData.ticker_info
                                                  .price_difference_pct
                                          ) < 2
                                        ? "Moderate movement expected"
                                        : "Significant movement expected"}
                                </div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">Model Accuracy</div>
                                <div className="info-value">
                                    {predictionData.prediction_metrics.accuracy}
                                </div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">Trend</div>
                                <div className="info-value">
                                    {predictionData.trend_analysis.trend}
                                </div>
                            </div>
                            <div className="info-item">
                                <div className="info-label">Confidence</div>
                                <div className="info-value">
                                    {(
                                        predictionData.prediction_metrics
                                            .prediction_confidence * 100
                                    ).toFixed(0)}
                                    %
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Prediction Chart */}
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
                                    📈 Actual vs Predicted Prices
                                </h3>
                                <ExplainButton
                                    chartData={predictionChartData}
                                    chartRef={predictionChartRef}
                                    defaultQuestion="How accurate is this prediction? What do the differences between actual and predicted prices tell us?"
                                    contextInfo={`ML prediction for ${predictionData.ticker} using ${predictionData.model_used} model. Accuracy: ${predictionData.prediction_metrics.accuracy}, Trend: ${predictionData.trend_analysis.trend}`}
                                    size="small"
                                />
                            </div>
                        </div>
                        {predictionChartData && (
                            <div className="chart-container">
                                <Line
                                    ref={predictionChartRef}
                                    data={predictionChartData}
                                    options={chartOptions}
                                />
                            </div>
                        )}
                    </div>
                </>
            )}

            {/* Investment Recommendations */}
            {recommendations && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            💡 Investment Recommendation
                        </h3>
                    </div>
                    <div className="recommendation-container">
                        <div
                            className={`recommendation-badge ${recommendations.recommendation.toLowerCase()}`}
                        >
                            {recommendations.recommendation}
                        </div>
                        <div className="recommendation-details">
                            <div className="recommendation-reason">
                                <strong>Reason:</strong>{" "}
                                {recommendations.reason}
                            </div>
                            <div className="recommendation-metrics">
                                <span>
                                    Confidence:{" "}
                                    {recommendations.confidence_level}
                                </span>
                                <span>
                                    Risk Level: {recommendations.risk_level}
                                </span>
                                <span>
                                    Best Model:{" "}
                                    {recommendations.best_model_used
                                        .replace("_", " ")
                                        .toUpperCase()}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Model Comparison - Simplified for single model */}
            {modelComparison && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            📊 Linear Regression Performance
                        </h3>
                    </div>
                    <div className="model-metrics-simple">
                        {Object.entries(modelComparison.comparison).map(
                            ([modelName, metrics]) => (
                                <div
                                    key={modelName}
                                    className="performance-grid"
                                >
                                    <div className="metric-card">
                                        <div className="metric-label">
                                            Root Mean Square Error
                                        </div>
                                        <div className="metric-value">
                                            {metrics.rmse?.toFixed(2) || "N/A"}
                                        </div>
                                        <div className="metric-description">
                                            Lower values indicate better
                                            accuracy
                                        </div>
                                    </div>
                                    <div className="metric-card">
                                        <div className="metric-label">
                                            Mean Absolute Error
                                        </div>
                                        <div className="metric-value">
                                            {metrics.mae?.toFixed(2) || "N/A"}
                                        </div>
                                        <div className="metric-description">
                                            Average prediction error
                                        </div>
                                    </div>
                                    <div className="metric-card">
                                        <div className="metric-label">
                                            Prediction Accuracy
                                        </div>
                                        <div className="metric-value">
                                            {metrics.accuracy || "N/A"}
                                        </div>
                                        <div className="metric-description">
                                            Percentage of accurate predictions
                                        </div>
                                    </div>
                                    <div className="metric-card">
                                        <div className="metric-label">
                                            Predicted Trend
                                        </div>
                                        <div className="metric-value">
                                            {metrics.trend || "N/A"}
                                        </div>
                                        <div className="metric-description">
                                            Expected price direction
                                        </div>
                                    </div>
                                </div>
                            )
                        )}
                    </div>
                </div>
            )}

            {/* Model Information - Simplified */}
            {modelInfo && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">
                            🧠 Linear Regression Model
                        </h3>
                    </div>
                    <div className="model-info-simple">
                        <div className="info-section">
                            <h4>Why Linear Regression?</h4>
                            <ul>
                                <li>🚀 Fast predictions and training</li>
                                <li>📊 Easy to interpret results</li>
                                <li>💡 Good baseline for stock prediction</li>
                                <li>⚡ Lightweight and efficient</li>
                            </ul>
                        </div>
                        <div className="info-section">
                            <h4>Feature Engineering</h4>
                            <p>
                                Using {modelInfo.feature_count} engineered
                                features including:
                            </p>
                            <div className="feature-list">
                                <span className="feature-tag">
                                    Price Changes
                                </span>
                                <span className="feature-tag">
                                    Moving Averages
                                </span>
                                <span className="feature-tag">Volatility</span>
                                <span className="feature-tag">
                                    Volume Indicators
                                </span>
                                <span className="feature-tag">
                                    Technical Indicators
                                </span>
                                <span className="feature-tag">
                                    Date Features
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <style jsx>{`
                .prediction-controls {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr auto;
                    gap: 1rem;
                    align-items: end;
                }

                .prediction-info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                }

                .info-item {
                    text-align: center;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                }

                .info-value.predicted {
                    color: #9b59b6;
                    font-weight: bold;
                }

                .service-status {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                }

                .status-indicator {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-weight: 600;
                }

                .status-indicator.healthy {
                    color: #27ae60;
                }

                .status-indicator.warning {
                    color: #f39c12;
                }

                .status-details {
                    display: flex;
                    gap: 1rem;
                    font-size: 0.9rem;
                    color: #6c757d;
                }

                .recommendation-container {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                    padding: 1rem;
                }

                .recommendation-badge {
                    padding: 1rem 2rem;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 1.2rem;
                    text-align: center;
                }

                .recommendation-badge.buy {
                    background: #27ae60;
                    color: white;
                }

                .recommendation-badge.sell {
                    background: #e74c3c;
                    color: white;
                }

                .recommendation-badge.hold {
                    background: #f39c12;
                    color: white;
                }

                .recommendation-details {
                    flex: 1;
                }

                .recommendation-metrics {
                    display: flex;
                    gap: 1rem;
                    margin-top: 0.5rem;
                    font-size: 0.9rem;
                    color: #6c757d;
                }

                .prediction-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                }

                .model-metrics {
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                }

                .metric-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.75rem;
                    background: #f8f9fa;
                    border-radius: 6px;
                }

                .metric-row.best-model {
                    background: #e8f5e8;
                    border: 2px solid #27ae60;
                }

                .metric-name {
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .best-badge {
                    background: #27ae60;
                    color: white;
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.7rem;
                }

                .metric-values {
                    display: flex;
                    gap: 1rem;
                    font-size: 0.9rem;
                }

                .model-info {
                    display: flex;
                    flex-direction: column;
                    gap: 2rem;
                }

                .model-list,
                .feature-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-top: 0.5rem;
                }

                .model-tag,
                .feature-tag {
                    background: #667eea;
                    color: white;
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.8rem;
                }

                .feature-tag {
                    background: #f093fb;
                }

                .form-help {
                    display: block;
                    margin-top: 0.25rem;
                    font-size: 0.8rem;
                    color: #6c757d;
                }

                .model-metrics-simple {
                    padding: 1rem;
                }

                .performance-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                }

                .metric-card {
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #e9ecef;
                }

                .metric-card .metric-label {
                    font-size: 0.9rem;
                    color: #6c757d;
                    margin-bottom: 0.5rem;
                }

                .metric-card .metric-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #495057;
                    margin-bottom: 0.5rem;
                }

                .metric-card .metric-description {
                    font-size: 0.8rem;
                    color: #868e96;
                }

                .model-info-simple {
                    display: flex;
                    flex-direction: column;
                    gap: 2rem;
                }

                .model-info-simple ul {
                    list-style: none;
                    padding: 0;
                }

                .model-info-simple li {
                    padding: 0.5rem 0;
                    border-bottom: 1px solid #f1f3f4;
                }

                .model-info-simple li:last-child {
                    border-bottom: none;
                }

                .chart-container {
                    margin: 1rem 0;
                    height: 400px;
                }

                @media (max-width: 768px) {
                    .prediction-controls {
                        grid-template-columns: 1fr;
                    }

                    .prediction-grid {
                        grid-template-columns: 1fr;
                    }

                    .recommendation-container {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                }

                .error-container {
                    padding: 2rem;
                    text-align: center;
                    background: #fff5f5;
                    border-left: 4px solid #e74c3c;
                    border-radius: 6px;
                }

                .error-message h3 {
                    color: #e74c3c;
                    margin-bottom: 1rem;
                }

                .error-message p {
                    color: #2c3e50;
                    margin-bottom: 1.5rem;
                    font-size: 1rem;
                }

                .error-suggestions {
                    background: #f8f9fa;
                    padding: 1rem;
                    border-radius: 6px;
                    margin: 1rem 0;
                    text-align: left;
                }

                .error-suggestions h4 {
                    color: #667eea;
                    margin-bottom: 0.5rem;
                    font-size: 1rem;
                }

                .error-suggestions ul {
                    margin: 0;
                    padding-left: 1.5rem;
                    color: #2c3e50;
                }

                .error-suggestions li {
                    margin-bottom: 0.5rem;
                }

                .error-actions {
                    display: flex;
                    gap: 1rem;
                    justify-content: center;
                    margin-top: 1.5rem;
                }

                .button-secondary {
                    background: #6c757d;
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: background 0.2s ease;
                }

                .button-secondary:hover {
                    background: #5a6268;
                }

                .loading-spinner-inline {
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border: 2px solid #f3f3f3;
                    border-top: 2px solid #667eea;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-right: 0.5rem;
                }

                @keyframes spin {
                    0% {
                        transform: rotate(0deg);
                    }
                    100% {
                        transform: rotate(360deg);
                    }
                }

                .form-select:focus {
                    border-color: #667eea;
                    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
                    outline: none;
                }

                .button-primary:disabled {
                    background: #6c757d;
                    cursor: not-allowed;
                }

                .help-container {
                    padding: 2rem;
                    text-align: center;
                    background: #f8f9fa;
                    border-left: 4px solid #667eea;
                    border-radius: 6px;
                }

                .help-container h3 {
                    color: #667eea;
                    margin-bottom: 1rem;
                }

                .help-container p {
                    color: #2c3e50;
                    margin-bottom: 1.5rem;
                    font-size: 1rem;
                }

                .help-tips {
                    background: white;
                    padding: 1rem;
                    border-radius: 6px;
                    margin: 1rem 0;
                    text-align: left;
                }

                .help-tips h4 {
                    color: #667eea;
                    margin-bottom: 0.5rem;
                    font-size: 1rem;
                }

                .help-tips ul {
                    margin: 0;
                    padding-left: 1.5rem;
                    color: #2c3e50;
                }

                .help-tips li {
                    margin-bottom: 0.5rem;
                }

                .prediction-explanation {
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 6px;
                    margin-bottom: 1.5rem;
                }

                .prediction-explanation h4 {
                    color: #667eea;
                    margin-bottom: 1rem;
                    font-size: 1rem;
                }

                .explanation-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1rem;
                }

                .explanation-item {
                    display: flex;
                    align-items: flex-start;
                    gap: 0.75rem;
                    padding: 0.75rem;
                    background: white;
                    border-radius: 4px;
                    font-size: 0.9rem;
                }

                .explanation-icon {
                    font-size: 1.2rem;
                    flex-shrink: 0;
                }

                .info-description {
                    font-size: 0.8rem;
                    color: #6c757d;
                    margin-top: 0.25rem;
                    font-style: italic;
                }
            `}</style>
        </div>
    );
};

export default StockPrediction;
