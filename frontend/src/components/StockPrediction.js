import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import ApiService from '../services/apiService';

const StockPrediction = () => {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [days, setDays] = useState(90);
  const [tickers, setTickers] = useState([]);
  const [modelType, setModelType] = useState('linear_regression');
  const [predictionData, setPredictionData] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadTickers();
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
      console.error('Error loading tickers:', err);
    }
  };

  const loadModelInfo = async () => {
    try {
      const info = await ApiService.getModelInfo();
      setModelInfo(info);
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  };

  const predictStock = async () => {
    if (!selectedTicker) return;

    try {
      setLoading(true);
      setError(null);
      
      // Get prediction data using the selected model
      const data = await ApiService.predictStock(selectedTicker, days, modelType);
      setPredictionData(data);
      
      // Get recommendation data
      const recData = await ApiService.getRecommendation(selectedTicker, days);
      setRecommendation(recData);
    } catch (err) {
      setError('Failed to generate prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Chart configuration for predictions
  const predictionChartData = predictionData && {
    labels: predictionData.chart_data.dates,
    datasets: [
      {
        label: 'Actual Price',
        data: predictionData.chart_data.actual_prices,
        borderColor: '#4C51BF',
        backgroundColor: 'rgba(76, 81, 191, 0.1)',
        tension: 0.1,
        fill: false,
        pointRadius: 2,
      },
      {
        label: 'Predicted Price',
        data: predictionData.chart_data.predicted_prices,
        borderColor: '#ED64A6',
        backgroundColor: 'rgba(237, 100, 166, 0.1)',
        tension: 0.1,
        fill: false,
        borderDash: [5, 5],
        pointRadius: 2,
      }
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        mode: 'index',
        intersect: false,
      },
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${selectedTicker} - Price Prediction`,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Price (BDT)',
        },
        beginAtZero: false,
      },
    },
  };

  const getRecommendationColor = () => {
    if (!recommendation) return '';
    
    switch(recommendation.recommendation) {
      case 'BUY': return 'recommendation-buy';
      case 'SELL': return 'recommendation-sell';
      default: return 'recommendation-hold';
    }
  };

  return (
    <div className="stock-prediction">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🔮 Stock Price Prediction</h2>
          <div className="prediction-controls">
            <div className="control-group">
              <label className="form-label">Ticker:</label>
              <select 
                className="form-select" 
                value={selectedTicker}
                onChange={(e) => setSelectedTicker(e.target.value)}
              >
                {tickers.map(ticker => (
                  <option key={ticker} value={ticker}>{ticker}</option>
                ))}
              </select>
            </div>
            
            <div className="control-group">
              <label className="form-label">Historical Data:</label>
              <select
                className="form-select"
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value))}
              >
                <option value={30}>30 days</option>
                <option value={60}>60 days</option>
                <option value={90}>90 days</option>
                <option value={180}>180 days</option>
                <option value={365}>365 days</option>
              </select>
            </div>
            
            <div className="control-group">
              <label className="form-label">Prediction Model:</label>
              <select
                className="form-select"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                {modelInfo && modelInfo.available_models ? 
                  modelInfo.available_models.map(model => (
                    <option key={model} value={model}>
                      {model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </option>
                  )) 
                  : 
                  <option value="linear_regression">Linear Regression</option>
                }
              </select>
            </div>
            
            <button 
              className="button-primary"
              onClick={predictStock}
              disabled={loading || !selectedTicker}
            >
              {loading ? '⏳ Predicting...' : '🔮 Generate Prediction'}
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="card">
          <div className="text-center">
            <div className="loading-spinner"></div>
            <p>Generating prediction for {selectedTicker}...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="card">
          <div className="text-center text-danger">
            <h3>⚠️ {error}</h3>
            <p>Unable to generate prediction. The model may not have sufficient data for this ticker.</p>
            <button onClick={predictStock} className="button-primary">
              🔄 Try Again
            </button>
          </div>
        </div>
      )}

      {predictionData && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">📊 Price Prediction Chart</h3>
            </div>
            <div className="prediction-chart-container">
              <Line data={predictionChartData} options={chartOptions} />
            </div>
            <div className="prediction-summary">
              <div className="metric-card">
                <div className="metric-title">Current Price</div>
                <div className="metric-value">
                  ৳{predictionData.ticker_info.current_price.toFixed(2)}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-title">Predicted Price</div>
                <div className="metric-value">
                  ৳{predictionData.ticker_info.predicted_price.toFixed(2)}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-title">Price Change</div>
                <div className={`metric-value ${predictionData.ticker_info.price_difference_pct >= 0 ? 'positive' : 'negative'}`}>
                  {predictionData.ticker_info.price_difference_pct >= 0 ? '+' : ''}
                  {predictionData.ticker_info.price_difference_pct.toFixed(2)}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-title">Trend</div>
                <div className="metric-value">
                  {predictionData.trend_analysis.trend}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-title">Accuracy</div>
                <div className="metric-value">
                  {predictionData.prediction_metrics.accuracy}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-title">Model</div>
                <div className="metric-value model-name">
                  {predictionData.model_used}
                </div>
              </div>
            </div>
          </div>

          {recommendation && (
            <div className={`card recommendation-card ${getRecommendationColor()}`}>
              <div className="card-header">
                <h3 className="card-title">💡 Investment Recommendation</h3>
              </div>
              <div className="recommendation-content">
                <div className="recommendation-decision">
                  {recommendation.recommendation === 'BUY' && '🟢'}
                  {recommendation.recommendation === 'SELL' && '🔴'}
                  {recommendation.recommendation === 'HOLD' && '🟡'}
                  <span className="recommendation-text">{recommendation.recommendation}</span>
                </div>
                
                <div className="recommendation-reason">
                  <strong>Reason:</strong> {recommendation.reason}
                </div>
                
                <div className="recommendation-metrics">
                  <div className="rec-metric">
                    <span className="rec-label">Confidence:</span>
                    <span className="rec-value">{recommendation.confidence_level}</span>
                  </div>
                  
                  <div className="rec-metric">
                    <span className="rec-label">Risk Level:</span>
                    <span className="rec-value">{recommendation.risk_level}</span>
                  </div>
                  
                  <div className="rec-metric">
                    <span className="rec-label">Best Model:</span>
                    <span className="rec-value">{recommendation.best_model_used}</span>
                  </div>
                </div>
                
                <div className="recommendation-disclaimer">
                  <strong>Disclaimer:</strong> This is an automated recommendation based on machine learning models.
                  Always do your own research before making investment decisions.
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {modelInfo && (
        <div className="card model-info-card">
          <div className="card-header">
            <h3 className="card-title">🧠 Prediction Model Information</h3>
          </div>
          <div className="model-info-content">
            <div className="info-item">
              <span className="info-label">Available Models:</span>
              <span className="info-value">{modelInfo.available_models.join(', ')}</span>
            </div>
            
            <div className="info-item">
              <span className="info-label">Features Used:</span>
              <span className="info-value">{modelInfo.feature_count}</span>
            </div>
            
            <div className="info-item">
              <span className="info-label">System Status:</span>
              <span className={`info-value ${modelInfo.preprocessing_ready ? 'status-healthy' : 'status-error'}`}>
                {modelInfo.preprocessing_ready ? '✅ Ready' : '❌ Not Ready'}
              </span>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .prediction-controls {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 1rem;
        }

        .control-group {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .form-select {
          padding: 0.5rem;
          border-radius: 4px;
          border: 1px solid #e2e8f0;
        }

        .prediction-chart-container {
          height: 400px;
          padding: 1rem;
        }

        .prediction-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
          padding: 1rem;
          background-color: #f8fafc;
          border-top: 1px solid #e2e8f0;
        }

        .metric-card {
          background: white;
          border-radius: 8px;
          padding: 1rem;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          text-align: center;
        }

        .metric-title {
          font-size: 0.875rem;
          color: #64748b;
          margin-bottom: 0.5rem;
        }

        .metric-value {
          font-size: 1.25rem;
          font-weight: 600;
        }

        .metric-value.positive {
          color: #10b981;
        }

        .metric-value.negative {
          color: #ef4444;
        }

        .model-name {
          color: #6366f1;
          font-family: monospace;
        }

        .recommendation-card {
          border-left: 4px solid #cbd5e1;
        }

        .recommendation-buy {
          border-left-color: #10b981;
        }

        .recommendation-sell {
          border-left-color: #ef4444;
        }

        .recommendation-hold {
          border-left-color: #f59e0b;
        }

        .recommendation-content {
          padding: 1rem;
        }

        .recommendation-decision {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .recommendation-reason {
          margin-bottom: 1.5rem;
          line-height: 1.5;
        }

        .recommendation-metrics {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .rec-metric {
          background: #f8fafc;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          display: flex;
          flex-direction: column;
        }

        .rec-label {
          font-size: 0.75rem;
          color: #64748b;
        }

        .rec-value {
          font-weight: 600;
        }

        .recommendation-disclaimer {
          font-size: 0.75rem;
          color: #64748b;
          border-top: 1px solid #e2e8f0;
          padding-top: 1rem;
        }

        .model-info-card {
          margin-top: 2rem;
        }

        .model-info-content {
          padding: 1rem;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
        }

        .info-item {
          background: #f8fafc;
          padding: 0.75rem;
          border-radius: 4px;
          display: flex;
          flex-direction: column;
        }

        .info-label {
          font-size: 0.75rem;
          color: #64748b;
          margin-bottom: 0.25rem;
        }

        .info-value {
          font-weight: 600;
        }

        .status-healthy {
          color: #10b981;
        }

        .status-error {
          color: #ef4444;
        }

        @media (max-width: 768px) {
          .prediction-controls {
            flex-direction: column;
            align-items: stretch;
          }

          .control-group {
            flex-direction: column;
            align-items: stretch;
          }

          .recommendation-metrics {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
};

export default StockPrediction;