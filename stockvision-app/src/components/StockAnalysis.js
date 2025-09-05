import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar } from 'react-chartjs-2';
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
} from 'chart.js';
import ApiService from '../services/apiService';
import ExplainButton from './ExplainButton';

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

const StockAnalysis = () => {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [days, setDays] = useState(90);
  const [tickers, setTickers] = useState([]);
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Chart refs for capturing images
  const priceChartRef = useRef(null);
  const volumeChartRef = useRef(null);

  useEffect(() => {
    loadTickers();
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

  const analyzeStock = async () => {
    if (!selectedTicker) return;

    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.analyzeTicker(selectedTicker, days);
      setAnalysisData(data);
    } catch (err) {
      setError('Failed to analyze stock');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Chart configurations
  const priceChartData = analysisData ? {
    labels: analysisData.chart_data.dates,
    datasets: [
      {
        label: 'Price',
        data: analysisData.chart_data.prices,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        tension: 0.1,
      },
      {
        label: 'MA5',
        data: analysisData.chart_data.ma5,
        borderColor: '#f39c12',
        backgroundColor: 'transparent',
        tension: 0.1,
      },
      {
        label: 'MA20',
        data: analysisData.chart_data.ma20,
        borderColor: '#27ae60',
        backgroundColor: 'transparent',
        tension: 0.1,
      },
    ],
  } : null;

  const volumeChartData = analysisData ? {
    labels: analysisData.chart_data.dates,
    datasets: [
      {
        label: 'Volume',
        data: analysisData.chart_data.volumes,
        backgroundColor: 'rgba(155, 89, 182, 0.6)',
        borderColor: '#9b59b6',
        borderWidth: 1,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };

  return (
    <div className="stock-analysis">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🔍 Stock Analysis</h2>
        </div>

        <div className="analysis-controls">
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
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 3 months</option>
              <option value={180}>Last 6 months</option>
              <option value={365}>Last 1 year</option>
            </select>
          </div>

          <button
            className="button-primary"
            onClick={analyzeStock}
            disabled={!selectedTicker || loading}
          >
            {loading ? '🔄 Analyzing...' : '📊 Analyze Stock'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card">
          <div className="text-center text-danger">
            <h3>⚠️ {error}</h3>
            <button onClick={analyzeStock} className="button-primary">
              🔄 Retry
            </button>
          </div>
        </div>
      )}

      {analysisData && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">📊 {analysisData.ticker_info.ticker} Analysis</h3>
            </div>

            <div className="stock-info-grid">
              <div className="info-item">
                <div className="info-label">Sector</div>
                <div className="info-value">{analysisData.ticker_info.sector}</div>
              </div>
              <div className="info-item">
                <div className="info-label">Current Price</div>
                <div className="info-value">৳{analysisData.ticker_info.current_price.toFixed(2)}</div>
              </div>
              <div className="info-item">
                <div className="info-label">Price Change</div>
                <div className={`info-value ${analysisData.ticker_info.price_change_pct >= 0 ? 'positive' : 'negative'}`}>
                  {analysisData.ticker_info.price_change_pct >= 0 ? '+' : ''}{analysisData.ticker_info.price_change_pct.toFixed(2)}%
                </div>
              </div>
              <div className="info-item">
                <div className="info-label">Volatility</div>
                <div className="info-value">৳{analysisData.ticker_info.volatility.toFixed(2)}</div>
              </div>
              <div className="info-item">
                <div className="info-label">Avg Volume</div>
                <div className="info-value">{analysisData.ticker_info.avg_volume.toLocaleString()}</div>
              </div>
              <div className="info-item">
                <div className="info-label">Date Range</div>
                <div className="info-value">{analysisData.ticker_info.date_range}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 className="card-title">📈 Price Chart with Moving Averages</h3>
                <ExplainButton 
                  chartData={priceChartData}
                  chartRef={priceChartRef}
                  defaultQuestion="Can you explain this price chart? What do the moving averages tell us about the stock's trend?"
                  contextInfo={`This is a price chart for ${analysisData.ticker_info.ticker} (${analysisData.ticker_info.sector} sector) showing ${days} days of data. Current price: ৳${analysisData.ticker_info.current_price.toFixed(2)}, Price change: ${analysisData.ticker_info.price_change_pct.toFixed(2)}%`}
                  size="small"
                />
              </div>
            </div>
            {priceChartData && (
              <div className="chart-container">
                <Line ref={priceChartRef} data={priceChartData} options={chartOptions} />
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 className="card-title">📊 Trading Volume</h3>
                <ExplainButton 
                  chartData={volumeChartData}
                  chartRef={volumeChartRef}
                  defaultQuestion="What does this volume chart show? How should beginners interpret trading volume?"
                  contextInfo={`This shows trading volume for ${analysisData.ticker_info.ticker}. Average volume: ${analysisData.ticker_info.avg_volume.toLocaleString()}. Volume can indicate interest and momentum in the stock.`}
                  size="small"
                />
              </div>
            </div>
            {volumeChartData && (
              <div className="chart-container">
                <Bar ref={volumeChartRef} data={volumeChartData} options={chartOptions} />
              </div>
            )}
          </div>

          <div className="analysis-grid">
            <div className="card">
              <div className="card-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="card-title">🔧 Technical Indicators</h3>
                  <ExplainButton 
                    chartData={null}
                    chartRef={null}
                    defaultQuestion="Can you explain these technical indicators? What do moving averages (MA5, MA20) mean for stock analysis?"
                    contextInfo={`Technical indicators for ${analysisData.ticker_info.ticker}: Current vs MA5: ${analysisData.technical_indicators.current_vs_ma5}, Current vs MA20: ${analysisData.technical_indicators.current_vs_ma20}. These help identify trends and potential buy/sell signals.`}
                    size="small"
                  />
                </div>
              </div>
              <div className="technical-indicators">
                <div className="indicator-item">
                  <span className="indicator-label">Current vs MA5:</span>
                  <span className={`indicator-value ${analysisData.technical_indicators.current_vs_ma5 === 'Above' ? 'positive' : 'negative'}`}>
                    {analysisData.technical_indicators.current_vs_ma5}
                    {analysisData.technical_indicators.ma5_value && 
                      ` (৳${analysisData.technical_indicators.ma5_value.toFixed(2)})`
                    }
                  </span>
                </div>
                <div className="indicator-item">
                  <span className="indicator-label">Current vs MA20:</span>
                  <span className={`indicator-value ${analysisData.technical_indicators.current_vs_ma20 === 'Above' ? 'positive' : 'negative'}`}>
                    {analysisData.technical_indicators.current_vs_ma20}
                    {analysisData.technical_indicators.ma20_value && 
                      ` (৳${analysisData.technical_indicators.ma20_value.toFixed(2)})`
                    }
                  </span>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="card-title">⚖️ Risk Metrics</h3>
                  <ExplainButton 
                    chartData={null}
                    chartRef={null}
                    defaultQuestion="What do these risk metrics mean? How should beginners understand volatility and Sharpe ratio?"
                    contextInfo={`Risk metrics for ${analysisData.ticker_info.ticker}: Daily volatility: ${(analysisData.risk_metrics.daily_volatility * 100).toFixed(2)}%, Sharpe ratio: ${analysisData.risk_metrics.sharpe_ratio.toFixed(4)}. Best day: ${(analysisData.risk_metrics.best_return * 100).toFixed(2)}%, Worst day: ${(analysisData.risk_metrics.worst_return * 100).toFixed(2)}%.`}
                    size="small"
                  />
                </div>
              </div>
              <div className="risk-metrics">
                <div className="metric-item">
                  <span className="metric-label">Daily Volatility:</span>
                  <span className="metric-value">{(analysisData.risk_metrics.daily_volatility * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Annual Volatility:</span>
                  <span className="metric-value">{(analysisData.risk_metrics.annualized_volatility * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Sharpe Ratio:</span>
                  <span className="metric-value">{analysisData.risk_metrics.sharpe_ratio.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Best Day:</span>
                  <span className="metric-value text-success">
                    {analysisData.risk_metrics.best_day} ({(analysisData.risk_metrics.best_return * 100).toFixed(2)}%)
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Worst Day:</span>
                  <span className="metric-value text-danger">
                    {analysisData.risk_metrics.worst_day} ({(analysisData.risk_metrics.worst_return * 100).toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <style jsx>{`
        .analysis-controls {
          display: grid;
          grid-template-columns: 1fr 1fr auto;
          gap: 1rem;
          align-items: end;
        }

        .stock-info-grid {
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

        .info-label {
          font-size: 0.9rem;
          color: #6c757d;
          margin-bottom: 0.25rem;
        }

        .info-value {
          font-size: 1.1rem;
          font-weight: 600;
          color: #2c3e50;
        }

        .info-value.positive {
          color: #27ae60;
        }

        .info-value.negative {
          color: #e74c3c;
        }

        .chart-container {
          margin: 1rem 0;
          height: 400px;
        }

        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }

        .technical-indicators, .risk-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .indicator-item, .metric-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem;
          background: #f8f9fa;
          border-radius: 4px;
        }

        .indicator-label, .metric-label {
          font-weight: 500;
          color: #2c3e50;
        }

        .indicator-value.positive {
          color: #27ae60;
          font-weight: 600;
        }

        .indicator-value.negative {
          color: #e74c3c;
          font-weight: 600;
        }

        .metric-value {
          font-weight: 600;
          color: #2c3e50;
        }

        @media (max-width: 768px) {
          .analysis-controls {
            grid-template-columns: 1fr;
          }
          
          .analysis-grid {
            grid-template-columns: 1fr;
          }
          
          .indicator-item, .metric-item {
            flex-direction: column;
            text-align: center;
            gap: 0.25rem;
          }
        }
      `}</style>
    </div>
  );
};

export default StockAnalysis;
