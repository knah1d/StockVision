import React, { useState, useEffect, useRef } from 'react';
import { Line, Scatter } from 'react-chartjs-2';
import ApiService from '../services/apiService';

const StockComparison = () => {
  const [availableTickers, setAvailableTickers] = useState([]);
  const [selectedTickers, setSelectedTickers] = useState([]);
  const [days, setDays] = useState(90);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const lineChartRef = useRef(null);
  const scatterChartRef = useRef(null);

  useEffect(() => {
    loadTickers();
    
    // Cleanup function
    return () => {
      // Copy refs to variables to avoid stale closure warnings
      const lineChart = lineChartRef.current;
      const scatterChart = scatterChartRef.current;
      
      if (lineChart) {
        lineChart.destroy();
      }
      if (scatterChart) {
        scatterChart.destroy();
      }
    };
  }, []);

  const loadTickers = async () => {
    try {
      const data = await ApiService.getTickers();
      setAvailableTickers(data.tickers);
    } catch (err) {
      console.error('Error loading tickers:', err);
    }
  };

  const handleTickerSelection = (ticker) => {
    if (selectedTickers.includes(ticker)) {
      setSelectedTickers(selectedTickers.filter(t => t !== ticker));
    } else if (selectedTickers.length < 5) { // Limit to 5 stocks
      setSelectedTickers([...selectedTickers, ticker]);
    }
  };

  const compareStocks = async () => {
    if (selectedTickers.length < 2) return;

    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.compareTickers(selectedTickers, days);
      setComparisonData(data);
    } catch (err) {
      setError('Failed to compare stocks');
      console.error('Comparison error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Chart data for normalized performance
  const performanceChartData = comparisonData ? {
    labels: comparisonData.chart_data[selectedTickers[0]]?.dates || [],
    datasets: selectedTickers.map((ticker, index) => {
      const colors = ['#667eea', '#f39c12', '#27ae60', '#e74c3c', '#9b59b6'];
      return {
        label: ticker,
        data: comparisonData.chart_data[ticker]?.normalized_prices || [],
        borderColor: colors[index % colors.length],
        backgroundColor: 'transparent',
        tension: 0.1,
      };
    }),
  } : null;

  // Risk-Return scatter chart
  const riskReturnData = comparisonData ? {
    datasets: [{
      label: 'Risk vs Return',
      data: comparisonData.chart_data.risk_return?.map(item => ({
        x: item.risk * 100, // Convert to percentage
        y: item.return * 100, // Convert to percentage
        ticker: item.ticker,
      })) || [],
      backgroundColor: selectedTickers.map((_, index) => {
        const colors = ['#667eea', '#f39c12', '#27ae60', '#e74c3c', '#9b59b6'];
        return colors[index % colors.length];
      }),
      borderColor: selectedTickers.map((_, index) => {
        const colors = ['#667eea', '#f39c12', '#27ae60', '#e74c3c', '#9b59b6'];
        return colors[index % colors.length];
      }),
      pointRadius: 8,
    }],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.dataset.label === 'Risk vs Return') {
              const point = context.raw;
              return `${point.ticker}: Risk ${point.x.toFixed(2)}%, Return ${point.y.toFixed(2)}%`;
            }
            return `${context.dataset.label}: ${context.parsed.y}`;
          }
        }
      }
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
          text: 'Normalized Price (Base = 100)',
        },
      },
    },
  };

  const scatterOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = context.raw;
            return `${point.ticker}: Risk ${point.x.toFixed(2)}%, Return ${point.y.toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Risk (Annualized Volatility %)',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Return (Annualized %)',
        },
      },
    },
  };

  return (
    <div className="stock-comparison">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🔄 Stock Comparison</h2>
        </div>

        <div className="comparison-controls">
          <div className="form-group">
            <label className="form-label">Select Stocks to Compare (2-5 stocks):</label>
            <div className="ticker-selection">
              {availableTickers.slice(0, 20).map((ticker) => (
                <button
                  key={ticker}
                  className={`ticker-button ${selectedTickers.includes(ticker) ? 'selected' : ''}`}
                  onClick={() => handleTickerSelection(ticker)}
                  disabled={!selectedTickers.includes(ticker) && selectedTickers.length >= 5}
                >
                  {ticker}
                </button>
              ))}
            </div>
            <div className="selected-tickers">
              <strong>Selected: </strong>
              {selectedTickers.length > 0 ? selectedTickers.join(', ') : 'None'}
            </div>
          </div>

          <div className="comparison-settings">
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
              onClick={compareStocks}
              disabled={selectedTickers.length < 2 || loading}
            >
              {loading ? '🔄 Comparing...' : '📊 Compare Stocks'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="card">
          <div className="text-center text-danger">
            <h3>⚠️ {error}</h3>
            <button onClick={compareStocks} className="button-primary">
              🔄 Retry
            </button>
          </div>
        </div>
      )}

      {comparisonData && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">📊 Performance Comparison</h3>
            </div>
            <div className="comparison-summary">
              {comparisonData.comparison_summary.map((stock) => (
                <div key={stock.ticker} className="summary-item">
                  <div className="stock-header">
                    <div className="stock-ticker">{stock.ticker}</div>
                    <div className="stock-sector">{stock.sector}</div>
                  </div>
                  <div className="stock-metrics">
                    <div className="metric">
                      <span className="metric-label">Price:</span>
                      <span className="metric-value">৳{stock.current_price.toFixed(2)}</span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Change:</span>
                      <span className={`metric-value ${stock.price_change_pct >= 0 ? 'positive' : 'negative'}`}>
                        {stock.price_change_pct >= 0 ? '+' : ''}{stock.price_change_pct.toFixed(2)}%
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Volatility:</span>
                      <span className="metric-value">৳{stock.volatility.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="card-title">📈 Normalized Performance Chart</h3>
            </div>
            {performanceChartData && (
              <div className="chart-container">
                <Line 
                  ref={lineChartRef}
                  data={performanceChartData} 
                  options={chartOptions}
                  key={`line-chart-${selectedTickers.join('-')}-${days}`}
                />
              </div>
            )}
          </div>

          <div className="comparison-grid">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">🏆 Performance Ranking</h3>
              </div>
              <div className="ranking-list">
                {comparisonData.performance_ranking.map((item) => (
                  <div key={item.ticker} className="ranking-item">
                    <div className="ranking-position">
                      <span className="rank-emoji">{item.emoji}</span>
                      <span className="rank-number">#{item.rank}</span>
                    </div>
                    <div className="ranking-info">
                      <div className="ranking-ticker">{item.ticker}</div>
                      <div className={`ranking-change ${item.price_change_pct >= 0 ? 'positive' : 'negative'}`}>
                        {item.price_change_pct >= 0 ? '+' : ''}{item.price_change_pct.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title">⚖️ Risk vs Return Analysis</h3>
              </div>
              {riskReturnData && (
                <div className="chart-container">
                  <Scatter 
                    ref={scatterChartRef}
                    data={riskReturnData} 
                    options={scatterOptions}
                    key={`scatter-chart-${selectedTickers.join('-')}-${days}`}
                  />
                </div>
              )}
              <div className="risk-return-info">
                <p><strong>Ideal Position:</strong> High Return (top) + Low Risk (left)</p>
                <p><strong>Interpretation:</strong> Stocks in the top-left quadrant offer better risk-adjusted returns</p>
              </div>
            </div>
          </div>
        </>
      )}

      <style jsx>{`
        .comparison-controls {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .ticker-selection {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .ticker-button {
          background: #f8f9fa;
          border: 2px solid #dee2e6;
          color: #495057;
          padding: 0.5rem 1rem;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.3s ease;
          font-weight: 500;
        }

        .ticker-button:hover {
          background: #e9ecef;
          border-color: #adb5bd;
        }

        .ticker-button.selected {
          background: #667eea;
          border-color: #667eea;
          color: white;
        }

        .ticker-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .selected-tickers {
          color: #6c757d;
          font-size: 0.9rem;
          margin-top: 0.5rem;
        }

        .comparison-settings {
          display: flex;
          gap: 1rem;
          align-items: end;
        }

        .comparison-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem;
        }

        .summary-item {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 1rem;
          border-left: 4px solid #667eea;
        }

        .stock-header {
          margin-bottom: 0.75rem;
        }

        .stock-ticker {
          font-size: 1.2rem;
          font-weight: 700;
          color: #2c3e50;
        }

        .stock-sector {
          font-size: 0.9rem;
          color: #6c757d;
        }

        .stock-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .metric-label {
          font-weight: 500;
          color: #6c757d;
        }

        .metric-value {
          font-weight: 600;
          color: #2c3e50;
        }

        .metric-value.positive {
          color: #27ae60;
        }

        .metric-value.negative {
          color: #e74c3c;
        }

        .chart-container {
          margin: 1rem 0;
          height: 400px;
        }

        .comparison-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }

        .ranking-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .ranking-item {
          display: flex;
          align-items: center;
          padding: 0.75rem;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .ranking-position {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-right: 1rem;
          min-width: 60px;
        }

        .rank-emoji {
          font-size: 1.2rem;
        }

        .rank-number {
          font-weight: 600;
          color: #6c757d;
        }

        .ranking-info {
          flex: 1;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .ranking-ticker {
          font-weight: 600;
          color: #2c3e50;
        }

        .ranking-change.positive {
          color: #27ae60;
          font-weight: 600;
        }

        .ranking-change.negative {
          color: #e74c3c;
          font-weight: 600;
        }

        .risk-return-info {
          margin-top: 1rem;
          padding: 1rem;
          background: #f8f9fa;
          border-radius: 6px;
          font-size: 0.9rem;
          color: #6c757d;
        }

        @media (max-width: 768px) {
          .comparison-settings {
            flex-direction: column;
            align-items: stretch;
          }
          
          .comparison-grid {
            grid-template-columns: 1fr;
          }
          
          .comparison-summary {
            grid-template-columns: 1fr;
          }
          
          .ranking-info {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.25rem;
          }
        }
      `}</style>
    </div>
  );
};

export default StockComparison;
