import React, { useState, useEffect, useRef } from 'react';
import { Bar, Doughnut } from 'react-chartjs-2';
import ApiService from '../services/apiService';
import ExplainButton from './ExplainButton';

const SectorAnalysis = () => {
  const [sectorData, setSectorData] = useState(null);
  const [days, setDays] = useState(90);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const barChartRef = useRef(null);
  const pieChartRef = useRef(null);

  useEffect(() => {
    loadSectorAnalysis();
    
    // Cleanup function
    return () => {
      // Copy refs to variables to avoid stale closure warnings
      const barChart = barChartRef.current;
      const pieChart = pieChartRef.current;
      
      if (barChart) {
        barChart.destroy();
      }
      if (pieChart) {
        pieChart.destroy();
      }
    };
  }, []);

  const loadSectorAnalysis = async (daysParam = days) => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getSectorOverview(daysParam);
      setSectorData(data);
    } catch (err) {
      setError('Failed to load sector analysis');
      console.error('Sector analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDaysChange = (newDays) => {
    setDays(newDays);
    loadSectorAnalysis();
  };

  // Sector performance bar chart
  const sectorPerformanceData = sectorData ? {
    labels: sectorData.sector_performance.map(item => item.sector),
    datasets: [{
      label: 'Average Return (%)',
      data: sectorData.sector_performance.map(item => item.avg_return * 100),
      backgroundColor: sectorData.sector_performance.map((_, index) => {
        const colors = ['#667eea', '#f39c12', '#27ae60', '#e74c3c', '#9b59b6', '#1abc9c', '#f1c40f', '#e67e22'];
        return colors[index % colors.length];
      }),
      borderColor: 'rgba(255, 255, 255, 0.8)',
      borderWidth: 2,
    }],
  } : null;

  // Market cap distribution pie chart
  const marketCapData = sectorData ? {
    labels: sectorData.sector_stats.map(item => item.sector),
    datasets: [{
      data: sectorData.sector_stats.map(item => item.total_market_cap),
      backgroundColor: [
        '#667eea', '#f39c12', '#27ae60', '#e74c3c', '#9b59b6', 
        '#1abc9c', '#f1c40f', '#e67e22', '#95a5a6', '#34495e'
      ],
      borderColor: '#fff',
      borderWidth: 2,
    }],
  } : null;

  const barChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Average Return (%)',
        },
      },
    },
  };

  const pieChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          generateLabels: function(chart) {
            const data = chart.data;
            if (data.labels.length && data.datasets.length) {
              return data.labels.map((label, i) => {
                const value = data.datasets[0].data[i];
                const percentage = ((value / data.datasets[0].data.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                return {
                  text: `${label} (${percentage}%)`,
                  fillStyle: data.datasets[0].backgroundColor[i],
                  strokeStyle: data.datasets[0].borderColor,
                  lineWidth: data.datasets[0].borderWidth,
                  index: i
                };
              });
            }
            return [];
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label;
            const value = context.parsed;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ৳${(value / 1000000).toFixed(1)}M (${percentage}%)`;
          }
        }
      }
    },
  };

  return (
    <div className="sector-analysis">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🏢 Sector Analysis</h2>
          <div className="sector-controls">
            <label className="form-label">Analysis Period:</label>
            <select
              className="form-select"
              value={days}
              onChange={(e) => handleDaysChange(parseInt(e.target.value))}
            >
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 3 months</option>
              <option value={180}>Last 6 months</option>
              <option value={365}>Last 1 year</option>
            </select>
          </div>
        </div>
      </div>

      {loading && (
        <div className="card">
          <div className="text-center">
            <h3>🔄 Loading sector analysis...</h3>
          </div>
        </div>
      )}

      {error && (
        <div className="card">
          <div className="text-center text-danger">
            <h3>⚠️ {error}</h3>
            <button onClick={loadSectorAnalysis} className="button-primary">
              🔄 Retry
            </button>
          </div>
        </div>
      )}

      {sectorData && (
        <>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">📊 Sector Performance Overview</h3>
            </div>
            <div className="sector-overview">
              {sectorData.sector_performance.map((sector, index) => (
                <div key={sector.sector} className="sector-card">
                  <div className="sector-icon">
                    {getSectorIcon(sector.sector)}
                  </div>
                  <div className="sector-info">
                    <div className="sector-name">{sector.sector}</div>
                    <div className={`sector-return ${sector.avg_return >= 0 ? 'positive' : 'negative'}`}>
                      {sector.avg_return >= 0 ? '+' : ''}{(sector.avg_return * 100).toFixed(2)}%
                    </div>
                    <div className="sector-stocks">{sector.stock_count} stocks</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="chart-grid">
            <div className="card">
              <div className="card-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="card-title">📈 Average Returns by Sector</h3>
                  <ExplainButton 
                    chartData={sectorPerformanceData}
                    chartRef={barChartRef}
                    defaultQuestion="What does this sector performance chart show? Which sectors should beginners consider?"
                    contextInfo={`This chart shows average returns for different market sectors over ${days} days. Each bar represents a sector's performance - higher bars indicate better returns. This helps investors understand which industries are performing well.`}
                    size="small"
                  />
                </div>
              </div>
              {sectorPerformanceData && (
                <div className="chart-container">
                  <Bar 
                    ref={barChartRef}
                    data={sectorPerformanceData} 
                    options={barChartOptions}
                    key={`bar-chart-${days}`}
                  />
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-header">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="card-title">💰 Market Cap Distribution</h3>
                  <ExplainButton 
                    chartData={marketCapData}
                    chartRef={pieChartRef}
                    defaultQuestion="What does this market cap distribution show? How important is market size when choosing stocks?"
                    contextInfo={`This pie chart shows how market capitalization (total value) is distributed across different sectors. Larger slices represent sectors with higher total market value. This helps understand which sectors dominate the market.`}
                    size="small"
                  />
                </div>
              </div>
              {marketCapData && (
                <div className="chart-container">
                  <Doughnut 
                    ref={pieChartRef}
                    data={marketCapData} 
                    options={pieChartOptions}
                    key={`doughnut-chart-${days}`}
                  />
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="card-title">🌟 Sector Statistics</h3>
            </div>
            <div className="sector-stats-grid">
              {sectorData.sector_stats.map((sector) => (
                <div key={sector.sector} className="stats-card">
                  <div className="stats-header">
                    <div className="stats-sector">{sector.sector}</div>
                    <div className="stats-icon">{getSectorIcon(sector.sector)}</div>
                  </div>
                  <div className="stats-metrics">
                    <div className="stats-metric">
                      <span className="metric-label">Companies:</span>
                      <span className="metric-value">{sector.company_count}</span>
                    </div>
                    <div className="stats-metric">
                      <span className="metric-label">Avg Market Cap:</span>
                      <span className="metric-value">৳{(sector.avg_market_cap / 1000000).toFixed(1)}M</span>
                    </div>
                    <div className="stats-metric">
                      <span className="metric-label">Total Market Cap:</span>
                      <span className="metric-value">৳{(sector.total_market_cap / 1000000).toFixed(1)}M</span>
                    </div>
                    <div className="stats-metric">
                      <span className="metric-label">Avg Volatility:</span>
                      <span className="metric-value">{(sector.avg_volatility * 100).toFixed(2)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="card-title">🏆 Top Performers by Sector</h3>
            </div>
            <div className="top-performers-grid">
              {sectorData.top_performers.map((sector) => (
                <div key={sector.sector} className="performers-card">
                  <div className="performers-header">
                    <h4>{sector.sector}</h4>
                    <span className="sector-emoji">{getSectorIcon(sector.sector)}</span>
                  </div>
                  <div className="performers-list">
                    {sector.top_stocks.map((stock, index) => (
                      <div key={stock.ticker} className="performer-item">
                        <div className="performer-rank">
                          <span className="rank-emoji">
                            {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                          </span>
                          <span className="performer-ticker">{stock.ticker}</span>
                        </div>
                        <div className={`performer-return ${stock.return >= 0 ? 'positive' : 'negative'}`}>
                          {stock.return >= 0 ? '+' : ''}{(stock.return * 100).toFixed(2)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <style jsx>{`
        .sector-controls {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .sector-overview {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1rem;
        }

        .sector-card {
          display: flex;
          align-items: center;
          gap: 1rem;
          background: #f8f9fa;
          border-radius: 8px;
          padding: 1rem;
          border-left: 4px solid #667eea;
        }

        .sector-icon {
          font-size: 2rem;
          min-width: 50px;
          text-align: center;
        }

        .sector-info {
          flex: 1;
        }

        .sector-name {
          font-weight: 600;
          color: #2c3e50;
          margin-bottom: 0.25rem;
        }

        .sector-return {
          font-size: 1.2rem;
          font-weight: 700;
          margin-bottom: 0.25rem;
        }

        .sector-return.positive {
          color: #27ae60;
        }

        .sector-return.negative {
          color: #e74c3c;
        }

        .sector-stocks {
          font-size: 0.9rem;
          color: #6c757d;
        }

        .chart-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: 1.5rem;
        }

        .chart-container {
          margin: 1rem 0;
          height: 400px;
        }

        .sector-stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem;
        }

        .stats-card {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid #dee2e6;
        }

        .stats-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .stats-sector {
          font-weight: 600;
          color: #2c3e50;
          font-size: 1.1rem;
        }

        .stats-icon {
          font-size: 1.5rem;
        }

        .stats-metrics {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .stats-metric {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .metric-label {
          color: #6c757d;
          font-size: 0.9rem;
        }

        .metric-value {
          font-weight: 600;
          color: #2c3e50;
        }

        .top-performers-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem;
        }

        .performers-card {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid #dee2e6;
        }

        .performers-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
          padding-bottom: 0.5rem;
          border-bottom: 1px solid #dee2e6;
        }

        .performers-header h4 {
          margin: 0;
          color: #2c3e50;
        }

        .sector-emoji {
          font-size: 1.5rem;
        }

        .performers-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .performer-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem;
          background: white;
          border-radius: 6px;
        }

        .performer-rank {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .rank-emoji {
          font-size: 1.2rem;
        }

        .performer-ticker {
          font-weight: 600;
          color: #2c3e50;
        }

        .performer-return {
          font-weight: 600;
        }

        .performer-return.positive {
          color: #27ae60;
        }

        .performer-return.negative {
          color: #e74c3c;
        }

        @media (max-width: 768px) {
          .sector-controls {
            flex-direction: column;
            align-items: stretch;
          }
          
          .chart-grid {
            grid-template-columns: 1fr;
          }
          
          .sector-overview {
            grid-template-columns: 1fr;
          }
          
          .sector-stats-grid {
            grid-template-columns: 1fr;
          }
          
          .top-performers-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

// Helper function to get sector icons
const getSectorIcon = (sector) => {
  const sectorIcons = {
    'Bank': '🏦',
    'Pharmaceutical': '💊',
    'Food & Beverage': '🍽️',
    'Textile': '👕',
    'Cement': '🏗️',
    'Insurance': '🛡️',
    'Engineering': '⚙️',
    'IT': '💻',
    'Telecoms': '📡',
    'Power': '⚡',
    'Paper': '📄',
    'Ceramic': '🏺',
    'Travel': '✈️',
    'Miscellaneous': '📦',
    'Fuel & Power': '⛽',
    'Jute': '🌾',
    'Tannery': '👜',
    'Services': '🔧',
    'Financial Institution': '💰',
    'Mutual Fund': '📈'
  };
  
  return sectorIcons[sector] || '🏢';
};

export default SectorAnalysis;
