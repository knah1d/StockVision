# 🚀 StockVision FastAPI Backend

A powerful and simple FastAPI backend for stock market analysis, extracted from Jupyter notebook analysis functions.

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── start_server.sh        # Startup script
├── controllers/
│   ├── __init__.py
│   └── analysis_controller.py  # API endpoints
├── services/
│   ├── __init__.py
│   ├── data_service.py     # Data loading and basic operations
│   └── analysis_service.py # Complex analysis operations
└── models/
    ├── __init__.py
    └── schemas.py          # Pydantic models for API requests/responses
```

## 🎯 Features

- **📈 Single Ticker Analysis**: Comprehensive analysis with technical indicators
- **🔄 Multi-Ticker Comparison**: Compare multiple stocks side by side  
- **🏢 Sector Analysis**: Analyze performance within specific sectors
- **🌪️ Volatility Analysis**: Find most volatile stocks
- **📊 Chart-Ready Data**: All responses include data formatted for frontend charts
- **⚡ Fast & Simple**: Clean, organized code structure

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Server
```bash
# Using the startup script
./start_server.sh

# Or directly with Python
python main.py
```

### 3. Access the API
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📚 API Endpoints

### Get Available Tickers
```http
GET /api/analysis/tickers?sector=Bank&limit=20
```

### Analyze Single Ticker
```http
POST /api/analysis/ticker
Content-Type: application/json

{
  "ticker": "ABBANK",
  "days": 90
}
```

### Compare Multiple Tickers
```http
POST /api/analysis/compare
Content-Type: application/json

{
  "tickers": ["ABBANK", "ACI", "ACFL"],
  "days": 90
}
```

### Analyze Sector
```http
POST /api/analysis/sector
Content-Type: application/json

{
  "sector": "Bank",
  "days": 90,
  "top_n": 10
}
```

### Find Volatile Stocks
```http
POST /api/analysis/volatility
Content-Type: application/json

{
  "sector": "IT Sector",
  "days": 90,
  "top_n": 10
}
```

## 📊 Response Format

All analysis endpoints return structured data including:

- **Basic Info**: Ticker details, price, volume, volatility
- **Technical Indicators**: Moving averages, trend analysis
- **Risk Metrics**: Sharpe ratio, volatility, best/worst days
- **Chart Data**: Ready-to-use data for frontend visualization

### Example Response
```json
{
  "ticker_info": {
    "ticker": "ABBANK",
    "sector": "Bank",
    "current_price": 45.50,
    "price_change_pct": 12.5,
    "volatility": 2.3
  },
  "technical_indicators": {
    "current_vs_ma5": "Above",
    "current_vs_ma20": "Below",
    "ma5_value": 44.2,
    "ma20_value": 46.8
  },
  "risk_metrics": {
    "daily_volatility": 0.023,
    "sharpe_ratio": 1.45,
    "best_day": "2024-01-15"
  },
  "chart_data": {
    "dates": ["2024-01-01", "2024-01-02", ...],
    "prices": [42.1, 43.2, 44.5, ...],
    "volumes": [123456, 234567, ...],
    "ma5": [41.8, 42.5, ...]
  }
}
```

## 🛠️ Frontend Integration

The API provides chart-ready data that you can directly use with:

- **Chart.js**: For line charts, bar charts, scatter plots
- **D3.js**: For custom visualizations
- **Plotly.js**: For interactive charts
- **Any charting library**: Data is in standard array format

### Example Frontend Usage (JavaScript)
```javascript
// Fetch ticker analysis
const response = await fetch('http://localhost:8000/api/analysis/ticker', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ticker: 'ABBANK', days: 90})
});

const data = await response.json();

// Use chart data directly
const chartData = {
  labels: data.chart_data.dates,
  datasets: [{
    label: 'Price',
    data: data.chart_data.prices,
    borderColor: 'blue'
  }, {
    label: 'MA20',
    data: data.chart_data.ma20,
    borderColor: 'orange'
  }]
};
```

## 🔧 Development

### Project Philosophy
- **Keep it Simple**: Easy to understand and modify
- **Clear Separation**: Controllers handle HTTP, Services handle business logic
- **Type Safety**: Pydantic models for request/response validation
- **Scalable**: Easy to add new endpoints and features

### Adding New Features
1. **Add Request/Response models** in `models/schemas.py`
2. **Implement business logic** in appropriate service
3. **Create API endpoint** in `controllers/analysis_controller.py`
4. **Test with automatic docs** at `/docs`

### Code Structure
- **Controllers**: Handle HTTP requests, validation, error handling
- **Services**: Business logic, data processing, analysis
- **Models**: Data structures, request/response schemas
- **Main**: FastAPI app configuration, middleware, startup

## 📈 Performance

- **Lazy Loading**: Services initialize on first request
- **Efficient Data Processing**: Pandas operations optimized
- **Memory Management**: Data loaded once, reused across requests
- **Fast Response**: Typical response time < 500ms

## 🔒 Production Notes

- Update CORS origins in `main.py` for security
- Add authentication if needed
- Configure logging for production
- Consider adding Redis caching for large datasets
- Use environment variables for configuration

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install requirements with `pip install -r requirements.txt`
2. **File not found**: Ensure data files exist in `../data/processed/`
3. **CORS errors**: Check CORS configuration in `main.py`
4. **Port already in use**: Change port in `main.py` or kill existing process

### Support
Check the interactive docs at `/docs` for detailed API information and testing interface.

---

🎉 **Ready to build amazing stock analysis frontends!** 🎉
