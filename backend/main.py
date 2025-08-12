# FastAPI Backend for Stock Analysis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from controllers.analysis_controller import router as analysis_router

# Create FastAPI app
app = FastAPI(
    title="StockVision API",
    description="Stock Analysis API for ticker-wise analysis, comparison, and sector insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>StockVision API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { color: #27ae60; font-weight: bold; }
            .post { color: #e74c3c; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .example { background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 StockVision API</h1>
            <p>Welcome to the StockVision Stock Analysis API! This API provides comprehensive stock analysis capabilities including individual ticker analysis, multi-ticker comparisons, sector analysis, and volatility insights.</p>
            
            <h2>📊 Available Endpoints</h2>
            
            <div class="endpoint">
                <strong class="method">GET</strong> <code>/api/analysis/tickers</code>
                <p>Get available tickers and sectors</p>
            </div>
            
            <div class="endpoint">
                <strong class="method post">POST</strong> <code>/api/analysis/ticker</code>
                <p>Comprehensive analysis of a single ticker</p>
                <div class="example">{"ticker": "ABBANK", "days": 90}</div>
            </div>
            
            <div class="endpoint">
                <strong class="method post">POST</strong> <code>/api/analysis/compare</code>
                <p>Compare multiple tickers</p>
                <div class="example">{"tickers": ["ABBANK", "ACI", "ACFL"], "days": 90}</div>
            </div>
            
            <div class="endpoint">
                <strong class="method post">POST</strong> <code>/api/analysis/sector</code>
                <p>Analyze sector performance</p>
                <div class="example">{"sector": "Bank", "days": 90, "top_n": 10}</div>
            </div>
            
            <div class="endpoint">
                <strong class="method post">POST</strong> <code>/api/analysis/volatility</code>
                <p>Find most volatile stocks</p>
                <div class="example">{"sector": "IT Sector", "days": 90, "top_n": 10}</div>
            </div>
            
            <div class="endpoint">
                <strong class="method">GET</strong> <code>/api/analysis/health</code>
                <p>API health check</p>
            </div>
            
            <div class="endpoint">
                <strong class="method">GET</strong> <code>/api/analysis/stats</code>
                <p>Get API statistics</p>
            </div>
            
            <h2>📚 Documentation</h2>
            <p>
                <a href="/docs">🔗 Interactive API Docs (Swagger)</a><br>
                <a href="/redoc">📖 Alternative API Docs (ReDoc)</a>
            </p>
            
            <h2>🛠️ Quick Start</h2>
            <p>1. Get available tickers: <code>GET /api/analysis/tickers</code></p>
            <p>2. Analyze a ticker: <code>POST /api/analysis/ticker</code></p>
            <p>3. Use the chart data for visualization in your frontend</p>
            
            <h2>💡 Features</h2>
            <ul>
                <li>📈 Individual ticker analysis with technical indicators</li>
                <li>🔄 Multi-ticker performance comparison</li>
                <li>🏢 Sector-wise analysis and rankings</li>
                <li>🌪️ Volatility analysis and risk metrics</li>
                <li>📊 Chart-ready data for frontend visualization</li>
                <li>⚡ Fast API responses with caching</li>
            </ul>
        </div>
    </body>
    </html>
    """

# Initialize services on startup
# @app.on_event("startup")
# async def startup_event():
#     """Initialize services on startup"""
#     print("🚀 Starting StockVision API...")
#     print("📊 Loading stock data...")
#     # Services will be initialized on first request due to lazy loading
#     print("✅ StockVision API ready!")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     print("👋 Shutting down StockVision API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001,  # Changed port to avoid conflicts
        reload=True,
        log_level="info"
    )
