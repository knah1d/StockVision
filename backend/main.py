# FastAPI Backend for Stock Analysis
import sys
import os

# Add the parent directory to Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from controllers.analysis_controller import router as analysis_router
from controllers.llm_controller import router as llm_router

# Create FastAPI app
app = FastAPI(
    title="StockVision API",
    description="Stock Analysis API for ticker-wise analysis, comparison, sector insights, and ML predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router)
app.include_router(llm_router)

# Prediction endpoints
from controllers.prediction_controller import prediction_controller
from fastapi import HTTPException, Query
from typing import List

# Specific routes must come before parameterized routes
@app.get("/api/v1/predict/health")
async def get_prediction_health():
    """Get health status of the prediction service"""
    return await prediction_controller.get_health_status()

@app.get("/api/v1/predict/models/info")
async def get_model_info():
    """Get information about available models and features"""
    return await prediction_controller.get_model_info()

@app.post("/api/v1/predict/multiple")
async def predict_multiple_stocks(
    tickers: List[str],
    days: int = Query(default=90, ge=1, le=1000, description="Number of days of historical data"),
    model: str = Query(default="linear_regression", description="ML model to use for prediction")
):
    """Predict stock prices for multiple tickers"""
    return await prediction_controller.predict_multiple(tickers, days, model)

# Parameterized routes come after specific routes
@app.get("/api/v1/predict/{ticker}")
async def predict_stock(
    ticker: str,
    days: int = Query(default=90, ge=1, le=1000, description="Number of days of historical data"),
    model: str = Query(default="linear_regression", description="ML model to use for prediction")
):
    """Predict stock price for a single ticker"""
    return await prediction_controller.predict_stock(ticker, days, model)

@app.get("/api/v1/predict/{ticker}/compare")
async def compare_models(
    ticker: str,
    days: int = Query(default=90, ge=1, le=1000, description="Number of days of historical data")
):
    """Compare all available models for a ticker"""
    return await prediction_controller.compare_models(ticker, days)

@app.get("/api/v1/predict/{ticker}/recommend")
async def get_investment_recommendations(
    ticker: str,
    days: int = Query(default=90, ge=1, le=1000, description="Number of days of historical data")
):
    """Get investment recommendations for a ticker"""
    return await prediction_controller.get_recommendations(ticker, days)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <html>
        <head>
            <title>StockVision API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { color: #007bff; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🚀 StockVision API</h1>
            <p>Advanced Stock Analysis and Prediction API</p>
            
            <h2>📊 Analysis Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/analysis/ticker/{ticker} - Analyze a single stock
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/analysis/compare - Compare multiple stocks
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/analysis/sector/{sector} - Analyze a sector
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/analysis/volatile - Find volatile stocks
            </div>
            
            <h2>🤖 ML Prediction Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/predict/{ticker} - Predict stock price
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/predict/{ticker}/compare - Compare prediction models
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/predict/multiple - Predict multiple stocks
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/predict/{ticker}/recommend - Get investment recommendations
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/predict/models/info - Model information
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/v1/predict/health - Prediction service health
            </div>
            
            <h2>🧠 LLM Endpoints</h2>
            <div class="endpoint">
                <span class="method">POST</span> /api/v1/llm/explain - Get AI explanations
            </div>
            
            <p><a href="/docs">📖 Interactive API Documentation</a></p>
            <p><a href="/redoc">📘 ReDoc Documentation</a></p>
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
        port=8000,  # Changed port to avoid conflicts
        reload=True,
        log_level="info"
    )
