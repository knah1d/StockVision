# FastAPI Backend for Stock Analysis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from controllers.analysis_controller import router as analysis_router
from controllers.llm_controller import router as llm_router

# Create FastAPI app
app = FastAPI(
    title="StockVision API",
    description="Stock Analysis API for ticker-wise analysis, comparison, and sector insights",
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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
        From StockVision Backend
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
