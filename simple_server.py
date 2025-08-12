#!/usr/bin/env python3
"""
Simple test server for precomputed data
Bypasses import issues and serves instant page data
"""
import sys
import os
sys.path.append('/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import the precomputed service directly
from backend.services.precomputed_service import precomputed_service

# Create FastAPI app
app = FastAPI(
    title="StockVision Precomputed API",
    description="Instant page data API - no delays, no multiple calls",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StockVision Precomputed API - Instant Page Data",
        "version": "2.0.0",
        "endpoints": {
            "dashboard": "/api/precomputed/dashboard",
            "sectors": "/api/precomputed/sectors", 
            "sector_analysis": "/api/precomputed/sector-analysis",
            "tickers": "/api/precomputed/tickers",
            "volatile_stocks": "/api/precomputed/volatile-stocks",
            "market_overview": "/api/precomputed/market-overview",
            "stats": "/api/precomputed/stats",
            "all_data": "/api/precomputed/all"
        }
    }

# Precomputed data endpoints
@app.get("/api/precomputed/dashboard")
async def get_dashboard_data():
    """Get precomputed dashboard data - loads instantly"""
    try:
        return precomputed_service.get_data('dashboard')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")

@app.get("/api/precomputed/tickers")
async def get_tickers_data():
    """Get precomputed tickers data - loads instantly"""
    try:
        return precomputed_service.get_data('tickers')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tickers data: {str(e)}")

@app.get("/api/precomputed/sectors")
async def get_sectors_data():
    """Get precomputed sectors data - loads instantly"""
    try:
        return precomputed_service.get_data('sectors')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sectors data: {str(e)}")

@app.get("/api/precomputed/sector-analysis")
async def get_sector_analysis_data():
    """Get precomputed sector analysis data - loads instantly"""
    try:
        return precomputed_service.get_data('sector_analysis')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector analysis data: {str(e)}")

@app.get("/api/precomputed/market-overview")
async def get_market_overview_data():
    """Get precomputed market overview data - loads instantly"""
    try:
        return precomputed_service.get_data('market_overview')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market overview data: {str(e)}")

@app.get("/api/precomputed/volatile-stocks")
async def get_volatile_stocks_data():
    """Get precomputed volatile stocks data - loads instantly"""
    try:
        return precomputed_service.get_data('volatile_stocks')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting volatile stocks data: {str(e)}")

@app.get("/api/precomputed/stats")
async def get_basic_stats():
    """Get precomputed basic statistics - loads instantly"""
    try:
        return precomputed_service.get_data('basic_stats')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting basic stats: {str(e)}")

@app.get("/api/precomputed/all")
async def get_all_precomputed_data():
    """Get ALL precomputed data in one call - ultimate performance"""
    try:
        return precomputed_service.get_all_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting all data: {str(e)}")

@app.post("/api/precomputed/refresh")
async def refresh_precomputed_data():
    """Refresh all precomputed data"""
    try:
        precomputed_service.refresh_data()
        return {"success": True, "message": "All data refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

@app.get("/api/precomputed/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Quick check to see if data is available
        dashboard_data = precomputed_service.get_data('dashboard')
        has_data = 'total_tickers' in dashboard_data
        
        return {
            "status": "healthy" if has_data else "no_data",
            "data_available": has_data,
            "total_tickers": dashboard_data.get('total_tickers', 0),
            "total_sectors": dashboard_data.get('total_sectors', 0),
            "message": "Precomputed service is running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting StockVision Precomputed API Server...")
    print("📊 Loading all page data on startup...")
    
    # Test that data is loaded
    try:
        dashboard_data = precomputed_service.get_data('dashboard')
        print(f"✅ Data loaded: {dashboard_data.get('total_tickers', 'N/A')} tickers")
        print("⚡ All pages will load instantly!")
        print("🌐 Server starting on http://localhost:8000")
        print("📚 API docs at http://localhost:8000/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
