# Precomputed Data Controller - Serve precomputed page data instantly
from fastapi import APIRouter, HTTPException
from backend.services.precomputed_service import precomputed_service

# Create router
router = APIRouter(prefix="/api/precomputed", tags=["Precomputed Data"])

@router.get("/dashboard")
async def get_dashboard_data():
    """Get precomputed dashboard data - loads instantly"""
    try:
        return precomputed_service.get_data('dashboard')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")

@router.get("/tickers")
async def get_tickers_data():
    """Get precomputed tickers data - loads instantly"""
    try:
        return precomputed_service.get_data('tickers')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tickers data: {str(e)}")

@router.get("/sectors")
async def get_sectors_data():
    """Get precomputed sectors data - loads instantly"""
    try:
        return precomputed_service.get_data('sectors')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sectors data: {str(e)}")

@router.get("/sector-analysis")
async def get_sector_analysis_data():
    """Get precomputed sector analysis data - loads instantly"""
    try:
        return precomputed_service.get_data('sector_analysis')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector analysis data: {str(e)}")

@router.get("/market-overview")
async def get_market_overview_data():
    """Get precomputed market overview data - loads instantly"""
    try:
        return precomputed_service.get_data('market_overview')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market overview data: {str(e)}")

@router.get("/volatile-stocks")
async def get_volatile_stocks_data():
    """Get precomputed volatile stocks data - loads instantly"""
    try:
        return precomputed_service.get_data('volatile_stocks')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting volatile stocks data: {str(e)}")

@router.get("/stats")
async def get_basic_stats():
    """Get precomputed basic statistics - loads instantly"""
    try:
        return precomputed_service.get_data('basic_stats')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting basic stats: {str(e)}")

@router.get("/all")
async def get_all_precomputed_data():
    """Get ALL precomputed data in one call - ultimate performance"""
    try:
        return precomputed_service.get_all_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting all data: {str(e)}")

@router.post("/refresh")
async def refresh_precomputed_data():
    """Refresh all precomputed data"""
    try:
        precomputed_service.refresh_data()
        return {"success": True, "message": "All data refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Quick check to see if data is available
        dashboard_data = precomputed_service.get_data('dashboard')
        has_data = 'total_tickers' in dashboard_data
        
        return {
            "status": "healthy" if has_data else "no_data",
            "data_available": has_data,
            "message": "Precomputed service is running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
