# Stock Analysis Controllers - Handle API endpoints
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from models.schemas import (
    TickerAnalysisRequest, TickerAnalysisResponse,
    TickerComparisonRequest, TickerComparisonResponse,
    SectorAnalysisRequest, SectorAnalysisResponse,
    VolatilityAnalysisRequest, VolatilityAnalysisResponse,
    AvailableTickersResponse, ErrorResponse
)
from services.data_service import DataService
from services.analysis_service import AnalysisService

# Create router
router = APIRouter(prefix="/api/analysis", tags=["Stock Analysis"])

# Global services (will be initialized in main.py)
data_service = None
analysis_service = None

def get_data_service():
    """Dependency to get data service"""
    global data_service
    if data_service is None:
        data_service = DataService()
    return data_service

def get_analysis_service():
    """Dependency to get analysis service"""
    global analysis_service
    if analysis_service is None:
        analysis_service = AnalysisService(get_data_service())
    return analysis_service

@router.get("/tickers", response_model=AvailableTickersResponse)
async def get_available_tickers(
    sector: Optional[str] = None,
    limit: Optional[int] = None,
    data_svc: DataService = Depends(get_data_service)
):
    """Get available tickers, optionally filtered by sector"""
    try:
        result = data_svc.get_available_tickers(sector, limit)
        return AvailableTickersResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tickers: {str(e)}")

@router.post("/ticker", response_model=TickerAnalysisResponse)
async def analyze_ticker(
    request: TickerAnalysisRequest,
    analysis_svc: AnalysisService = Depends(get_analysis_service)
):
    """Analyze a single ticker comprehensively"""
    try:
        result = analysis_svc.analyze_single_ticker(request.ticker, request.days)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for ticker: {request.ticker}"
            )
        
        return TickerAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing ticker: {str(e)}")

@router.post("/compare", response_model=TickerComparisonResponse)
async def compare_tickers(
    request: TickerComparisonRequest,
    analysis_svc: AnalysisService = Depends(get_analysis_service)
):
    """Compare multiple tickers"""
    try:
        if len(request.tickers) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 tickers required for comparison"
            )
        
        result = analysis_svc.compare_tickers(request.tickers, request.days)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail="Not enough valid data found for comparison"
            )
        
        return TickerComparisonResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing tickers: {str(e)}")

@router.post("/sector", response_model=SectorAnalysisResponse)
async def analyze_sector(
    request: SectorAnalysisRequest,
    analysis_svc: AnalysisService = Depends(get_analysis_service)
):
    """Analyze sector performance"""
    try:
        result = analysis_svc.analyze_sector(request.sector, request.days, request.top_n)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for sector: {request.sector}"
            )
        
        return SectorAnalysisResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sector: {str(e)}")

@router.get("/sectors/{days}")
async def get_sector_analysis(
    days: int,
    analysis_svc: AnalysisService = Depends(get_analysis_service)
):
    """Get sector analysis overview"""
    try:
        result = analysis_svc.get_sector_overview(days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sector analysis: {str(e)}")

@router.post("/volatility", response_model=VolatilityAnalysisResponse)
async def analyze_volatility(
    request: VolatilityAnalysisRequest,
    analysis_svc: AnalysisService = Depends(get_analysis_service)
):
    """Analyze most volatile stocks"""
    try:
        result = analysis_svc.analyze_volatility(request.sector, request.days, request.top_n)
        return VolatilityAnalysisResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing volatility: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Stock Analysis API is running"}

@router.get("/stats")
async def get_api_stats(data_svc: DataService = Depends(get_data_service)):
    """Get API statistics"""
    try:
        tickers_info = data_svc.get_available_tickers()
        return {
            "total_tickers": tickers_info['total_count'],
            "total_sectors": len(tickers_info['sectors']),
            "data_shape": data_svc.df.shape if data_svc.df is not None else None,
            "date_range": {
                "start": data_svc.df['date'].min().strftime('%Y-%m-%d') if data_svc.df is not None else None,
                "end": data_svc.df['date'].max().strftime('%Y-%m-%d') if data_svc.df is not None else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
