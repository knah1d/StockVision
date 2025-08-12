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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
# Analysis Service - Handles complex analysis operations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from services.data_service import DataService

class AnalysisService:
    def __init__(self, data_service: DataService):
        """Initialize with DataService dependency"""
        self.data_service = data_service
    
    def analyze_single_ticker(self, ticker: str, days: int = 90) -> Optional[Dict[str, Any]]:
        """Comprehensive analysis of a single ticker"""
        # Get ticker data
        result = self.data_service.get_ticker_data(ticker, days)
        if result is None:
            return None
        
        stats, ticker_data = result
        
        # Calculate technical indicators
        technical_indicators, processed_data = self.data_service.calculate_technical_indicators(ticker_data)
        
        # Calculate risk metrics
        risk_metrics = self.data_service.calculate_risk_metrics(processed_data)
        
        # Prepare chart data
        chart_data = self.data_service.prepare_chart_data(processed_data)
        
        return {
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data
        }
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                
                # Format for comparison summary
                comparison_stats.append({
                    'ticker': ticker,
                    'sector': stats['sector'],
                    'current_price': stats['current_price'],
                    'price_change_pct': stats['price_change_pct'],
                    'volatility': stats['volatility'],
                    'avg_volume': stats['avg_volume']
                })
        
        if len(comparison_stats) < 2:
            return None
        
        # Create performance ranking
        ranked_stats = sorted(comparison_stats, key=lambda x: x['price_change_pct'], reverse=True)
        performance_ranking = []
        
        for i, ticker_stat in enumerate(ranked_stats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance_ranking.append({
                'rank': i,
                'emoji': emoji,
                'ticker': ticker_stat['ticker'],
                'price_change_pct': ticker_stat['price_change_pct']
            })
        
        # Prepare comparison chart data
        chart_data = self._prepare_comparison_chart_data(ticker_data_dict)
        
        return {
            'comparison_summary': comparison_stats,
            'performance_ranking': performance_ranking,
            'chart_data': chart_data
        }
    
    def analyze_sector(self, sector_name: str, days: int = 90, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """Analyze sector performance"""
        # Get all tickers in sector
        sector_tickers = self.data_service.get_sector_tickers(sector_name)
        
        if len(sector_tickers) == 0:
            return None
        
        # Analyze each ticker
        sector_performance = []
        
        for ticker in sector_tickers:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 0:
                    sector_performance.append({
                        'ticker': ticker,
                        'price_change_pct': stats['price_change_pct'],
                        'current_price': stats['current_price'],
                        'volatility': stats['volatility'],
                        'avg_volume': stats['avg_volume']
                    })
        
        if len(sector_performance) == 0:
            return None
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
        top_performers = sector_performance[:top_n]
        
        # Calculate sector statistics
        sector_stats = {
            'total_stocks': len(sector_performance),
            'avg_price_change': float(np.mean([s['price_change_pct'] for s in sector_performance])),
            'best_performer': sector_performance[0]['ticker'],
            'worst_performer': sector_performance[-1]['ticker'],
            'avg_volatility': float(np.mean([s['volatility'] for s in sector_performance])),
            'total_volume': float(sum([s['avg_volume'] for s in sector_performance]))
        }
        
        # Prepare chart data for sector
        chart_data = self._prepare_sector_chart_data(top_performers, sector_performance)
        
        return {
            'sector_name': sector_name,
            'top_performers': top_performers,
            'sector_stats': sector_stats,
            'chart_data': chart_data
        }
    
    def analyze_volatility(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> Dict[str, Any]:
        """Analyze most volatile stocks"""
        volatile_stocks = self.data_service.find_volatile_stocks(sector, days, top_n)
        
        # Prepare chart data for volatility analysis
        chart_data = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks]
        }
        
        return {
            'volatile_stocks': volatile_stocks,
            'chart_data': chart_data
        }
    
    def _prepare_comparison_chart_data(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare chart data for ticker comparison"""
        comparison_data = {}
        
        for ticker, data in ticker_data_dict.items():
            # Normalize prices to start from 100
            normalized_prices = ((data['closing_price'] / data['closing_price'].iloc[0]) * 100).tolist()
            
            comparison_data[ticker] = {
                'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'normalized_prices': normalized_prices,
                'volumes': data['volume'].tolist(),
                'prices': data['closing_price'].tolist()
            }
        
        # Risk-return data
        risk_return_data = []
        for ticker, data in ticker_data_dict.items():
            returns = data['closing_price'].pct_change().dropna()
            if len(returns) > 1:
                avg_return = float(returns.mean() * 252)  # Annualized
                volatility = float(returns.std() * np.sqrt(252))  # Annualized
                risk_return_data.append({
                    'ticker': ticker,
                    'return': avg_return,
                    'risk': volatility
                })
        
        comparison_data['risk_return'] = risk_return_data
        
        return comparison_data
    
    def _prepare_sector_chart_data(self, top_performers: List[Dict], all_performers: List[Dict]) -> Dict[str, Any]:
        """Prepare chart data for sector analysis"""
        return {
            'top_performers': {
                'tickers': [p['ticker'] for p in top_performers],
                'price_changes': [p['price_change_pct'] for p in top_performers],
                'volatilities': [p['volatility'] for p in top_performers],
                'volumes': [p['avg_volume'] for p in top_performers]
            },
            'all_performers': {
                'price_changes': [p['price_change_pct'] for p in all_performers],
                'volatilities': [p['volatility'] for p in all_performers],
                'volumes': [p['avg_volume'] for p in all_performers]
            }
        }
    
    def get_sector_overview(self, days: int = 90) -> Dict[str, Any]:
        """Get overview of all sectors performance"""
        try:
            # Get available sectors
            sectors = self.data_service.df['sector'].dropna().unique()
            
            sector_performance = []
            sector_stats = []
            top_performers = []
            
            for sector in sectors:
                # Get sector data
                sector_data = self.data_service.df[self.data_service.df['sector'] == sector]
                tickers = sector_data['trading_code'].unique()
                
                if len(tickers) == 0:
                    continue
                
                # Calculate sector metrics
                sector_returns = []
                sector_volumes = []
                sector_market_caps = []
                top_stocks = []
                
                for ticker in tickers[:20]:  # Limit to first 20 tickers per sector
                    ticker_data = sector_data[sector_data['trading_code'] == ticker].tail(days)
                    
                    if len(ticker_data) < 2:
                        continue
                    
                    # Calculate return
                    start_price = ticker_data['closing_price'].iloc[0]
                    end_price = ticker_data['closing_price'].iloc[-1]
                    
                    if start_price > 0:
                        return_pct = ((end_price - start_price) / start_price) * 100
                        sector_returns.append(return_pct)
                        
                        # Add other metrics
                        avg_volume = ticker_data['volume'].mean()
                        sector_volumes.append(avg_volume)
                        
                        market_cap = end_price * avg_volume
                        sector_market_caps.append(market_cap)
                        
                        top_stocks.append({
                            'ticker': ticker,
                            'return': return_pct
                        })
                
                if len(sector_returns) == 0:
                    continue
                
                # Calculate averages
                avg_return = sum(sector_returns) / len(sector_returns)
                avg_market_cap = sum(sector_market_caps) / len(sector_market_caps)
                total_market_cap = sum(sector_market_caps)
                avg_volatility = 5.0  # Simplified
                
                sector_performance.append({
                    'sector': sector,
                    'avg_return': avg_return / 100,  # Convert to decimal
                    'stock_count': len(sector_returns)
                })
                
                sector_stats.append({
                    'sector': sector,
                    'company_count': len(tickers),
                    'avg_market_cap': avg_market_cap,
                    'total_market_cap': total_market_cap,
                    'avg_volatility': avg_volatility / 100
                })
                
                # Top 3 performers in sector
                top_stocks.sort(key=lambda x: x['return'], reverse=True)
                top_performers.append({
                    'sector': sector,
                    'top_stocks': top_stocks[:3]
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'sector_stats': sector_stats,
                'top_performers': top_performers
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': []
            }
# Data Service - Handles all data operations and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os

class DataService:
    def __init__(self):
        """Initialize the DataService with data loading"""
        self.df = None
        self.securities = None
        self.load_data()
    
    def load_data(self):
        """Load stock data and securities information"""
        try:
            # Load data files
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Data loaded successfully: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def get_available_tickers(self, sector: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get available tickers, optionally filtered by sector"""
        if sector:
            filtered_tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            filtered_tickers = self.df['trading_code'].unique()
        
        tickers_list = sorted(filtered_tickers)
        if limit:
            tickers_list = tickers_list[:limit]
        
        sectors = sorted(self.df['sector'].dropna().unique())
        
        return {
            'tickers': tickers_list,
            'total_count': len(filtered_tickers),
            'sectors': sectors
        }
    
    def get_ticker_data(self, ticker_symbol: str, days: Optional[int] = None) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get ticker data and basic statistics"""
        # Filter data for specific ticker
        ticker_data = self.df[self.df['trading_code'] == ticker_symbol].copy()
        
        if ticker_data.empty:
            return None
        
        # Sort by date
        ticker_data = ticker_data.sort_values('date')
        
        # Filter by days if specified
        if days is not None:
            cutoff_date = ticker_data['date'].max() - timedelta(days=days)
            ticker_data = ticker_data[ticker_data['date'] >= cutoff_date]
        
        # Calculate basic statistics
        stats = {
            'ticker': ticker_symbol,
            'sector': ticker_data['sector'].iloc[0] if not ticker_data['sector'].isna().all() else 'Unknown',
            'total_records': len(ticker_data),
            'date_range': f"{ticker_data['date'].min().strftime('%Y-%m-%d')} to {ticker_data['date'].max().strftime('%Y-%m-%d')}",
            'current_price': float(ticker_data['closing_price'].iloc[-1]),
            'price_range': f"{ticker_data['closing_price'].min():.2f} - {ticker_data['closing_price'].max():.2f}",
            'avg_volume': float(ticker_data['volume'].mean()),
            'total_volume': float(ticker_data['volume'].sum()),
            'volatility': float(ticker_data['closing_price'].std()),
            'price_change_pct': float(((ticker_data['closing_price'].iloc[-1] - ticker_data['closing_price'].iloc[0]) / ticker_data['closing_price'].iloc[0]) * 100)
        }
        
        return stats, ticker_data
    
    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for ticker data"""
        ticker_data = ticker_data.copy()
        
        # Calculate moving averages
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['MA_50'] = ticker_data['closing_price'].rolling(window=50).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        ticker_data['Price_Change'] = ticker_data['closing_price'].diff()
        
        current_price = ticker_data['closing_price'].iloc[-1]
        
        technical_indicators = {
            'current_vs_ma5': '',
            'current_vs_ma20': '',
            'ma5_value': None,
            'ma20_value': None
        }
        
        # MA5 comparison
        if len(ticker_data) >= 5:
            ma5_value = ticker_data['MA_5'].iloc[-1]
            if not pd.isna(ma5_value):
                technical_indicators['ma5_value'] = float(ma5_value)
                technical_indicators['current_vs_ma5'] = 'Above' if current_price > ma5_value else 'Below'
        
        # MA20 comparison
        if len(ticker_data) >= 20:
            ma20_value = ticker_data['MA_20'].iloc[-1]
            if not pd.isna(ma20_value):
                technical_indicators['ma20_value'] = float(ma20_value)
                technical_indicators['current_vs_ma20'] = 'Above' if current_price > ma20_value else 'Below'
        
        return technical_indicators, ticker_data
    
    def calculate_risk_metrics(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for ticker data"""
        returns = ticker_data['closing_price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {
                'daily_volatility': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'best_day': None,
                'worst_day': None,
                'best_return': 0.0,
                'worst_return': 0.0
            }
        
        daily_vol = float(returns.std())
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe_ratio = float((returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0)
        
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        
        best_day = ticker_data.loc[best_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(best_idx) else None
        worst_day = ticker_data.loc[worst_idx, 'date'].strftime('%Y-%m-%d') if not pd.isna(worst_idx) else None
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_return': float(returns.max()),
            'worst_return': float(returns.min())
        }
    
    def prepare_chart_data(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        ticker_data = ticker_data.copy()
        
        # Calculate indicators
        ticker_data['MA_5'] = ticker_data['closing_price'].rolling(window=5).mean()
        ticker_data['MA_20'] = ticker_data['closing_price'].rolling(window=20).mean()
        ticker_data['Daily_Return'] = ticker_data['closing_price'].pct_change()
        
        # Convert to lists for JSON serialization
        chart_data = {
            'dates': ticker_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': ticker_data['closing_price'].tolist(),
            'volumes': ticker_data['volume'].tolist(),
            'ma5': ticker_data['MA_5'].fillna(0).tolist(),
            'ma20': ticker_data['MA_20'].fillna(0).tolist(),
            'daily_returns': ticker_data['Daily_Return'].fillna(0).tolist(),
            'high': ticker_data['high'].tolist(),
            'low': ticker_data['low'].tolist(),
            'opening_price': ticker_data['opening_price'].tolist()
        }
        
        return chart_data
    
    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a specific sector"""
        return self.df[self.df['sector'] == sector_name]['trading_code'].unique().tolist()
    
    def find_volatile_stocks(self, sector: Optional[str] = None, days: int = 90, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most volatile stocks"""
        if sector:
            tickers = self.df[self.df['sector'] == sector]['trading_code'].unique()
        else:
            tickers = self.df['trading_code'].unique()
        
        volatility_data = []
        
        # Limit to 50 tickers for performance
        for ticker in tickers[:50]:
            result = self.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                if len(data) > 10:  # Need enough data points
                    returns = data['closing_price'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std())
                        volatility_data.append({
                            'ticker': ticker,
                            'volatility': volatility,
                            'sector': stats['sector'],
                            'price_change_pct': stats['price_change_pct']
                        })
        
        # Sort by volatility and return top N
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data[:top_n]
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
