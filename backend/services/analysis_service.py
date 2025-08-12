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
