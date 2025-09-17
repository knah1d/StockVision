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
        
        # Create formatted result that matches frontend expectations
        analysis_result = {
            'ticker': ticker,
            'name': stats.get('name', ticker),
            'sector': stats.get('sector', 'Unknown'),
            'ticker_info': stats,
            'technical_indicators': technical_indicators,
            'risk_metrics': risk_metrics,
            'chart_data': chart_data,  # Frontend expects chart_data, not price_data
            'summary': {
                'current_price': stats.get('current_price', 0.0),
                'price_change': stats.get('price_change_pct', 0.0),
                'date_range': stats.get('date_range', ''),
                'volume': stats.get('avg_volume', 0.0)
            },
            'price_data': chart_data,  # Keep this for API compatibility
            'statistics': {
                **stats,
                'risk_metrics': risk_metrics
            },
            'analysis': {
                'trend': 'Bullish' if stats.get('price_change_pct', 0) > 0 else 'Bearish',
                'summary': f"Analysis for {ticker} over {days} days",
                'recommendations': 'Buy' if stats.get('price_change_pct', 0) > 5 else 'Hold' if stats.get('price_change_pct', 0) > 0 else 'Sell'
            }
        }
        
        return analysis_result
    
    def compare_tickers(self, ticker_list: List[str], days: int = 90) -> Optional[Dict[str, Any]]:
        """Compare multiple tickers"""
        if len(ticker_list) < 2:
            return None
        
        comparison_stats = []
        ticker_data_dict = {}
        names_dict = {}
        sectors_dict = {}
        
        # Collect data for all tickers
        for ticker in ticker_list:
            result = self.data_service.get_ticker_data(ticker, days)
            if result is not None:
                stats, data = result
                ticker_data_dict[ticker] = data
                names_dict[ticker] = stats.get('name', ticker)
                sectors_dict[ticker] = stats.get('sector', 'Unknown')
                
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
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(ticker_data_dict)
        
        # Calculate risk metrics for each ticker
        risk_metrics = {}
        for ticker, data in ticker_data_dict.items():
            risk_metrics[ticker] = self.data_service.calculate_risk_metrics(data)
        
        # Format the result to match TickerComparisonResponse schema
        return {
            'tickers': ticker_list,
            'names': names_dict,
            'sectors': sectors_dict,
            'price_data': chart_data,
            'performance_metrics': {
                'ranking': performance_ranking,
                'summary': comparison_stats
            },
            'correlation_matrix': correlation_matrix,
            'risk_metrics': risk_metrics,
            'analysis': {
                'best_performer': ranked_stats[0]['ticker'] if ranked_stats else None,
                'worst_performer': ranked_stats[-1]['ticker'] if ranked_stats else None,
                'summary': f"Comparison of {len(ticker_list)} stocks over {days} days",
                'recommendations': "Consider investing in the top-performing stocks while maintaining diversification."
            }
        }
        
    def _calculate_correlation_matrix(self, ticker_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate correlation matrix for ticker comparison"""
        # Extract closing prices for each ticker
        prices_dict = {}
        for ticker, data in ticker_data_dict.items():
            prices_dict[ticker] = data['closing_price'].values
        
        # Create a DataFrame for correlation calculation
        df = pd.DataFrame(prices_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr().round(3)
        
        # Convert to dictionary format for JSON serialization
        corr_data = {
            'tickers': list(corr_matrix.columns),
            'matrix': corr_matrix.values.tolist()
        }
        
        return corr_data
    
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
        volatility_metrics = {
            'tickers': [stock['ticker'] for stock in volatile_stocks],
            'volatilities': [stock['volatility'] for stock in volatile_stocks],
            'price_changes': [stock['price_change_pct'] for stock in volatile_stocks],
            'sectors': [stock['sector'] for stock in volatile_stocks],
            'average_volatility': float(np.mean([stock['volatility'] for stock in volatile_stocks])) if volatile_stocks else 0.0
        }
        
        # Create analysis text
        analysis = {
            'summary': f"Analysis of the {top_n} most volatile stocks over {days} days" + (f" in the {sector} sector" if sector else ""),
            'insights': "Higher volatility stocks may present trading opportunities but come with increased risk."
        }
        
        return {
            'period_days': days,
            'sector': sector,
            'volatile_stocks': volatile_stocks,
            'volatility_metrics': volatility_metrics,
            'analysis': analysis
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
            # Get available sectors as a list
            sectors = sorted(self.data_service.df['sector'].dropna().unique().tolist())
            
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
                'top_performers': top_performers,
                'sectors_list': sectors  # Add the sectors list for consistent API
            }
            
        except Exception as e:
            print(f"Error in get_sector_overview: {str(e)}")
            return {
                'sector_performance': [],
                'sector_stats': [],
                'top_performers': [],
                'sectors_list': []
            }
