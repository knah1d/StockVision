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
