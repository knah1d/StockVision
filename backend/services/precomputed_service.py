"""
Precomputed Data Service - Load and cache all page data on startup
This eliminates multiple API calls and provides instant page loads
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class PrecomputedDataService:
    def __init__(self):
        """Initialize and precompute all page data"""
        self.data = {}
        self.df = None
        self.securities = None
        self.load_base_data()
        self.precompute_all_data()
    
    def load_base_data(self):
        """Load the base data files"""
        try:
            data_path = "/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision/data/processed"
            self.df = pd.read_csv(f"{data_path}/all_data.csv")
            self.securities = pd.read_csv(f"{data_path}/securities.csv")
            
            # Clean and prepare data
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['trading_code'] = self.df['trading_code'].str.strip()
            self.securities['trading_code'] = self.securities['trading_code'].str.strip()
            
            # Merge with securities data
            self.df = self.df.merge(self.securities, on='trading_code', how='left')
            
            print(f"✅ Base data loaded: {self.df.shape[0]} records, {self.df['trading_code'].nunique()} tickers")
            
        except Exception as e:
            print(f"❌ Error loading base data: {str(e)}")
            raise
    
    def precompute_all_data(self):
        """Precompute all data for all pages"""
        print("🔄 Precomputing all page data...")
        
        try:
            # 1. Dashboard Data
            self.data['dashboard'] = self._compute_dashboard_data()
            
            # 2. All Tickers and Sectors
            self.data['tickers'] = self._compute_tickers_data()
            self.data['sectors'] = self._compute_sectors_data()
            
            # 3. Sector Analysis Data
            self.data['sector_analysis'] = self._compute_sector_analysis_data()
            
            # 4. Market Overview
            self.data['market_overview'] = self._compute_market_overview_data()
            
            # 5. Volatile Stocks
            self.data['volatile_stocks'] = self._compute_volatile_stocks_data()
            
            # 6. Basic Stats
            self.data['basic_stats'] = self._compute_basic_stats()
            
            print("✅ All data precomputed successfully!")
            
        except Exception as e:
            print(f"❌ Error precomputing data: {str(e)}")
            raise
    
    def _compute_dashboard_data(self) -> Dict[str, Any]:
        """Compute dashboard data"""
        try:
            total_tickers = self.df['trading_code'].nunique()
            sectors = sorted(self.df['sector'].dropna().unique())
            
            # Get latest data for each ticker
            latest_data = self.df.groupby('trading_code').tail(1)
            
            # Top performers by volume
            top_by_volume = latest_data.nlargest(10, 'volume')[
                ['trading_code', 'volume', 'closing_price', 'sector']
            ]
            
            # Market overview stats
            total_volume = self.df['volume'].sum()
            avg_price = self.df['closing_price'].mean()
            date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}"
            
            return {
                'total_tickers': total_tickers,
                'total_sectors': len(sectors),
                'sectors': sectors,
                'total_volume': float(total_volume),
                'avg_price': float(avg_price),
                'date_range': date_range,
                'top_performers': [
                    {
                        'ticker': row['trading_code'],
                        'volume': float(row['volume']),
                        'price': float(row['closing_price']),
                        'sector': row['sector'] if pd.notna(row['sector']) else 'Unknown'
                    }
                    for _, row in top_by_volume.iterrows()
                ],
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error computing dashboard data: {e}")
            return {'error': str(e)}
    
    def _compute_tickers_data(self) -> Dict[str, Any]:
        """Compute tickers data"""
        try:
            all_tickers = sorted(self.df['trading_code'].unique())
            sectors = sorted(self.df['sector'].dropna().unique())
            
            # Tickers by sector
            tickers_by_sector = {}
            for sector in sectors:
                sector_tickers = sorted(
                    self.df[self.df['sector'] == sector]['trading_code'].unique()
                )
                tickers_by_sector[sector] = sector_tickers
            
            return {
                'all_tickers': all_tickers,
                'total_count': len(all_tickers),
                'sectors': sectors,
                'tickers_by_sector': tickers_by_sector
            }
        except Exception as e:
            print(f"Error computing tickers data: {e}")
            return {'error': str(e)}
    
    def _compute_sectors_data(self) -> Dict[str, Any]:
        """Compute sectors data"""
        try:
            sectors = sorted(self.df['sector'].dropna().unique())
            
            # Sector statistics
            sector_stats = {}
            for sector in sectors:
                sector_data = self.df[self.df['sector'] == sector]
                latest_sector_data = sector_data.groupby('trading_code').tail(1)
                
                sector_stats[sector] = {
                    'total_tickers': sector_data['trading_code'].nunique(),
                    'avg_price': float(latest_sector_data['closing_price'].mean()),
                    'total_volume': float(sector_data['volume'].sum()),
                    'price_range': {
                        'min': float(latest_sector_data['closing_price'].min()),
                        'max': float(latest_sector_data['closing_price'].max())
                    }
                }
            
            return {
                'sectors': sectors,
                'sector_stats': sector_stats,
                'total_sectors': len(sectors)
            }
        except Exception as e:
            print(f"Error computing sectors data: {e}")
            return {'error': str(e)}
    
    def _compute_sector_analysis_data(self) -> Dict[str, Any]:
        """Compute sector analysis data"""
        try:
            sectors = self.df['sector'].dropna().unique()
            sector_performance = []
            
            for sector in sectors:
                sector_data = self.df[self.df['sector'] == sector]
                latest_data = sector_data.groupby('trading_code').tail(1)
                
                # Calculate sector performance metrics
                total_stocks = sector_data['trading_code'].nunique()
                avg_price = latest_data['closing_price'].mean()
                total_volume = sector_data['volume'].sum()
                
                # Calculate price change (simplified)
                if len(latest_data) > 0:
                    current_avg = latest_data['closing_price'].mean()
                    # Get data from 30 days ago for comparison
                    month_ago = self.df['date'].max() - timedelta(days=30)
                    old_data = sector_data[sector_data['date'] <= month_ago].groupby('trading_code').tail(1)
                    
                    if len(old_data) > 0:
                        old_avg = old_data['closing_price'].mean()
                        price_change = ((current_avg - old_avg) / old_avg) * 100
                    else:
                        price_change = 0.0
                else:
                    price_change = 0.0
                
                sector_performance.append({
                    'sector': sector,
                    'total_stocks': int(total_stocks),
                    'avg_price': float(avg_price),
                    'total_volume': float(total_volume),
                    'price_change_pct': float(price_change),
                    'performance_class': 'positive' if price_change >= 0 else 'negative'
                })
            
            # Sort by performance
            sector_performance.sort(key=lambda x: x['price_change_pct'], reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'total_sectors_analyzed': len(sector_performance)
            }
        except Exception as e:
            print(f"Error computing sector analysis: {e}")
            return {'error': str(e)}
    
    def _compute_market_overview_data(self) -> Dict[str, Any]:
        """Compute market overview data"""
        try:
            return {
                'total_stocks': int(self.df['trading_code'].nunique()),
                'total_sectors': int(self.df['sector'].nunique()),
                'data_from': self.df['date'].min().strftime('%Y-%m-%d'),
                'data_until': self.df['date'].max().strftime('%Y-%m-%d'),
                'total_records': int(len(self.df)),
                'avg_daily_volume': float(self.df.groupby('date')['volume'].sum().mean())
            }
        except Exception as e:
            print(f"Error computing market overview: {e}")
            return {'error': str(e)}
    
    def _compute_volatile_stocks_data(self) -> Dict[str, Any]:
        """Compute volatile stocks data"""
        try:
            # Calculate volatility for each stock
            volatility_data = []
            tickers = self.df['trading_code'].unique()[:50]  # Limit for performance
            
            for ticker in tickers:
                ticker_data = self.df[self.df['trading_code'] == ticker].sort_values('date')
                if len(ticker_data) >= 30:  # Need enough data points
                    # Calculate price volatility (standard deviation of daily returns)
                    ticker_data['daily_return'] = ticker_data['closing_price'].pct_change()
                    volatility = ticker_data['daily_return'].std() * 100  # Convert to percentage
                    
                    latest_price = ticker_data['closing_price'].iloc[-1]
                    sector = ticker_data['sector'].iloc[-1] if pd.notna(ticker_data['sector'].iloc[-1]) else 'Unknown'
                    
                    # Calculate price change over last 30 days
                    if len(ticker_data) >= 30:
                        price_30_days_ago = ticker_data['closing_price'].iloc[-30]
                        price_change = ((latest_price - price_30_days_ago) / price_30_days_ago) * 100
                    else:
                        price_change = 0.0
                    
                    volatility_data.append({
                        'ticker': ticker,
                        'sector': sector,
                        'volatility': float(volatility),
                        'current_price': float(latest_price),
                        'price_change_30d': float(price_change)
                    })
            
            # Sort by volatility (highest first)
            volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
            
            return {
                'volatile_stocks': volatility_data[:10],  # Top 10 most volatile
                'analysis_period': '30 days',
                'total_analyzed': len(volatility_data)
            }
        except Exception as e:
            print(f"Error computing volatile stocks: {e}")
            return {'error': str(e)}
    
    def _compute_basic_stats(self) -> Dict[str, Any]:
        """Compute basic statistics"""
        try:
            return {
                'data_shape': list(self.df.shape),
                'total_tickers': int(self.df['trading_code'].nunique()),
                'total_sectors': int(self.df['sector'].nunique()),
                'date_range': {
                    'start': self.df['date'].min().strftime('%Y-%m-%d'),
                    'end': self.df['date'].max().strftime('%Y-%m-%d')
                },
                'price_stats': {
                    'min': float(self.df['closing_price'].min()),
                    'max': float(self.df['closing_price'].max()),
                    'mean': float(self.df['closing_price'].mean()),
                    'median': float(self.df['closing_price'].median())
                },
                'volume_stats': {
                    'total': float(self.df['volume'].sum()),
                    'mean': float(self.df['volume'].mean()),
                    'max': float(self.df['volume'].max())
                }
            }
        except Exception as e:
            print(f"Error computing basic stats: {e}")
            return {'error': str(e)}
    
    def get_data(self, data_type: str) -> Dict[str, Any]:
        """Get precomputed data by type"""
        return self.data.get(data_type, {'error': f'Data type {data_type} not found'})
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all precomputed data"""
        return self.data
    
    def refresh_data(self):
        """Refresh all precomputed data"""
        print("🔄 Refreshing all precomputed data...")
        self.load_base_data()
        self.precompute_all_data()

# Global instance
precomputed_service = PrecomputedDataService()
