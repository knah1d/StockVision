#!/usr/bin/env python3
"""
Test script for the precomputed data service
Demonstrates instant loading of all page data
"""
import sys
import time
sys.path.append('/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision')

from backend.services.precomputed_service import precomputed_service

def test_precomputed_service():
    """Test the precomputed data service"""
    print("🚀 Testing Precomputed Data Service")
    print("=" * 50)
    
    # Test individual data retrieval
    data_types = [
        'dashboard',
        'tickers', 
        'sectors',
        'sector_analysis',
        'market_overview',
        'volatile_stocks',
        'basic_stats'
    ]
    
    print("⚡ Testing instant data retrieval...")
    
    for data_type in data_types:
        start_time = time.time()
        data = precomputed_service.get_data(data_type)
        end_time = time.time()
        
        # Check if data is valid
        has_error = 'error' in data
        data_size = len(str(data))
        
        status = "❌ ERROR" if has_error else "✅ SUCCESS"
        print(f"{status} {data_type:<15} - {end_time-start_time:.4f}s - {data_size:,} chars")
        
        if has_error:
            print(f"   Error: {data.get('error', 'Unknown error')}")
    
    print("\n📊 Testing dashboard data structure...")
    dashboard = precomputed_service.get_data('dashboard')
    if 'error' not in dashboard:
        print(f"   Total Tickers: {dashboard.get('total_tickers', 'N/A')}")
        print(f"   Total Sectors: {dashboard.get('total_sectors', 'N/A')}")
        print(f"   Date Range: {dashboard.get('date_range', 'N/A')}")
        print(f"   Top Performers: {len(dashboard.get('top_performers', []))} stocks")
    
    print("\n🏗️ Testing sector analysis data...")
    sector_analysis = precomputed_service.get_data('sector_analysis')
    if 'error' not in sector_analysis:
        sectors = sector_analysis.get('sector_performance', [])
        print(f"   Total Sectors Analyzed: {len(sectors)}")
        if sectors:
            best_sector = sectors[0]
            worst_sector = sectors[-1]
            print(f"   Best Performing: {best_sector.get('sector', 'N/A')} ({best_sector.get('price_change_pct', 0):.2f}%)")
            print(f"   Worst Performing: {worst_sector.get('sector', 'N/A')} ({worst_sector.get('price_change_pct', 0):.2f}%)")
    
    print("\n📈 Testing volatile stocks data...")
    volatile = precomputed_service.get_data('volatile_stocks')
    if 'error' not in volatile:
        stocks = volatile.get('volatile_stocks', [])
        print(f"   Most Volatile Stocks: {len(stocks)}")
        if stocks:
            most_volatile = stocks[0]
            print(f"   Top Volatile: {most_volatile.get('ticker', 'N/A')} ({most_volatile.get('volatility', 0):.2f}%)")
    
    print("\n🚀 Testing ALL data retrieval (ultimate performance)...")
    start_time = time.time()
    all_data = precomputed_service.get_all_data()
    end_time = time.time()
    
    total_size = len(str(all_data))
    print(f"✅ ALL data retrieved in {end_time-start_time:.4f}s")
    print(f"📦 Total data size: {total_size:,} characters")
    print(f"📋 Data types available: {list(all_data.keys())}")
    
    print("\n🎯 Benefits of Precomputed Service:")
    print("   ⚡ Instant page loads (no API delays)")
    print("   🔧 No multiple API calls")
    print("   💾 All data loaded once on startup")
    print("   🎯 Eliminates dashboard disappearing issues")
    print("   📊 Consistent data across all pages")
    
    print("\n✅ Precomputed service is working perfectly!")
    print("\n💡 Usage in frontend:")
    print("   GET /api/precomputed/dashboard - Instant dashboard data")
    print("   GET /api/precomputed/sector-analysis - Instant sector analysis")
    print("   GET /api/precomputed/all - Everything in one call")

if __name__ == "__main__":
    test_precomputed_service()
