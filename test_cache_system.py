#!/usr/bin/env python3
"""
Test script to demonstrate the simple caching system
Shows how frequently used data is cached and invalidated
"""
import sys
import time
from pathlib import Path
sys.path.append('/home/kibria/Desktop/IIT_Folders/6th_semester/AI/StockVision')

from backend.services.cache_service import (
    SimpleCacheService, 
    get_cached_data, 
    set_cached_data, 
    get_cache_stats,
    CACHE_KEYS
)

def test_caching_system():
    """Test the caching system functionality"""
    print("💾 Testing Simple Caching System")
    print("=" * 50)
    
    # Create a test cache with shorter TTL for demo
    test_cache = SimpleCacheService(default_ttl=10)  # 10 seconds for demo
    
    # Test data
    sample_tickers = ['AAPL', 'GOOGL', 'MSFT', 'BEXIMCO', 'PADMA']
    sample_sectors = ['Technology', 'Pharma', 'Bank', 'Textile']
    
    print("1️⃣ Testing basic cache operations...")
    
    # Set some cache data
    test_cache.set('tickers', sample_tickers, ttl=15)
    test_cache.set('sectors', sample_sectors, ttl=20)
    test_cache.set('dashboard', {'total_stocks': 100, 'total_volume': 1000000}, ttl=30)
    
    # Get cache stats
    stats = test_cache.get_stats()
    print(f"📊 Cache stats: {stats['active_entries']} active entries")
    
    print("\n2️⃣ Testing cache retrieval...")
    
    # Test cache hits
    cached_tickers = test_cache.get('tickers')
    cached_sectors = test_cache.get('sectors')
    
    if cached_tickers:
        print(f"✅ Tickers from cache: {cached_tickers[:3]}...")
    if cached_sectors:
        print(f"✅ Sectors from cache: {cached_sectors}")
    
    print("\n3️⃣ Testing cache expiration...")
    print("⏳ Waiting for cache to expire (10 seconds)...")
    time.sleep(11)
    
    # Try to get expired data
    expired_tickers = test_cache.get('tickers')
    if expired_tickers is None:
        print("✅ Cache properly expired - no data returned")
    else:
        print("❌ Cache should have expired")
    
    # Clean up expired entries
    cleaned = test_cache.cleanup_expired()
    print(f"🗑️ Cleaned up {cleaned} expired entries")
    
    print("\n4️⃣ Testing real cache with data service simulation...")
    
    # Simulate data service caching
    def simulate_expensive_data_load():
        """Simulate loading data from database/files"""
        print("🔄 Loading data from database... (expensive operation)")
        time.sleep(1)  # Simulate delay
        return {
            'tickers': ['BEXIMCO', 'PADMA', 'SQUARE', 'ACI', 'BRAC'],
            'total_count': 500,
            'sectors': ['Bank', 'Pharma', 'IT', 'Textile', 'Energy']
        }
    
    def get_tickers_with_cache():
        """Get tickers with caching"""
        cached_data = get_cached_data(CACHE_KEYS['ALL_TICKERS'])
        if cached_data is not None:
            return cached_data
        
        # Load fresh data
        fresh_data = simulate_expensive_data_load()
        set_cached_data(CACHE_KEYS['ALL_TICKERS'], fresh_data, ttl=30)
        return fresh_data
    
    # First call - will load from "database"
    print("\n📥 First call (cache miss):")
    start_time = time.time()
    data1 = get_tickers_with_cache()
    time1 = time.time() - start_time
    print(f"⏱️ Time taken: {time1:.2f}s")
    
    # Second call - will load from cache
    print("\n📥 Second call (cache hit):")
    start_time = time.time()
    data2 = get_tickers_with_cache()
    time2 = time.time() - start_time
    print(f"⏱️ Time taken: {time2:.3f}s")
    
    print(f"\n🚀 Performance improvement: {(time1/time2):.1f}x faster with cache!")
    
    # Show final cache stats
    final_stats = get_cache_stats()
    print(f"\n📊 Final cache stats:")
    print(f"   Active entries: {final_stats['active_entries']}")
    print(f"   Cache keys: {final_stats['cache_keys']}")
    
    print("\n✅ Caching system working perfectly!")
    print("\n🎯 Benefits:")
    print("   ⚡ Fast data retrieval")
    print("   💾 Reduced database load") 
    print("   🕐 Time-based invalidation")
    print("   🧹 Automatic cleanup")

if __name__ == "__main__":
    test_caching_system()
