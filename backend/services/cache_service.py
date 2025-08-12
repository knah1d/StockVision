"""
Simple Cache Service for frequently used data
Caches dashboard data, tickers, and categories with time-based invalidation
"""
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class SimpleCacheService:
    def __init__(self, default_ttl: int = 1800):  # 30 minutes default
        """
        Initialize the cache service
        
        Args:
            default_ttl: Time To Live in seconds (1800 = 30 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired"""
        if key not in self._cache:
            return True
            
        entry = self._cache[key]
        current_time = time.time()
        return current_time > entry['expires_at']
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache if it exists and is not expired"""
        if self._is_expired(key):
            # Clean up expired entry
            if key in self._cache:
                del self._cache[key]
            return None
            
        entry = self._cache[key]
        print(f"🔄 Cache HIT for '{key}' (expires in {int(entry['expires_at'] - time.time())}s)")
        return entry['data']
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Store data in cache with expiration time"""
        if ttl is None:
            ttl = self.default_ttl
            
        expires_at = time.time() + ttl
        self._cache[key] = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        expiry_time = datetime.fromtimestamp(expires_at).strftime('%H:%M:%S')
        print(f"💾 Cached '{key}' (expires at {expiry_time})")
    
    def invalidate(self, key: str) -> bool:
        """Manually invalidate a specific cache entry"""
        if key in self._cache:
            del self._cache[key]
            print(f"🗑️ Invalidated cache for '{key}'")
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        print("🗑️ Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() 
                            if current_time > entry['expires_at'])
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_keys': list(self._cache.keys())
        }
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries and return count of removed items"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            print(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
            
        return len(expired_keys)

# Global cache instance
cache = SimpleCacheService(default_ttl=1800)  # 30 minutes default

# Cache key constants
CACHE_KEYS = {
    'DASHBOARD_DATA': 'dashboard_data',
    'ALL_TICKERS': 'all_tickers',
    'SECTORS': 'sectors',
    'CATEGORIES': 'categories',
    'TOP_PERFORMERS': 'top_performers',
    'MARKET_OVERVIEW': 'market_overview'
}

def get_cached_data(key: str) -> Optional[Any]:
    """Helper function to get cached data"""
    return cache.get(key)

def set_cached_data(key: str, data: Any, ttl: Optional[int] = None) -> None:
    """Helper function to set cached data"""
    cache.set(key, data, ttl)

def invalidate_cache(key: str) -> bool:
    """Helper function to invalidate specific cache"""
    return cache.invalidate(key)

def get_cache_stats() -> Dict[str, Any]:
    """Helper function to get cache statistics"""
    return cache.get_stats()
