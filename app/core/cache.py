"""Redis cache management"""
import redis
import pickle
import hashlib
from typing import Optional, Any
from functools import wraps
from .config import get_settings

settings = get_settings()


class RedisCache:
    """Redis cache wrapper with fallback"""

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.enabled = settings.redis_enabled
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        if not self.enabled:
            return

        try:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.client.ping()
            print(f"✓ Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            print(f"⚠ Redis not available: {e}")
            self.client = None
            self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.client:
            return None

        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        if not self.enabled or not self.client:
            return

        try:
            ttl = ttl or settings.cache_ttl
            data = pickle.dumps(value)
            self.client.setex(key, ttl, data)
        except Exception as e:
            print(f"Cache set error: {e}")

    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return

        try:
            self.client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if not self.enabled or not self.client:
            return

        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            print(f"Cache clear error: {e}")

    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.enabled or not self.client:
            return False

        try:
            self.client.ping()
            return True
        except:
            return False


# Global cache instance
cache = RedisCache()


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: int = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)

            return result
        return wrapper
    return decorator
