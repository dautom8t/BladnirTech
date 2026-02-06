"""
Cache abstraction layer.

Demo / development  →  in-memory dict with TTL eviction
Production          →  Redis (when CACHE_BACKEND=redis)

Usage::

    from services.cache import cache

    cache.set("workflow:42", data, ttl=60)
    hit = cache.get("workflow:42")
    cache.delete("workflow:42")
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =====================================
# Abstract interface
# =====================================

class CacheBackend(ABC):
    """Minimal cache contract shared by every backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        ...


# =====================================
# In-memory implementation (demo)
# =====================================

class MemoryCache(CacheBackend):
    """Thread-safe in-memory cache with per-key TTL."""

    def __init__(self, default_ttl: int = 300) -> None:
        self._store: Dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if expires_at and time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + ttl if ttl > 0 else 0.0
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            live = sum(1 for _, (_, exp) in self._store.items() if not exp or now <= exp)
            return {
                "backend": "memory",
                "keys": len(self._store),
                "live_keys": live,
                "hits": self._hits,
                "misses": self._misses,
            }


# =====================================
# Redis stub (production placeholder)
# =====================================

class RedisCache(CacheBackend):
    """
    Placeholder for a Redis-backed cache.

    Requires ``redis`` package (``pip install redis``).
    In demo mode this class is never instantiated.
    """

    def __init__(self, url: str, default_ttl: int = 300) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis cache backend requires the 'redis' package. "
                "Install it with: pip install redis"
            ) from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._default_ttl = default_ttl
        logger.info(f"Redis cache connected: {url.split('@')[-1]}")

    def get(self, key: str) -> Optional[Any]:
        import json
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        import json
        ttl = ttl if ttl is not None else self._default_ttl
        serialized = json.dumps(value) if not isinstance(value, str) else value
        if ttl > 0:
            self._client.setex(key, ttl, serialized)
        else:
            self._client.set(key, serialized)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def clear(self) -> None:
        self._client.flushdb()

    def stats(self) -> Dict[str, Any]:
        info = self._client.info(section="keyspace")
        return {"backend": "redis", "info": info}


# =====================================
# Factory
# =====================================

def _build_cache() -> CacheBackend:
    from config import settings

    if settings.cache_backend == "redis":
        return RedisCache(url=settings.redis_url, default_ttl=settings.cache_default_ttl)

    return MemoryCache(default_ttl=settings.cache_default_ttl)


cache: CacheBackend = _build_cache()
logger.info(f"Cache initialised: {cache.stats()['backend']}")
