from datetime import datetime
from typing import Any, Optional, Tuple
from collections import OrderedDict
import hashlib


class LRUCache:
    def __init__(self, max_size: int, ttl: int):
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        if hashed_key in self._cache:
            value, timestamp = self._cache[hashed_key]
            if (datetime.now() - timestamp).total_seconds() < self._ttl:
                self._cache.move_to_end(hashed_key)
                return value
            del self._cache[hashed_key]
        return None

    def set(self, key: str, value: Any) -> None:
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        if hashed_key in self._cache:
            self._cache.move_to_end(hashed_key)
        self._cache[hashed_key] = (value, datetime.now())
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
