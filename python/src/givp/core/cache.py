"""LRU evaluation cache used by the GRASP/ILS/VND algorithm."""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from givp.core.helpers import _get_half

try:
    import xxhash as _xxhash

    _FAST_HASH = True
except ImportError:  # pragma: no cover
    _FAST_HASH = False


class EvaluationCache:
    """LRU cache for objective function evaluations.

    Dramatically reduces the number of objective evaluations, especially during
    local search that revisits similar solutions.

    Args:
        maxsize: Maximum number of entries to keep in the cache.

    Attributes:
        cache: Cache mapping ``{hash: cost}``.
        hits: Number of cache hits.
        misses: Number of cache misses.
    """

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache: dict[int, float] = {}
        self.hits = 0
        self.misses = 0
        self.insertion_order: deque[int] = deque()

    def _hash_solution(self, solution: np.ndarray) -> int:
        """Return a deterministic integer hash of the rounded solution.

        Uses ``xxhash.xxh64`` when available (10× faster; install with
        ``pip install givp[cache]``), otherwise falls back to
        ``hashlib.sha1`` from the standard library.

        The built-in ``hash()`` is intentionally avoided: ``PYTHONHASHSEED``
        makes it non-deterministic across processes and sessions.
        """
        half = _get_half(solution.size)
        rounded = np.empty_like(solution)
        rounded[:half] = np.round(solution[:half], decimals=3)
        rounded[half:] = np.round(solution[half:], decimals=0)
        data = np.ascontiguousarray(rounded).tobytes()
        if _FAST_HASH:
            return _xxhash.xxh64_intdigest(data)  # type: ignore[union-attr]
        return int.from_bytes(
            hashlib.sha1(data, usedforsecurity=False).digest()[:8], "big"
        )

    def get(self, solution: np.ndarray) -> float | None:
        """Return the cached cost or ``None`` if not found."""
        key = self._hash_solution(solution)
        if key in self.cache:
            self.hits += 1
            return float(self.cache[key])
        self.misses += 1
        return None

    def put(self, solution: np.ndarray, cost: float) -> None:
        """Store a solution and its cost in the cache."""
        key = self._hash_solution(solution)
        if key not in self.cache and len(self.cache) >= self.maxsize:
            oldest = self.insertion_order.popleft()
            self.cache.pop(oldest, None)
        if key not in self.cache:
            self.insertion_order.append(key)
        self.cache[key] = cost

    def clear(self) -> None:
        """Clear the cache and reset all counters."""
        self.cache.clear()
        self.insertion_order = deque()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Return cache statistics as a plain dict."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }
