# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""LRU evaluation cache used by the GRASP/ILS/VND algorithm."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

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
        """Initialize the evaluation cache."""
        self.maxsize = maxsize
        self.cache: OrderedDict[int, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

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
        data = rounded.tobytes()
        if _FAST_HASH:
            return int(_xxhash.xxh64_intdigest(data))  # type: ignore[union-attr]
        return int.from_bytes(
            hashlib.sha1(data, usedforsecurity=False).digest()[:8], "big"
        )

    def get(self, solution: np.ndarray) -> float | None:
        """Return the cached cost or ``None`` if not found."""
        key = self._hash_solution(solution)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return float(self.cache[key])
        self.misses += 1
        return None

    def put(self, solution: np.ndarray, cost: float) -> None:
        """Store a solution and its cost in the cache."""
        key = self._hash_solution(solution)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = cost
            return
        if len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = cost

    def clear(self) -> None:
        """Clear the cache and reset all counters."""
        self.cache.clear()
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
