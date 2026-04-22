"""LRU evaluation cache used by the GRASP/ILS/VND algorithm."""

from __future__ import annotations

from collections import deque

import numpy as np

from givp._core._helpers import _get_half


class EvaluationCache:
    """Cache LRU para armazenar avaliações de soluções.

    Reduz drasticamente o número de avaliações da função objetivo,
    especialmente em buscas locais que revisitam soluções similares.

    Args:
        maxsize (int): Tamanho máximo do cache.

    Attributes:
        cache (dict): Dicionário de cache ``{hash: custo}``.
        hits (int): Contador de acertos no cache.
        misses (int): Contador de falhas no cache.
    """

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache: dict[int, float] = {}
        self.hits = 0
        self.misses = 0
        self.insertion_order: deque[int] = deque()

    def _hash_solution(self, solution: np.ndarray) -> int:
        """Gera hash rápido da solução arredondada."""
        half = _get_half(solution.size)
        rounded = np.empty_like(solution)
        rounded[:half] = np.round(solution[:half], decimals=3)
        rounded[half:] = np.round(solution[half:], decimals=0)
        return hash(np.ascontiguousarray(rounded).tobytes())

    def get(self, solution: np.ndarray) -> float | None:
        """Retorna custo cached ou ``None`` se não encontrado."""
        key = self._hash_solution(solution)
        if key in self.cache:
            self.hits += 1
            return float(self.cache[key])
        self.misses += 1
        return None

    def put(self, solution: np.ndarray, cost: float) -> None:
        """Armazena solução e custo no cache."""
        key = self._hash_solution(solution)
        if key not in self.cache and len(self.cache) >= self.maxsize:
            oldest = self.insertion_order.popleft()
            self.cache.pop(oldest, None)
        if key not in self.cache:
            self.insertion_order.append(key)
        self.cache[key] = cost

    def clear(self) -> None:
        """Limpa o cache."""
        self.cache.clear()
        self.insertion_order = deque()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }
