"""Diversity-aware elite pool of high-quality solutions."""

from __future__ import annotations

import numpy as np

from givp.exceptions import EmptyPoolError


class ElitePool:
    """Diversity-aware elite pool of high-quality solutions.

    Maintains a set of high-quality solutions guaranteeing a minimum relative
    distance between them.  Solutions can be added, retrieved, and cleared.

    Args:
        max_size: Maximum pool size.
        min_distance: Minimum normalised relative distance between solutions (0–1).
        lower: Lower bounds for each variable (used to normalise distances).
        upper: Upper bounds for each variable (used to normalise distances).

    Attributes:
        pool: List of ``(solution, cost)`` pairs, sorted by ascending cost.
    """

    def __init__(
        self,
        max_size: int = 5,
        min_distance: float = 0.05,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ):
        self.max_size = max_size
        self.min_distance = min_distance
        self.pool: list[tuple[np.ndarray, float]] = []
        self._range = None
        if lower is not None and upper is not None:
            self._range = np.maximum(upper - lower, 1e-12)

    def _relative_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the normalised mean relative distance between two solutions."""
        if self._range is not None:
            return float(np.mean(np.abs(a - b) / self._range))
        return float(np.linalg.norm(a - b))

    def add(self, solution: np.ndarray, benefit: float) -> bool:
        """Add a solution to the pool if it is good enough and sufficiently diverse.

        Returns:
            ``True`` if the solution was added.
        """
        solution = np.array(solution, dtype=float)

        for elite_sol, _ in self.pool:
            distance = self._relative_distance(solution, elite_sol)
            if distance < self.min_distance:
                return False

        if len(self.pool) < self.max_size:
            self.pool.append((solution.copy(), benefit))
            self.pool.sort(key=lambda x: x[1])
            return True

        if benefit < self.pool[-1][1]:
            self.pool[-1] = (solution.copy(), benefit)
            self.pool.sort(key=lambda x: x[1])
            return True

        return False

    def get_best(self) -> tuple[np.ndarray, float]:
        """Return the best (lowest-cost) solution in the pool."""
        if not self.pool:
            raise EmptyPoolError("elite pool is empty; cannot return best solution")
        return self.pool[0]

    def get_all(self) -> list[tuple[np.ndarray, float]]:
        """Return a copy of all solutions in the pool."""
        return self.pool.copy()

    def size(self) -> int:
        """Retorna o tamanho atual do pool."""
        return len(self.pool)

    def clear(self):
        """Limpa o pool."""
        self.pool.clear()
