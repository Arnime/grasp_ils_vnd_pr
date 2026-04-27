"""Diversity-aware elite pool of high-quality solutions."""

from __future__ import annotations

import numpy as np

from givp.exceptions import EmptyPoolError


class ElitePool:
    """Pool de soluções elite com manutenção de diversidade.

    Mantém um conjunto de soluções de alta qualidade, garantindo distância
    relativa mínima entre elas. Permite adicionar, recuperar e limpar
    soluções elite.

    Args:
            max_size (int): Tamanho máximo do pool.
            min_distance (float): Distância relativa mínima normalizada (0-1).
            lower (np.ndarray | None): Limites inferiores das variáveis.
            upper (np.ndarray | None): Limites superiores das variáveis.

    Attributes:
            pool (list[tuple[np.ndarray, float]]): ``(solução, benefício)``.
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
        """Calcula distância relativa normalizada média entre duas soluções."""
        if self._range is not None:
            return float(np.mean(np.abs(a - b) / self._range))
        return float(np.linalg.norm(a - b))

    def add(self, solution: np.ndarray, benefit: float) -> bool:
        """Adiciona solução ao pool se for boa o suficiente e diversa.

        Returns:
                ``True`` se a solução foi adicionada.
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
        """Retorna a melhor solução do pool."""
        if not self.pool:
            raise EmptyPoolError("elite pool is empty; cannot return best solution")
        return self.pool[0]

    def get_all(self) -> list[tuple[np.ndarray, float]]:
        """Retorna todas as soluções do pool."""
        return self.pool.copy()

    def size(self) -> int:
        """Retorna o tamanho atual do pool."""
        return len(self.pool)

    def clear(self):
        """Limpa o pool."""
        self.pool.clear()
