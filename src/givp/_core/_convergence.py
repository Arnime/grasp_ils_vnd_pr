# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Convergence monitor that recommends restarts/intensification."""

from __future__ import annotations

import numpy as np


class ConvergenceMonitor:
    """Monitora convergência e decide quando fazer restart.

    Rastreia histórico de melhorias e detecta estagnação prolongada,
    sugerindo restart com perturbação forte quando necessário.

    Args:
        window_size (int): Tamanho da janela para análise de tendência.
        restart_threshold (int): Iterações sem melhoria para sugerir restart.

    Attributes:
        history (list): Histórico dos melhores custos.
        no_improve_count (int): Contador de iterações sem melhoria.
    """

    def __init__(self, window_size: int = 20, restart_threshold: int = 50):
        self.window_size = window_size
        self.restart_threshold = restart_threshold
        self.history: list[float] = []
        self.no_improve_count = 0
        self.best_ever = float("inf")
        self.diversity_scores: list[float] = []

    def update(self, current_cost: float, elite_pool=None) -> dict:
        """Atualiza monitor com novo custo e retorna recomendações.

        Returns:
            dict com chaves: ``should_restart``, ``should_intensify``,
            ``diversity``, ``no_improve_count``.
        """
        self.history.append(current_cost)

        if current_cost < self.best_ever:
            self.best_ever = current_cost
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        diversity = 0.0
        if elite_pool is not None and elite_pool.size() >= 2:
            solutions = [sol for sol, _ in elite_pool.get_all()]
            distances = []
            for i, sol_i in enumerate(solutions):
                for sol_j in solutions[i + 1 :]:
                    distances.append(np.linalg.norm(sol_i - sol_j))
            diversity = float(np.mean(distances)) if distances else 0.0

        self.diversity_scores.append(diversity)

        should_restart = self.no_improve_count >= self.restart_threshold
        should_intensify = (
            self.no_improve_count >= self.restart_threshold // 2 and diversity < 0.5
        )

        return {
            "should_restart": should_restart,
            "should_intensify": should_intensify,
            "diversity": diversity,
            "no_improve_count": self.no_improve_count,
        }
