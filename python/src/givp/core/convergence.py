# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Convergence monitor that recommends restarts/intensification."""

from __future__ import annotations

import numpy as np


class ConvergenceMonitor:
    """Monitor that tracks stagnation and recommends restarts or intensification.

    Tracks the improvement history and detects prolonged stagnation,
    suggesting a restart with strong perturbation when needed.

    Args:
        window_size: Window size for trend analysis.
        restart_threshold: Iterations without improvement before recommending a restart.

    Attributes:
        history: History of best objective costs.
        no_improve_count: Number of consecutive iterations without improvement.
    """

    def __init__(self, window_size: int = 20, restart_threshold: int = 50):
        """Initialize the convergence monitor."""
        self.window_size = window_size
        self.restart_threshold = restart_threshold
        self.history: list[float] = []
        self.no_improve_count = 0
        self.best_ever = float("inf")
        self.diversity_scores: list[float] = []

    def update(self, current_cost: float, elite_pool=None) -> dict:
        """Update the monitor with the latest cost and return recommendations.

        Returns:
            Dict with keys: ``should_restart``, ``should_intensify``,
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
