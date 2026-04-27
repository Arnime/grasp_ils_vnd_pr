# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Configuration dataclass for the GRASP-ILS-VND-PR algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from givp import core as _core
from givp.exceptions import InvalidConfigError

Direction = Literal["minimize", "maximize"]


@dataclass
class GIVPConfig:
    """
    Hyper-parameters for the GRASP-ILS-VND-PR algorithm.

    The optimization sense is controlled by ``minimize`` (boolean, preferred)
    or ``direction`` (string, SciPy/Optuna-style). Pass *one* of the two; if
    both are given, ``minimize`` wins and ``direction`` is rewritten to match.
    Internally the library always minimizes: when the user wants maximization
    the objective is wrapped with a sign flip and the returned value is
    restored to the original sign.

    Attributes:
        max_iterations: Maximum number of GRASP outer iterations.
        alpha: Initial randomization parameter for greedy construction (RCL).
        vnd_iterations: Maximum iterations of the VND local search.
        ils_iterations: Maximum iterations of the ILS loop.
        perturbation_strength: Magnitude of the ILS perturbation.
        use_elite_pool: Whether to maintain an elite pool for path relinking.
        elite_size: Maximum size of the elite pool.
        path_relink_frequency: GRASP iteration period at which to run PR.
        adaptive_alpha: If True, alpha varies between alpha_min and alpha_max.
        alpha_min: Lower bound used by adaptive alpha.
        alpha_max: Upper bound used by adaptive alpha.
        num_candidates_per_step: Candidates evaluated per construction step.
        use_cache: If True, evaluations are memoized via an LRU cache.
        cache_size: Maximum entries kept by the LRU cache.
        early_stop_threshold: Iterations without improvement to early-stop.
        use_convergence_monitor: Enable diversification/restart heuristics.
        n_workers: Threads used to evaluate candidates in parallel.
        time_limit: Wall-clock budget in seconds (0 = unlimited).
        minimize: Boolean convenience flag. ``True`` (default) means
            minimization, ``False`` means maximization. When set, it overrides
            ``direction``.
        direction: ``'minimize'`` (default) or ``'maximize'``. Kept for
            SciPy/Optuna-style API compatibility.
        integer_split: Index where integer variables begin in the decision
            vector. ``None`` preserves the legacy SOG2 behavior of splitting
            in half. Set to ``num_vars`` for fully continuous problems or to
            ``0`` for fully integer problems.
    """

    max_iterations: int = 100
    alpha: float = 0.12
    vnd_iterations: int = 200
    ils_iterations: int = 10
    perturbation_strength: int = 4
    use_elite_pool: bool = True
    elite_size: int = 7
    path_relink_frequency: int = 8
    adaptive_alpha: bool = True
    alpha_min: float = 0.08
    alpha_max: float = 0.18
    num_candidates_per_step: int = 20
    use_cache: bool = True
    cache_size: int = 10000
    early_stop_threshold: int = 80
    use_convergence_monitor: bool = True
    n_workers: int = 1
    time_limit: float = 0.0
    minimize: bool | None = None
    direction: Direction = "minimize"
    integer_split: int | None = (
        None  # index where integer vars begin; None -> n // 2 (legacy)
    )

    def __post_init__(self) -> None:
        # ``minimize`` is the canonical boolean flag. When the user sets it,
        # it wins and ``direction`` is rewritten to match. When omitted,
        # ``minimize`` is derived from ``direction`` so both fields are
        # always consistent for downstream readers.
        if self.minimize is None:
            if self.direction not in ("minimize", "maximize"):
                raise InvalidConfigError(
                    "direction must be 'minimize' or 'maximize', "
                    f"got {self.direction!r}"
                )
            self.minimize = self.direction == "minimize"
        else:
            self.direction = "minimize" if self.minimize else "maximize"

        self._validate_numeric_fields()

    def _validate_numeric_fields(self) -> None:
        """Validate numeric ranges; raises ``InvalidConfigError`` on failure."""
        positive_int = {
            "max_iterations": self.max_iterations,
            "vnd_iterations": self.vnd_iterations,
            "ils_iterations": self.ils_iterations,
            "elite_size": self.elite_size,
            "path_relink_frequency": self.path_relink_frequency,
            "num_candidates_per_step": self.num_candidates_per_step,
            "cache_size": self.cache_size,
            "early_stop_threshold": self.early_stop_threshold,
            "n_workers": self.n_workers,
        }
        for name, value in positive_int.items():
            if not isinstance(value, int) or value < 1:
                raise InvalidConfigError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        if (
            not isinstance(self.perturbation_strength, int)
            or self.perturbation_strength < 0
        ):
            raise InvalidConfigError(
                f"perturbation_strength must be a non-negative integer, "
                f"got {self.perturbation_strength!r}"
            )

        if not 0.0 <= self.alpha <= 1.0:
            raise InvalidConfigError(f"alpha must be in [0, 1], got {self.alpha!r}")
        if not 0.0 <= self.alpha_min <= 1.0:
            raise InvalidConfigError(
                f"alpha_min must be in [0, 1], got {self.alpha_min!r}"
            )
        if not 0.0 <= self.alpha_max <= 1.0:
            raise InvalidConfigError(
                f"alpha_max must be in [0, 1], got {self.alpha_max!r}"
            )
        if self.alpha_min > self.alpha_max:
            raise InvalidConfigError(
                f"alpha_min ({self.alpha_min}) must be <= alpha_max ({self.alpha_max})"
            )

        if self.time_limit < 0:
            raise InvalidConfigError(
                f"time_limit must be >= 0 (0 = unlimited), got {self.time_limit!r}"
            )

        if self.integer_split is not None and self.integer_split < 0:
            raise InvalidConfigError(
                f"integer_split must be >= 0 or None, got {self.integer_split!r}"
            )

    def as_core_config(self):
        """Return an internal config object compatible with ``core``.

        ``core`` defines its own ``GIVPConfig`` (without ``direction``),
        so we copy field values across to keep the two layers decoupled.
        """
        return _core.GIVPConfig(
            max_iterations=self.max_iterations,
            alpha=self.alpha,
            vnd_iterations=self.vnd_iterations,
            ils_iterations=self.ils_iterations,
            perturbation_strength=self.perturbation_strength,
            use_elite_pool=self.use_elite_pool,
            elite_size=self.elite_size,
            path_relink_frequency=self.path_relink_frequency,
            adaptive_alpha=self.adaptive_alpha,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            num_candidates_per_step=self.num_candidates_per_step,
            use_cache=self.use_cache,
            cache_size=self.cache_size,
            early_stop_threshold=self.early_stop_threshold,
            use_convergence_monitor=self.use_convergence_monitor,
            n_workers=self.n_workers,
            time_limit=self.time_limit,
            integer_split=self.integer_split,
        )
