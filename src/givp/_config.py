"""Configuration dataclass for the GRASP-ILS-VND-PR algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["minimize", "maximize"]


@dataclass
class GraspIlsVndConfig:
    """
    Hyper-parameters for the GRASP-ILS-VND-PR algorithm.

    The ``direction`` field makes the algorithm agnostic to minimization or
    maximization. Internally the library always minimizes; when
    ``direction='maximize'`` the user-supplied objective is wrapped with a sign
    flip and the returned value is restored to the original sign.

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
        direction: ``'minimize'`` (default) or ``'maximize'``.
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
    direction: Direction = "minimize"
    integer_split: int | None = None  # index where integer vars begin; None -> n // 2 (legacy)

    def as_core_config(self):
        """Return an internal config object compatible with ``_core``.

        ``_core`` defines its own ``GraspIlsVndConfig`` (without ``direction``),
        so we copy field values across to keep the two layers decoupled.
        """
        from givp import _core

        return _core.GraspIlsVndConfig(
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
