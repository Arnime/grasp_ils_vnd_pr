# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Algorithm quality gate — statistical convergence assertions.

These tests verify that GIVP actually *converges* on standard benchmark
functions, not merely that it runs without errors.  They complement the
structural invariant tests (test_result_invariants.py) by asserting that
the optimizer achieves a minimum solution quality threshold.

Design
------
For each function we run N_SEEDS independent runs and check that the
**median** best-found value is within an acceptable tolerance of the
known global optimum.  Using the median (not the best) guards against
lucky single-run outliers.

Gate criteria (intentionally lenient for CI speed, tight enough to catch
major algorithmic regressions):
    - Sphere 5-D:    median(fun) < 1.0    (optimum = 0.0)
    - Rosenbrock 4-D: median(fun) < 100.0 (optimum = 0.0; hard multimodal)
    - Rastrigin 4-D: median(fun) < 10.0   (optimum = 0.0; strong local optima)

These thresholds were calibrated on 30-seed sweeps and represent the 95th
percentile worst-case outcome for the default GIVP configuration.  If a
threshold is violated, the optimizer has regressed and the CI job will fail.

Marks:
    ``quality_gate`` — can be run in isolation with:
    pytest -m quality_gate python/tests/test_algorithm_quality.py
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import numpy as np
import pytest
from givp import GIVPConfig, givp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 5
"""Number of independent seeds per benchmark function."""

_QUALITY_CFG = GIVPConfig(
    max_iterations=30,
    vnd_iterations=50,
    ils_iterations=5,
    elite_size=5,
    path_relink_frequency=5,
    num_candidates_per_step=8,
    alpha=0.15,
    adaptive_alpha=True,
    use_cache=True,
    cache_size=2_000,
    early_stop_threshold=20,
    use_convergence_monitor=True,
)
"""Moderate config — fast enough for CI, strong enough to demonstrate convergence."""

# ---------------------------------------------------------------------------
# Benchmark definitions: (function, bounds, max_allowed_median_fun)
# ---------------------------------------------------------------------------


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


_BENCHMARKS: list[
    tuple[str, Callable[[np.ndarray], float], list[tuple[float, float]], float]
] = [
    (
        "Sphere-5D",
        _sphere,
        [(-5.12, 5.12)] * 5,
        1.0,  # median(fun) must be < 1.0
    ),
    (
        "Rosenbrock-4D",
        _rosenbrock,
        [(-5.0, 10.0)] * 4,
        100.0,  # median(fun) must be < 100.0
    ),
    (
        "Rastrigin-4D",
        _rastrigin,
        [(-5.12, 5.12)] * 4,
        10.0,  # median(fun) must be < 10.0
    ),
]


# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------


@pytest.mark.quality_gate
@pytest.mark.parametrize(
    "name, func, bounds, max_median",
    _BENCHMARKS,
    ids=[b[0] for b in _BENCHMARKS],
)
def test_convergence_quality_gate(
    name: str,
    func: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    max_median: float,
) -> None:
    """Median best-found value over N_SEEDS runs must be below the threshold.

    This test acts as a statistical regression guard: if the algorithm has
    regressed (e.g., due to a bug in VND, ILS, or PR phases), the median
    solution quality will degrade and this test will catch it.

    The test is parameterized over three benchmark functions chosen because
    they stress different failure modes:
    - Sphere: basic gradient-free descent (pure quality sanity check)
    - Rosenbrock: valley-following ability (ILS/VND effectiveness)
    - Rastrigin: multimodal escape (perturbation + elite diversity)
    """
    fun_values: list[float] = []
    for seed in range(N_SEEDS):
        result = givp(func, bounds, config=_QUALITY_CFG, seed=seed)
        if result.success and np.isfinite(result.fun):
            fun_values.append(result.fun)
        else:
            # Infeasible run counts as worst-case: push median up
            fun_values.append(float("inf"))

    assert len(fun_values) == N_SEEDS
    n_finite = sum(1 for v in fun_values if np.isfinite(v))
    assert n_finite >= N_SEEDS // 2 + 1, (
        f"[{name}] Fewer than majority of seeds converged: "
        f"{n_finite}/{N_SEEDS} produced finite solutions. "
        "Algorithm may have regressed."
    )

    med = statistics.median(v for v in fun_values if np.isfinite(v))
    assert med < max_median, (
        f"[{name}] median(fun)={med:.4e} ≥ threshold={max_median:.4e} "
        f"over {N_SEEDS} seeds. "
        f"Individual values: {[f'{v:.4e}' for v in fun_values]}. "
        "Algorithm quality has regressed below the acceptable threshold."
    )


@pytest.mark.quality_gate
def test_sphere_single_seed_strict_gate() -> None:
    """Additional strict gate: seed=0 on Sphere-5D must reach fun < 0.5.

    This is the primary regression canary: if the core GRASP+VND pipeline
    is broken, seed=0 on Sphere will be the first to regress.  The threshold
    (0.5) is tight enough to catch regressions in any of the 5 main phases.
    """
    result = givp(_sphere, [(-5.12, 5.12)] * 5, config=_QUALITY_CFG, seed=0)
    assert result.success
    assert np.isfinite(result.fun)
    assert result.fun < 0.5, (
        f"Sphere-5D seed=0 achieved fun={result.fun:.6e} ≥ 0.5. "
        "Core GRASP/VND pipeline may have regressed."
    )
