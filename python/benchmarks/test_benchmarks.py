# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Performance benchmarks for the public optimizer.

Run with::

    pytest benchmarks/ --benchmark-only

These benchmarks are excluded from the default test discovery (see
``pyproject.toml``); run them on demand to track regressions.
"""

from __future__ import annotations

import numpy as np
import pytest
from givp import GIVPConfig, givp


def sphere(x: np.ndarray) -> float:
    """Sum of squares: global minimum 0 at origin."""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock banana function: global minimum 0 at (1,...,1)."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: highly multimodal, global minimum 0 at origin."""
    n = x.size
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function: global minimum 0 at origin."""
    n = x.size
    sum_sq = float(np.sum(x**2))
    sum_cos = float(np.sum(np.cos(2.0 * np.pi * x)))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e
    )


_FUNCS = {
    "sphere": (sphere, (-5.0, 5.0)),
    "rosenbrock": (rosenbrock, (-2.0, 2.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-5.0, 5.0)),
}


@pytest.mark.parametrize("name", list(_FUNCS))
@pytest.mark.parametrize("dim", [5, 10])
def test_benchmark_classics(benchmark, name, dim):
    """Benchmark classic test functions at two dimensionalities."""
    func, (lo, hi) = _FUNCS[name]
    bounds = [(lo, hi)] * dim
    cfg = GIVPConfig(max_iterations=5, vnd_iterations=10)

    def run():
        return givp(func, bounds, config=cfg)

    result = benchmark(run)
    assert np.isfinite(result.fun)


def test_sphere_quality_gate():
    """Sphere (dim=10) should converge below 1.0 with 100 iterations."""
    bounds = [(-5.0, 5.0)] * 10
    cfg = GIVPConfig(max_iterations=100)
    result = givp(sphere, bounds, config=cfg, seed=42)
    assert result.fun < 1.0, (
        f"Sphere quality regression: expected fun < 1.0, got {result.fun}"
    )
