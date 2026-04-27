# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Classic optimization benchmark functions and small helpers used in examples/tests.

This module contains simple, well-known test functions (Sphere, Rosenbrock,
Rastrigin, Ackley, Griewank, Schwefel) and a few combinatorial helpers (knapsack
DP, a penalty wrapper) that are used by the project's examples and by the
unit tests added alongside them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def sphere(x: NDArray | Sequence[float]) -> float:
    """Sphere function: f(x) = sum(x_i^2). Global minimum at x=0 -> f=0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def rosenbrock(x: NDArray | Sequence[float]) -> float:
    """Rosenbrock 'banana' function. Minimum at x = 1..1 -> f = 0."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: NDArray | Sequence[float]) -> float:
    """Rastrigin multimodal function. Minimum at x=0 -> f=0."""
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: NDArray | Sequence[float]) -> float:
    """Ackley function. Minimum at x=0 -> f=0."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    s1 = np.sum(x * x) / n
    s2 = np.sum(np.cos(2.0 * np.pi * x)) / n
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e)


def griewank(x: NDArray | Sequence[float]) -> float:
    """Griewank function. Minimum at x=0 -> f=0."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    idx = np.arange(1, x.size + 1)
    s = np.sum(x * x) / 4000.0
    p = float(np.prod(np.cos(x / np.sqrt(idx))))
    return float(1.0 + s - p)


def schwefel(x: NDArray | Sequence[float]) -> float:
    """Schwefel function (classic formulation). Known minimum near 420.9687."""
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def knapsack_dp(
    values: Sequence[int], weights: Sequence[int], capacity: int
) -> tuple[int, NDArray]:
    """0/1 knapsack via dynamic programming.

    Returns a tuple ``(best_value, selection_array)`` where ``selection_array``
    is a 0/1 numpy array indicating chosen items.
    """
    vals = np.asarray(values, dtype=int)
    wts = np.asarray(weights, dtype=int)
    n = int(vals.size)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    for i in range(1, n + 1):
        w_i = int(wts[i - 1])
        v_i = int(vals[i - 1])
        for c in range(capacity + 1):
            dp[i, c] = dp[i - 1, c]
            if w_i <= c and dp[i - 1, c - w_i] + v_i > dp[i, c]:
                dp[i, c] = dp[i - 1, c - w_i] + v_i

    sol = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            sol[i - 1] = 1
            c -= int(wts[i - 1])

    return int(dp[n, capacity]), sol


def knapsack_penalty(
    x: NDArray | Sequence[float],
    values: Sequence[int],
    weights: Sequence[int],
    capacity: int,
    overflow_penalty: float = 1000.0,
) -> float:
    """Penalty-based objective for knapsack used in examples (continuous -> binary).

    Interprets continuous `x` as selection via threshold 0.5 and returns
    ``-value + penalty*overflow`` so that lower is better (for minimizers).
    """
    x = np.asarray(x, dtype=float)
    sel = (x > 0.5).astype(int)
    vals = np.asarray(values)
    wts = np.asarray(weights)
    total_value = float(np.sum(sel * vals))
    total_weight = float(np.sum(sel * wts))
    overflow = max(0.0, total_weight - capacity)
    return float(-total_value + overflow_penalty * overflow)


def qap_cost(x: NDArray | Sequence[float], flow: NDArray, dist: NDArray) -> float:
    """QAP cost for random-keys encoding: permutation = argsort(x)."""
    x = np.asarray(x)
    pi = np.argsort(x)
    return float(np.sum(np.asarray(flow) * np.asarray(dist)[np.ix_(pi, pi)]))


def g06(x: NDArray | Sequence[float]) -> float:
    """G06 constrained problem encoded with quadratic external penalty (as in examples)."""
    x1, x2 = float(x[0]), float(x[1])
    obj = (x1 - 10.0) ** 3 + (x2 - 20.0) ** 3
    v1 = max(0.0, -((x1 - 5.0) ** 2) - (x2 - 5.0) ** 2 + 100.0)
    v2 = max(0.0, (x1 - 6.0) ** 2 + (x2 - 5.0) ** 2 - 82.81)
    return float(obj + 1e6 * (v1 * v1 + v2 * v2))


__all__ = [
    "ackley",
    "g06",
    "griewank",
    "knapsack_dp",
    "knapsack_penalty",
    "qap_cost",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sphere",
]
