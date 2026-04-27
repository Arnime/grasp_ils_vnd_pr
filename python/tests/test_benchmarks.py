"""Unit tests for the benchmark functions in `givp.benchmarks`.

These tests assert that the classic functions return their known minima
at the canonical inputs (e.g. zeros or ones) and exercise the knapsack DP
helper with a small instance.
"""

from __future__ import annotations

import numpy as np
import pytest
from givp import benchmarks


def test_sphere_zero():
    assert benchmarks.sphere(np.zeros(3)) == pytest.approx(0.0)


def test_rosenbrock_ones():
    assert benchmarks.rosenbrock(np.ones(5)) == pytest.approx(0.0)


def test_rastrigin_zero():
    assert benchmarks.rastrigin(np.zeros(4)) == pytest.approx(0.0)


def test_ackley_zero():
    assert benchmarks.ackley(np.zeros(6)) == pytest.approx(0.0)


def test_griewank_zero():
    assert benchmarks.griewank(np.zeros(5)) == pytest.approx(0.0)


def test_schwefel_known_optimum():
    # Schwefel has known minimum near 420.9687 per-coordinate (value ~ 0)
    x = np.full(3, 420.9687)
    assert benchmarks.schwefel(x) == pytest.approx(0.0, abs=1e-3)


def test_knapsack_dp_small(knapsack_values, knapsack_weights, knapsack_capacity):
    val, sel = benchmarks.knapsack_dp(
        knapsack_values, knapsack_weights, knapsack_capacity
    )
    assert val == 220
    assert np.array_equal(sel, np.array([0, 1, 1]))


def test_knapsack_penalty_selection(
    knapsack_values, knapsack_weights, knapsack_capacity
):
    x = np.array([0.0, 1.0, 1.0])
    val = benchmarks.knapsack_penalty(
        x, knapsack_values, knapsack_weights, knapsack_capacity, overflow_penalty=1000.0
    )
    assert val == pytest.approx(-220.0)


def test_qap_cost_matches_manual(qap_flow, qap_dist):
    x = np.array([0.2, 0.1])  # permutation [1, 0]
    cost = benchmarks.qap_cost(x, qap_flow, qap_dist)
    assert cost == pytest.approx(10.0)


def test_rosenbrock_short_vector():
    # exercise the x.size < 2 branch
    assert benchmarks.rosenbrock(np.array([1.0])) == pytest.approx(0.0)


def test_ackley_empty_vector():
    # exercise the n == 0 branch
    assert benchmarks.ackley(np.array([])) == pytest.approx(0.0)


def test_griewank_empty_vector():
    # exercise the x.size == 0 branch
    assert benchmarks.griewank(np.array([])) == pytest.approx(1.0)


def test_g06_is_finite():
    # ensure g06 executes and returns a finite float for a sample input
    val = benchmarks.g06(np.array([10.0, 20.0]))
    assert isinstance(val, float)
    assert np.isfinite(val)
