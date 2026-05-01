"""Property-based tests using Hypothesis.

These tests exercise the public API with randomly generated inputs to surface
edge cases that the deterministic suite in ``test_api.py`` may miss.
"""

from __future__ import annotations

import numpy as np
import pytest
from givp import (
    GIVPConfig,
    InvalidBoundsError,
    InvalidInitialGuessError,
    givp,
)
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

_FAST = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _bounds_strategy(min_dim: int = 1, max_dim: int = 5):
    """Generate well-formed scipy-style bounds with strictly positive width."""

    def _build(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(lo, lo + max(width, 1e-3)) for lo, width in pairs]

    return st.lists(
        st.tuples(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
            st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        ),
        min_size=min_dim,
        max_size=max_dim,
    ).map(_build)


@_FAST
@given(bounds=_bounds_strategy())
def test_sphere_returns_solution_within_bounds(bounds: list[tuple[float, float]]):
    """For any valid bounds, the result vector lies inside them."""

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    cfg = GIVPConfig(max_iterations=2, vnd_iterations=3)
    result = givp(sphere, bounds, config=cfg)

    assert result.x.shape == (len(bounds),)
    lower = np.array([lo for lo, _ in bounds])
    upper = np.array([hi for _, hi in bounds])
    assert np.all(result.x >= lower - 1e-9)
    assert np.all(result.x <= upper + 1e-9)
    assert np.isfinite(result.fun)
    assert result.nfev > 0


@_FAST
@given(bounds=_bounds_strategy(min_dim=2, max_dim=4))
def test_direction_flip_preserves_magnitude(bounds: list[tuple[float, float]]):
    """Optimizing ``f`` and ``-f`` should yield results of compatible sign."""

    def f(x: np.ndarray) -> float:
        return float(np.sum(np.sin(x)))

    cfg = GIVPConfig(max_iterations=2, vnd_iterations=3)
    res_min = givp(f, bounds, direction="minimize", config=cfg)
    res_max = givp(f, bounds, direction="maximize", config=cfg)

    # Both optima evaluate to the user's original sign.
    assert np.isfinite(res_min.fun)
    assert np.isfinite(res_max.fun)
    # Minimizer should not be strictly worse than the maximizer's minimum:
    # i.e., the minimizer's value must be <= the maximizer's value, allowing
    # algorithmic noise (these are stochastic searches).
    assert res_min.fun <= res_max.fun + 1e-3


@_FAST
@given(
    bounds=_bounds_strategy(min_dim=1, max_dim=3),
    extra=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
)
def test_initial_guess_outside_bounds_raises(
    bounds: list[tuple[float, float]], extra: float
):
    """``initial_guess`` strictly outside the bounds must raise."""

    def f(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    bad_guess = [hi + extra for _, hi in bounds]
    with pytest.raises(InvalidInitialGuessError):
        givp(f, bounds, initial_guess=bad_guess)


@_FAST
@given(
    n=st.integers(min_value=1, max_value=4),
    delta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
)
def test_invalid_bounds_inverted_raises(n: int, delta: float):
    """Bounds where ``lower > upper`` must be rejected."""

    bounds = [(1.0, 1.0 - delta)] * n
    assume(any(lo > hi for lo, hi in bounds))

    def f(x: np.ndarray) -> float:
        return float(np.sum(x))

    with pytest.raises((InvalidBoundsError, ValueError)):
        givp(f, bounds)


@_FAST
@given(bounds=_bounds_strategy(min_dim=1, max_dim=3))
def test_nfev_matches_or_exceeds_iterations(bounds: list[tuple[float, float]]):
    """Each outer iteration must produce at least one evaluation."""

    def f(x: np.ndarray) -> float:
        return float(np.sum(np.abs(x)))

    cfg = GIVPConfig(max_iterations=3, vnd_iterations=2)
    result = givp(f, bounds, config=cfg)
    assert result.nfev >= result.nit
    assert result.nit >= 1
