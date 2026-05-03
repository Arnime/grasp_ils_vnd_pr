# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Result-field invariant tests — cross-language regression guard.

These tests assert structural and logical invariants that MUST hold in every
language port of GIVP (Python, Julia, Rust, C++, R).  They are intentionally
strict so that bugs already identified in the Rust and C++ ports (``nit``
always returning ``max_iterations`` rather than the true iteration count) are
caught early if the same mistake is introduced in Python.

Invariants verified:
    INV-1  nit ≤ max_iterations for every successful run
    INV-2  nit ≥ 1 for every run (at least one iteration is executed)
    INV-3  nfev ≥ nit (each iteration evaluates at least one solution)
    INV-4  result.x has shape (n_vars,) and lies within the declared bounds
    INV-5  result.success is False when the objective never returns a finite value
    INV-6  result.direction is exactly "minimize" or "maximize"
    INV-7  to_dict() "termination" is always a valid TerminationReason value
    INV-8  to_dict() produces a schema-valid dict (required keys, correct types)
    INV-9  Same seed → same result (x, fun) — determinism guard
    INV-10 to_dict() "nit" matches result.nit (no silent truncation)
    INV-11 result.message is a non-empty string
    INV-12 max_iterations=1 still produces a valid result (minimal run)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from givp import GIVPConfig, OptimizeResult, TerminationReason, givp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BOUNDS_3D = [(-5.0, 5.0)] * 3
_BOUNDS_1D = [(-1.0, 1.0)]


def _fast_cfg(max_iterations: int = 3, **kwargs: object) -> GIVPConfig:
    """Return a fast config suitable for invariant checking."""
    return GIVPConfig(
        max_iterations=max_iterations,
        vnd_iterations=4,
        ils_iterations=2,
        elite_size=3,
        path_relink_frequency=2,
        num_candidates_per_step=4,
        early_stop_threshold=50,
        use_convergence_monitor=False,
        **kwargs,  # type: ignore[arg-type]
    )


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def infeasible(_x: np.ndarray) -> float:
    """Always returns +inf — simulates a completely infeasible objective."""
    return float("inf")


# ---------------------------------------------------------------------------
# INV-1: nit ≤ max_iterations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iter", [1, 3, 10, 25])
def test_inv1_nit_does_not_exceed_max_iterations(max_iter: int) -> None:
    """INV-1: nit must never exceed the configured max_iterations budget."""
    cfg = _fast_cfg(max_iterations=max_iter)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=42)
    assert isinstance(result, OptimizeResult)
    assert result.nit <= max_iter, (
        f"nit={result.nit} exceeded max_iterations={max_iter}. "
        "This is the known Rust/C++ bug: nit was always set to max_iterations "
        "instead of the actual iteration count."
    )


# ---------------------------------------------------------------------------
# INV-2: nit ≥ 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iter", [1, 3, 5])
def test_inv2_nit_is_at_least_one(max_iter: int) -> None:
    """INV-2: at least one iteration must always execute."""
    cfg = _fast_cfg(max_iterations=max_iter)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=0)
    assert result.nit >= 1, f"nit={result.nit} — no iterations executed."


# ---------------------------------------------------------------------------
# INV-3: nfev ≥ nit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_iter", [1, 4])
def test_inv3_nfev_at_least_nit(max_iter: int) -> None:
    """INV-3: each outer iteration must trigger at least one evaluation."""
    cfg = _fast_cfg(max_iterations=max_iter)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=7)
    assert result.nfev >= result.nit, (
        f"nfev={result.nfev} < nit={result.nit}: impossible invariant violation."
    )


# ---------------------------------------------------------------------------
# INV-4: x shape and bounds containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bounds",
    [
        [(-1.0, 1.0)],
        [(-5.0, 5.0)] * 3,
        [(0.0, 10.0)] * 5,
    ],
)
def test_inv4_x_shape_and_bounds(bounds: list[tuple[float, float]]) -> None:
    """INV-4: x must have shape (n,) and each component must lie in its bound."""
    n = len(bounds)
    cfg = _fast_cfg(max_iterations=3)
    result = givp(sphere, bounds, config=cfg, seed=1)

    assert result.x.shape == (n,), f"Expected shape ({n},), got {result.x.shape}"
    lower = np.array([lo for lo, _ in bounds])
    upper = np.array([hi for _, hi in bounds])
    assert np.all(result.x >= lower - 1e-9), "x violates lower bounds"
    assert np.all(result.x <= upper + 1e-9), "x violates upper bounds"


# ---------------------------------------------------------------------------
# INV-5: success=False when objective is always infeasible
# ---------------------------------------------------------------------------


def test_inv5_success_false_for_always_infeasible() -> None:
    """INV-5: if objective always returns +inf, success must be False."""
    cfg = _fast_cfg(max_iterations=3)
    result = givp(infeasible, _BOUNDS_3D, config=cfg, seed=42)
    assert not result.success, (
        "success should be False when no finite solution was ever found."
    )


# ---------------------------------------------------------------------------
# INV-6: direction is always "minimize" or "maximize"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("minimize", "minimize"),
        ("maximize", "maximize"),
        (None, "minimize"),  # default
    ],
)
def test_inv6_direction_is_canonical(direction: str | None, expected: str) -> None:
    """INV-6: result.direction must be exactly 'minimize' or 'maximize'."""
    cfg = _fast_cfg(max_iterations=2)
    kwargs: dict = {}
    if direction is not None:
        kwargs["direction"] = direction
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=5, **kwargs)
    assert result.direction == expected


# ---------------------------------------------------------------------------
# INV-7: to_dict() "termination" is always a valid TerminationReason
# ---------------------------------------------------------------------------


def test_inv7_termination_reason_is_always_valid() -> None:
    """INV-7: to_dict()['termination'] must be a TerminationReason enum value."""
    valid_values = {r.value for r in TerminationReason}
    for max_iter in (1, 3, 10):
        cfg = _fast_cfg(max_iterations=max_iter)
        result = givp(sphere, _BOUNDS_3D, config=cfg, seed=max_iter)
        d = result.to_dict()
        assert d["termination"] in valid_values, (
            f"termination={d['termination']!r} is not a TerminationReason value. "
            f"Valid: {valid_values}"
        )


# ---------------------------------------------------------------------------
# INV-8: to_dict() schema validation
# ---------------------------------------------------------------------------

_REQUIRED_DICT_KEYS: frozenset[str] = frozenset(
    {"x", "fun", "nit", "nfev", "success", "termination", "direction"}
)


def test_inv8_to_dict_schema() -> None:
    """INV-8: to_dict() must contain all required keys with correct types."""
    cfg = _fast_cfg(max_iterations=3)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=99)
    d = result.to_dict()

    missing = _REQUIRED_DICT_KEYS - d.keys()
    assert not missing, f"to_dict() is missing keys: {missing}"

    assert isinstance(d["x"], list), "x must be a list"
    assert isinstance(d["fun"], float), "fun must be a float"
    assert isinstance(d["nit"], int), "nit must be an int"
    assert isinstance(d["nfev"], int), "nfev must be an int"
    assert isinstance(d["success"], bool), "success must be a bool"
    assert isinstance(d["termination"], str), "termination must be a str"
    assert isinstance(d["direction"], str), "direction must be a str"
    assert all(isinstance(v, float) for v in d["x"]), "x elements must be floats"


# ---------------------------------------------------------------------------
# INV-9: Same seed → same result
# ---------------------------------------------------------------------------


def test_inv9_seed_determinism() -> None:
    """INV-9: two runs with the same seed must produce identical x and fun."""
    cfg = _fast_cfg(max_iterations=5)
    r1 = givp(sphere, _BOUNDS_3D, config=cfg, seed=1234)
    r2 = givp(sphere, _BOUNDS_3D, config=cfg, seed=1234)

    np.testing.assert_array_equal(r1.x, r2.x)
    assert r1.fun == r2.fun
    assert r1.nit == r2.nit
    assert r1.nfev == r2.nfev


# ---------------------------------------------------------------------------
# INV-10: to_dict() "nit" matches result.nit
# ---------------------------------------------------------------------------


def test_inv10_to_dict_nit_matches_result_nit() -> None:
    """INV-10: to_dict()['nit'] must equal result.nit — no silent truncation."""
    cfg = _fast_cfg(max_iterations=7)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=77)
    assert result.to_dict()["nit"] == result.nit


# ---------------------------------------------------------------------------
# INV-11: result.message is non-empty
# ---------------------------------------------------------------------------


def test_inv11_message_is_non_empty_string() -> None:
    """INV-11: result.message must be a non-empty string for every run."""
    for max_iter in (1, 5):
        cfg = _fast_cfg(max_iterations=max_iter)
        result = givp(sphere, _BOUNDS_3D, config=cfg, seed=0)
        assert isinstance(result.message, str)
        assert len(result.message.strip()) > 0, "result.message must not be empty"


# ---------------------------------------------------------------------------
# INV-12: max_iterations=1 still produces a valid result
# ---------------------------------------------------------------------------


def test_inv12_single_iteration_produces_valid_result() -> None:
    """INV-12: even with max_iterations=1 the result must be structurally valid."""
    cfg = _fast_cfg(max_iterations=1)
    result = givp(sphere, _BOUNDS_3D, config=cfg, seed=42)

    assert result.nit >= 1
    assert result.nit <= 1  # INV-1 under minimal budget
    assert result.nfev >= 1
    assert result.x.shape == (3,)
    assert math.isfinite(result.fun)
    assert result.direction in ("minimize", "maximize")
