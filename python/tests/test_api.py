"""Tests for the public ``givp`` API (``givp`` and ``GIVPOptimizer``)."""

from __future__ import annotations

import numpy as np
import pytest
from givp import (
    GIVPConfig,
    GIVPOptimizer,
    InvalidBoundsError,
    InvalidInitialGuessError,
    OptimizeResult,
    givp,
)
from givp.core.grasp import _validate_bounds_and_initial


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def neg_sphere(x: np.ndarray) -> float:
    return -float(np.sum(x**2))


@pytest.fixture(name="fast_config")
def fixture_fast_config() -> GIVPConfig:
    return GIVPConfig(
        max_iterations=4,
        vnd_iterations=8,
        ils_iterations=2,
        elite_size=3,
        path_relink_frequency=2,
        num_candidates_per_step=4,
        early_stop_threshold=10,
        use_convergence_monitor=False,
    )


# ----------------------------- core happy paths -----------------------------


def test_minimize_sphere_returns_result(fast_config):
    bounds = [(-5.0, 5.0)] * 4
    result = givp(sphere, bounds, config=fast_config)
    assert isinstance(result, OptimizeResult)
    assert result.direction == "minimize"
    assert result.x.shape == (4,)
    assert np.isfinite(result.fun)
    assert result.nfev > 0


def test_maximize_returns_value_in_original_sign(fast_config):
    bounds = [(-5.0, 5.0)] * 4
    result = givp(neg_sphere, bounds, direction="maximize", config=fast_config)
    assert result.direction == "maximize"
    assert result.fun <= 0.0
    assert np.isfinite(result.fun)


def test_bounds_as_lower_upper_tuple(fast_config):
    lower = [-1.0, -1.0, -1.0]
    upper = [1.0, 1.0, 1.0]
    result = givp(sphere, (lower, upper), config=fast_config)
    assert result.x.shape == (3,)


def test_optimizer_class_keeps_history(fast_config):
    opt = GIVPOptimizer(sphere, [(-2.0, 2.0)] * 3, config=fast_config)
    r1 = opt.run()
    r2 = opt.run()
    assert len(opt.history) == 2
    assert opt.best_fun == min(r1.fun, r2.fun)
    assert opt.best_x is not None


def test_grasp_optimizer_maximize_tracks_best(fast_config):
    opt = GIVPOptimizer(
        neg_sphere,
        [(-1.0, 1.0)] * 2,
        direction="maximize",
        config=fast_config,
    )
    opt.run()
    opt.run()
    assert opt.best_x is not None
    assert len(opt.history) == 2


def test_result_is_iterable_for_legacy_unpacking(fast_config):
    result = givp(sphere, [(-1.0, 1.0)] * 2, config=fast_config)
    x, fun = result
    assert np.allclose(x, result.x)
    assert fun == result.fun


# ----------------------------- error handling -----------------------------


def test_objective_returning_nan_is_handled(fast_config):
    def nan_func(_x):
        return float("nan")

    result = givp(nan_func, [(0.0, 1.0)] * 2, config=fast_config)
    assert not result.success


def test_evaluator_raising_exception_is_handled(fast_config):
    def boom(_x):
        raise RuntimeError("explode")

    result = givp(boom, [(0.0, 1.0)] * 2, config=fast_config)
    assert not result.success


def test_invalid_direction_raises(fast_config):
    with pytest.raises(ValueError):
        givp(sphere, [(0.0, 1.0)], direction="bogus", config=fast_config)


def test_minimize_and_direction_conflict_raises(fast_config):
    with pytest.raises(ValueError):
        givp(
            sphere,
            [(0.0, 1.0)] * 2,
            minimize=True,
            direction="maximize",
            config=fast_config,
        )


def test_bounds_num_vars_mismatch_raises(fast_config):
    with pytest.raises(ValueError):
        givp(sphere, [(0.0, 1.0)] * 2, num_vars=5, config=fast_config)


def test_bounds_none_raises():
    with pytest.raises(ValueError):
        givp(sphere, None)  # type: ignore[arg-type]


def test_invalid_initial_guess_length_raises(fast_config):
    with pytest.raises(InvalidInitialGuessError):
        givp(
            sphere,
            [(-1.0, 1.0)] * 3,
            config=fast_config,
            initial_guess=[0.0, 0.0],
        )


def test_invalid_initial_guess_outside_bounds_raises(fast_config):
    with pytest.raises(InvalidInitialGuessError):
        givp(
            sphere,
            [(-1.0, 1.0)] * 3,
            config=fast_config,
            initial_guess=[5.0, 5.0, 5.0],
        )


def test_invalid_bounds_via_core_validate():
    with pytest.raises(InvalidBoundsError):
        _validate_bounds_and_initial(
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            None,
            num_vars=3,
        )


def test_evaluator_raising_value_error_in_wrapper(fast_config):
    """``_wrap_objective`` catches ValueError and returns +inf."""

    def bad(_x):
        raise ValueError("nope")

    result = givp(bad, [(0.0, 1.0)] * 2, config=fast_config)
    assert not result.success


# ----------------------------- config code paths -----------------------------


def test_initial_guess_warm_start(fast_config):
    bounds = [(-3.0, 3.0)] * 3
    initial = [0.1, 0.1, 0.1]
    result = givp(sphere, bounds, config=fast_config, initial_guess=initial)
    assert result.x.shape == (3,)
    assert np.isfinite(result.fun)


def test_iteration_callback_is_invoked(fast_config):
    calls = []

    def cb(it, cost, sol):
        calls.append((it, float(cost), np.array(sol)))

    givp(sphere, [(-1.0, 1.0)] * 2, config=fast_config, iteration_callback=cb)
    assert len(calls) >= 1


def test_iteration_callback_exception_is_swallowed(fast_config):
    def cb(_it, _cost, _sol):
        raise RuntimeError("callback boom")

    result = givp(
        sphere,
        [(-1.0, 1.0)] * 2,
        config=fast_config,
        iteration_callback=cb,
        verbose=True,
    )
    assert np.isfinite(result.fun)


def test_use_cache_path(fast_config):
    cfg = GIVPConfig(**{**fast_config.__dict__, "use_cache": True})
    result = givp(sphere, [(-1.0, 1.0)] * 2, config=cfg)
    assert np.isfinite(result.fun)


def test_no_cache_path(fast_config):
    cfg = GIVPConfig(**{**fast_config.__dict__, "use_cache": False})
    result = givp(sphere, [(-1.0, 1.0)] * 2, config=cfg)
    assert np.isfinite(result.fun)


def test_adaptive_alpha_disabled(fast_config):
    cfg = GIVPConfig(**{**fast_config.__dict__, "adaptive_alpha": False})
    result = givp(sphere, [(-1.0, 1.0)] * 2, config=cfg)
    assert np.isfinite(result.fun)


def test_convergence_monitor_enabled(fast_config):
    cfg = GIVPConfig(
        **{
            **fast_config.__dict__,
            "use_convergence_monitor": True,
            "early_stop_threshold": 2,
        }
    )
    result = givp(sphere, [(-1.0, 1.0)] * 2, config=cfg)
    assert np.isfinite(result.fun)


def test_n_workers_parallel_path(fast_config):
    cfg = GIVPConfig(**{**fast_config.__dict__, "n_workers": 2})
    result = givp(sphere, [(-1.0, 1.0)] * 3, config=cfg)
    assert np.isfinite(result.fun)


def test_n_workers_parity_serial_vs_parallel(fast_config):
    """n_workers=2 must return a finite, valid result for the same objective.

    Strict value equality is not guaranteed because thread scheduling affects
    the order in which candidates are evaluated, which can alter the random
    state and thus the trajectory.  The test verifies that:
    - Both executions return finite objective values.
    - The parallel result is within a reasonable absolute tolerance of the
      serial result (same seed, same low-iteration budget).
    - The parallel result respects bounds (sphere minimum is 0).

    Note: speedup from n_workers>1 requires the objective to release the GIL
    (e.g., NumPy/SciPy internals).  Pure-Python objectives run serially inside
    ThreadPoolExecutor due to the GIL.
    """
    bounds = [(-1.0, 1.0)] * 4
    cfg_serial = GIVPConfig(**{**fast_config.__dict__, "n_workers": 1})
    cfg_parallel = GIVPConfig(**{**fast_config.__dict__, "n_workers": 2})

    r_serial = givp(sphere, bounds, config=cfg_serial, seed=42)
    r_parallel = givp(sphere, bounds, config=cfg_parallel, seed=42)

    assert np.isfinite(r_serial.fun), "serial result must be finite"
    assert np.isfinite(r_parallel.fun), "parallel result must be finite"
    assert r_serial.fun >= 0.0, "sphere minimum is 0; serial result must be non-negative"
    assert r_parallel.fun >= 0.0, "sphere minimum is 0; parallel result must be non-negative"
    # Both should achieve a similar quality bound: within 10x of each other.
    assert r_parallel.fun < r_serial.fun * 10 + 1.0, (
        f"parallel result ({r_parallel.fun:.4f}) is unexpectedly much worse than "
        f"serial ({r_serial.fun:.4f})"
    )


def test_time_limit_triggers_early_stop(fast_config):
    cfg = GIVPConfig(
        **{
            **fast_config.__dict__,
            "max_iterations": 10_000,
            "vnd_iterations": 10_000,
            "ils_iterations": 50,
            "time_limit": 0.05,
        }
    )
    result = givp(sphere, [(-1.0, 1.0)] * 3, config=cfg)
    assert np.isfinite(result.fun)


def test_verbose_runs_without_error(fast_config, caplog):
    with caplog.at_level("INFO"):
        result = givp(sphere, [(-1.0, 1.0)] * 2, config=fast_config, verbose=True)
    assert np.isfinite(result.fun)


def test_long_run_triggers_path_relinking_and_restart(fast_config):
    cfg = GIVPConfig(
        max_iterations=8,
        vnd_iterations=6,
        ils_iterations=2,
        elite_size=4,
        path_relink_frequency=1,
        num_candidates_per_step=4,
        perturbation_strength=2,
        adaptive_alpha=True,
        use_cache=True,
        use_convergence_monitor=True,
        early_stop_threshold=100,
    )
    result = givp(sphere, [(-2.0, 2.0)] * 4, config=cfg, verbose=True)
    assert np.isfinite(result.fun)


# ----------------------------- _wrap_objective coverage --------------------


def test_wrap_objective_invalid_direction_raises():
    """`_wrap_objective` raises ValueError for an unknown direction string."""
    from givp.api import _wrap_objective

    with pytest.raises(ValueError, match="direction must be"):
        _wrap_objective(sphere, "sideways", [0])


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_wrap_objective_valid_directions(direction):
    """`_wrap_objective` accepts both valid direction strings."""
    from givp.api import _wrap_objective

    counter: list[int] = [0]
    wrapped = _wrap_objective(sphere, direction, counter)
    val = wrapped(np.array([1.0, 2.0]))
    assert np.isfinite(val)
    assert counter[0] == 1


# ----------------------------- integer_split pre-set ----------------------


def test_integer_split_preset_is_respected(fast_config):
    """When ``integer_split`` is already set on the config the branch that
    auto-fills it from ``n`` must NOT overwrite it (line 178 false-branch)."""
    cfg = GIVPConfig(**{**fast_config.__dict__, "integer_split": 2})
    # 4-variable problem but integer_split=2 pre-set — should not be overwritten
    result = givp(sphere, [(-2.0, 2.0)] * 4, config=cfg)
    assert np.isfinite(result.fun)


def test_grasp_optimizer_run_second_call_not_better(monkeypatch, fast_config):
    """Line 269->272: second run() result is NOT better -> best_fun/best_x unchanged."""
    from givp import api as api_mod
    from givp.result import OptimizeResult

    call_count = [0]
    results = [
        OptimizeResult(
            x=np.zeros(2),
            fun=0.5,
            nit=1,
            nfev=1,
            success=True,
            message="ok",
            direction="minimize",
        ),
        OptimizeResult(
            x=np.ones(2),
            fun=2.0,
            nit=1,
            nfev=1,
            success=True,
            message="ok",
            direction="minimize",
        ),
    ]

    def fake_run(*_args, **_kwargs):
        r = results[call_count[0]]
        call_count[0] += 1
        return r

    monkeypatch.setattr(api_mod, "givp", fake_run)

    opt = GIVPOptimizer(sphere, [(-1.0, 1.0)] * 2, config=fast_config)
    opt.run()  # best_fun set to 0.5
    opt.run()  # 2.0 is NOT better -> best_fun stays 0.5
    assert opt.best_fun == pytest.approx(0.5)
