"""Smoke tests for the public ``givp`` API."""

from __future__ import annotations

import numpy as np
import pytest

from givp import GraspIlsVndConfig, GraspOptimizer, OptimizeResult, grasp_ils_vnd_pr


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def neg_sphere(x: np.ndarray) -> float:
    return -float(np.sum(x ** 2))


@pytest.fixture
def fast_config() -> GraspIlsVndConfig:
    return GraspIlsVndConfig(
        max_iterations=5,
        vnd_iterations=10,
        ils_iterations=2,
        elite_size=3,
        path_relink_frequency=2,
        num_candidates_per_step=5,
        early_stop_threshold=10,
        use_convergence_monitor=False,
    )


def test_minimize_sphere_returns_result(fast_config):
    bounds = [(-5.0, 5.0)] * 4
    result = grasp_ils_vnd_pr(sphere, bounds, config=fast_config)
    assert isinstance(result, OptimizeResult)
    assert result.direction == "minimize"
    assert result.x.shape == (4,)
    assert np.isfinite(result.fun)
    assert result.nfev > 0


def test_maximize_returns_value_in_original_sign(fast_config):
    bounds = [(-5.0, 5.0)] * 4
    result = grasp_ils_vnd_pr(neg_sphere, bounds, direction="maximize", config=fast_config)
    assert result.direction == "maximize"
    # neg_sphere is always <= 0; max should be near 0.
    assert result.fun <= 0.0
    assert np.isfinite(result.fun)


def test_bounds_as_lower_upper_tuple(fast_config):
    lower = [-1.0, -1.0, -1.0]
    upper = [1.0, 1.0, 1.0]
    result = grasp_ils_vnd_pr(sphere, (lower, upper), config=fast_config)
    assert result.x.shape == (3,)


def test_invalid_direction_raises(fast_config):
    with pytest.raises(ValueError):
        grasp_ils_vnd_pr(sphere, [(0.0, 1.0)], direction="bogus", config=fast_config)


def test_optimizer_class_keeps_history(fast_config):
    opt = GraspOptimizer(sphere, [(-2.0, 2.0)] * 3, config=fast_config)
    r1 = opt.run()
    r2 = opt.run()
    assert len(opt.history) == 2
    assert opt.best_fun == min(r1.fun, r2.fun)
    assert opt.best_x is not None


def test_result_is_iterable_for_legacy_unpacking(fast_config):
    result = grasp_ils_vnd_pr(sphere, [(-1.0, 1.0)] * 2, config=fast_config)
    x, fun = result
    assert np.allclose(x, result.x)
    assert fun == result.fun


def test_objective_returning_nan_is_handled(fast_config):
    def nan_func(_x):
        return float("nan")

    result = grasp_ils_vnd_pr(nan_func, [(0.0, 1.0)] * 2, config=fast_config)
    # NaN is coerced to +inf inside, so success should be False.
    assert not result.success
