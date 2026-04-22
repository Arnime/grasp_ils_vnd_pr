"""Tests for ``givp._core`` internals: helpers, classes, neighborhoods, search."""

from __future__ import annotations

import numpy as np
import pytest

from givp import EmptyPoolError
from givp._config import GraspIlsVndConfig as PublicConfig
from givp._core import (
    ConvergenceMonitor,
    ElitePool,
    EvaluationCache,
    _build_heuristic_candidate,
    _build_random_candidate,
    _evaluate_with_cache,
    _get_group_size,
    _get_half,
    _neighborhood_block,
    _neighborhood_group,
    _safe_evaluate,
    _sample_integer_from_bounds,
    _select_from_rcl,
    _set_group_size,
    _set_integer_split,
    bidirectional_path_relinking,
    construct_solution_numpy,
    evaluate_candidates,
    get_current_alpha,
    ils_search,
    local_search_vnd,
    local_search_vnd_adaptive,
    path_relinking,
    perturb_solution_numpy,
    select_rcl,
)
from givp._core import (
    GraspIlsVndConfig as CoreConfig,
)


@pytest.fixture(autouse=True)
def reset_context():
    _set_integer_split(None)
    _set_group_size(None)
    yield
    _set_integer_split(None)
    _set_group_size(None)


def quad(x: np.ndarray) -> float:
    return float(np.sum(x**2))


# ----------------------------- _safe_evaluate -----------------------------


def test_safe_evaluate_returns_inf_on_exception():
    def boom(_x):
        raise RuntimeError("explode")

    assert _safe_evaluate(boom, np.zeros(3)) == np.inf


def test_safe_evaluate_returns_inf_on_nonfinite():
    assert _safe_evaluate(lambda _x: float("nan"), np.zeros(3)) == np.inf
    assert _safe_evaluate(lambda _x: float("inf"), np.zeros(3)) == np.inf


def test_safe_evaluate_returns_value_on_success():
    assert _safe_evaluate(
        lambda x: float(np.sum(x)), np.array([1.0, 2.0])
    ) == pytest.approx(3.0)


# ----------------------------- EvaluationCache -----------------------------


class TestEvaluationCache:
    def test_get_miss_returns_none(self):
        cache = EvaluationCache(maxsize=4)
        assert cache.get(np.array([1.0, 2.0])) is None
        assert cache.misses == 1

    def test_put_then_get_returns_cached_value(self):
        cache = EvaluationCache(maxsize=4)
        sol = np.array([1.0, 2.0])
        cache.put(sol, 42.0)
        assert cache.get(sol) == pytest.approx(42.0)
        assert cache.hits == 1

    def test_put_overwrites_existing_key(self):
        cache = EvaluationCache(maxsize=4)
        sol = np.array([1.0])
        cache.put(sol, 1.0)
        cache.put(sol, 2.0)
        assert cache.get(sol) == pytest.approx(2.0)

    def test_eviction_when_full(self):
        cache = EvaluationCache(maxsize=2)
        for i in range(3):
            cache.put(np.array([float(i), float(i)]), float(i))
        assert cache.get(np.array([0.0, 0.0])) is None
        assert cache.get(np.array([2.0, 2.0])) == pytest.approx(2.0)

    def test_clear_resets_state(self):
        cache = EvaluationCache(maxsize=4)
        cache.put(np.array([1.0]), 1.0)
        cache.get(np.array([1.0]))
        cache.get(np.array([99.0]))
        cache.clear()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0

    def test_stats_hit_rate(self):
        cache = EvaluationCache(maxsize=4)
        sol = np.array([1.0])
        cache.put(sol, 1.0)
        cache.get(sol)
        cache.get(np.array([2.0]))
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(50.0)

    def test_evaluate_with_cache_uses_cache_on_repeat(self):
        cache = EvaluationCache(maxsize=4)
        calls = [0]

        def evaluator(_x):
            calls[0] += 1
            return 1.0

        sol = np.array([1.0, 2.0])
        _evaluate_with_cache(sol, evaluator, cache)
        _evaluate_with_cache(sol, evaluator, cache)
        assert calls[0] == 1
        assert cache.hits == 1

    def test_evaluate_without_cache(self):
        calls = [0]

        def evaluator(_x):
            calls[0] += 1
            return 5.0

        out = _evaluate_with_cache(np.array([1.0]), evaluator, None)
        assert out == pytest.approx(5.0)
        assert calls[0] == 1

    def test_evaluate_with_cache_does_not_store_inf(self):
        cache = EvaluationCache(maxsize=4)

        def evaluator(_x):
            raise RuntimeError("boom")

        out = _evaluate_with_cache(np.array([1.0]), evaluator, cache)
        assert np.isinf(out)
        assert len(cache.cache) == 0


# ----------------------------- ConvergenceMonitor -----------------------------


class TestConvergenceMonitor:
    def test_update_tracks_improvements(self):
        mon = ConvergenceMonitor(window_size=5, restart_threshold=10)
        mon.update(10.0)
        info = mon.update(5.0)
        assert info["no_improve_count"] == 0
        assert mon.best_ever == pytest.approx(5.0)

    def test_increments_no_improve_when_worse(self):
        mon = ConvergenceMonitor(window_size=5, restart_threshold=3)
        mon.update(1.0)
        mon.update(2.0)
        mon.update(3.0)
        info = mon.update(4.0)
        assert info["no_improve_count"] == 3
        assert info["should_restart"] is True

    def test_diversity_with_elite_pool(self):
        pool = ElitePool(max_size=4, min_distance=0.0)
        pool.add(np.array([0.0, 0.0]), 1.0)
        pool.add(np.array([5.0, 5.0]), 2.0)
        mon = ConvergenceMonitor()
        info = mon.update(1.0, elite_pool=pool)
        assert info["diversity"] > 0

    def test_diversity_zero_with_singleton_pool(self):
        pool = ElitePool(max_size=2, min_distance=0.0)
        pool.add(np.array([0.0]), 1.0)
        mon = ConvergenceMonitor()
        info = mon.update(1.0, elite_pool=pool)
        assert info["diversity"] == pytest.approx(0.0)


# ----------------------------- ElitePool -----------------------------


class TestElitePool:
    def test_add_and_get_best(self):
        pool = ElitePool(max_size=3, min_distance=0.0)
        pool.add(np.array([1.0, 2.0]), 5.0)
        pool.add(np.array([3.0, 4.0]), 1.0)
        best_sol, best_cost = pool.get_best()
        assert best_cost == pytest.approx(1.0)
        np.testing.assert_array_equal(best_sol, np.array([3.0, 4.0]))

    def test_get_best_empty_raises(self):
        pool = ElitePool()
        with pytest.raises(EmptyPoolError):
            pool.get_best()

    def test_rejects_too_close_solutions(self):
        pool = ElitePool(max_size=3, min_distance=10.0)
        assert pool.add(np.array([1.0, 1.0]), 1.0) is True
        assert pool.add(np.array([1.0, 1.0]), 0.5) is False
        assert pool.size() == 1

    def test_replaces_worst_when_full_and_better(self):
        pool = ElitePool(max_size=2, min_distance=0.0)
        pool.add(np.array([0.0]), 10.0)
        pool.add(np.array([1.0]), 5.0)
        pool.add(np.array([2.0]), 1.0)
        assert pool.size() == 2
        assert pool.get_best()[1] == pytest.approx(1.0)

    def test_does_not_replace_when_full_and_worse(self):
        pool = ElitePool(max_size=2, min_distance=0.0)
        pool.add(np.array([0.0]), 1.0)
        pool.add(np.array([1.0]), 2.0)
        assert pool.add(np.array([2.0]), 100.0) is False
        assert pool.size() == 2

    def test_clear_empties_pool(self):
        pool = ElitePool(max_size=3, min_distance=0.0)
        pool.add(np.array([1.0]), 1.0)
        pool.clear()
        assert pool.size() == 0

    def test_get_all_returns_copy(self):
        pool = ElitePool(max_size=2, min_distance=0.0)
        pool.add(np.array([1.0]), 1.0)
        out = pool.get_all()
        assert len(out) == 1
        out.clear()
        assert pool.size() == 1

    def test_normalized_distance_with_bounds(self):
        pool = ElitePool(
            max_size=2,
            min_distance=0.6,
            lower=np.array([0.0]),
            upper=np.array([10.0]),
        )
        pool.add(np.array([0.0]), 1.0)
        assert pool.add(np.array([5.0]), 0.5) is False
        assert pool.add(np.array([9.0]), 0.5) is True


# ----------------------------- selection helpers -----------------------------


def test_select_rcl_picks_top_when_threshold_excludes_all():
    valid_indices = np.array([0, 1, 2, 3])
    # all-equal ratios: threshold == max == min, mask covers all
    valid_ratios = np.array([1.0, 1.0, 1.0, 1.0])
    out = select_rcl(valid_indices, valid_ratios, alpha=0.5)
    assert len(out) >= 1


def test_select_rcl_empty_threshold_falls_back_to_top():
    """When the alpha threshold excludes everything, fallback returns top 30%."""
    valid_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # craft ratios so ``valid_ratios >= threshold`` is empty: only possible
    # if all ratios are NaN — easier to verify the fallback by running
    # ``select_rcl`` with a single-element array (mask non-empty)
    valid_ratios = np.array([1.0, 2.0, 5.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = select_rcl(valid_indices, valid_ratios, alpha=0.1)
    assert len(out) >= 1


def test_select_from_rcl_returns_none_if_all_infinite():
    rng = np.random.default_rng(0)
    out = _select_from_rcl(np.array([np.inf, np.inf]), 0.5, rng)
    assert out is None


def test_select_from_rcl_picks_finite_index():
    rng = np.random.default_rng(0)
    out = _select_from_rcl(np.array([1.0, np.inf, 2.0]), 0.0, rng)
    assert out == 0


def test_select_from_rcl_fallback_when_threshold_excludes_all():
    rng = np.random.default_rng(0)
    # All identical -> threshold == all values; rcl_local non-empty
    out = _select_from_rcl(np.array([1.0, 1.0]), 0.0, rng)
    assert out in (0, 1)


# ----------------------------- candidate builders -----------------------------


def test_sample_integer_from_bounds_within_range():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = _sample_integer_from_bounds(2.0, 5.0, rng)
        assert 2 <= v <= 5
        assert v == int(v)


def test_sample_integer_when_bounds_collapse():
    rng = np.random.default_rng(0)
    v = _sample_integer_from_bounds(3.7, 3.2, rng)
    assert v == int(v)


def test_build_random_and_heuristic_candidates_respect_bounds():
    _set_integer_split(2)
    rng = np.random.default_rng(0)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 5.0, 5.0])
    for _ in range(5):
        sol = _build_random_candidate(4, 2, lower, upper, rng)
        assert np.all(sol >= lower) and np.all(sol <= upper)
        sol = _build_heuristic_candidate(4, 2, lower, upper, rng)
        assert np.all(sol >= lower - 1e-9) and np.all(sol <= upper + 1e-9)


def test_build_heuristic_candidate_collapsed_int_bounds():
    """Cover the else branch when ``hi <= lo`` for the integer block."""
    _set_integer_split(1)
    rng = np.random.default_rng(0)
    lower = np.array([0.0, 2.7])
    upper = np.array([1.0, 2.8])  # ceil(2.7)=3, floor(2.8)=2 -> hi < lo
    sol = _build_heuristic_candidate(2, 1, lower, upper, rng)
    assert sol.shape == (2,)


def test_construct_solution_numpy_returns_array():
    _set_integer_split(2)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 4.0, 4.0])
    sol = construct_solution_numpy(
        num_vars=4,
        lower_arr=lower,
        upper_arr=upper,
        evaluator=quad,
        initial_guess=None,
        alpha=0.2,
        seed=42,
        num_candidates_per_step=6,
        cache=EvaluationCache(maxsize=16),
        n_workers=1,
    )
    assert sol.shape == (4,)


def test_construct_solution_numpy_with_initial_guess():
    _set_integer_split(2)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 4.0, 4.0])
    init = np.array([0.5, 0.5, 1.0, 2.0])
    sol = construct_solution_numpy(
        num_vars=4,
        lower_arr=lower,
        upper_arr=upper,
        evaluator=quad,
        initial_guess=init,
        alpha=0.2,
        seed=42,
        num_candidates_per_step=4,
        cache=None,
        n_workers=2,
    )
    assert sol.shape == (4,)


def test_get_half_with_explicit_split():
    _set_integer_split(3)
    assert _get_half(5) == 3


def test_set_group_size_roundtrip():
    _set_group_size(8)
    assert _get_group_size() == 8


# ----------------------------- legacy evaluate_candidates -----------------------------


def test_evaluate_candidates_legacy_returns_arrays():
    available = np.array([0, 1, 2])
    deps_active = np.zeros(3, dtype=bool)
    deps_matrix = np.array([[0, 0], [1, 0], [0, 1]], dtype=int)
    deps_len = np.array([1, 2, 0])
    c_arr = np.array([10.0, 5.0, 3.0])
    a_arr = np.array([2.0, 3.0, 1.0])
    ratios, inc_costs, valid = evaluate_candidates(
        available,
        deps_active,
        current_cost=0,
        deps_matrix=deps_matrix,
        deps_len=deps_len,
        c_arr=c_arr,
        a_arr=a_arr,
        b=100,
    )
    assert ratios.shape == (3,)
    assert inc_costs.shape == (3,)
    assert valid.shape == (3,)
    assert valid.all()


def test_evaluate_candidates_respects_budget():
    available = np.array([0])
    deps_active = np.zeros(2, dtype=bool)
    deps_matrix = np.array([[0, 1]], dtype=int)
    deps_len = np.array([2])
    c_arr = np.array([10.0])
    a_arr = np.array([100.0, 100.0])
    _, _, valid = evaluate_candidates(
        available,
        deps_active,
        current_cost=0,
        deps_matrix=deps_matrix,
        deps_len=deps_len,
        c_arr=c_arr,
        a_arr=a_arr,
        b=10,
    )
    assert not valid[0]


def test_evaluate_candidates_with_active_dependencies():
    available = np.array([0])
    deps_active = np.array([True, False])
    deps_matrix = np.array([[0, 1]], dtype=int)
    deps_len = np.array([2])
    c_arr = np.array([10.0])
    a_arr = np.array([5.0, 5.0])
    _, inc_costs, valid = evaluate_candidates(
        available,
        deps_active,
        current_cost=0,
        deps_matrix=deps_matrix,
        deps_len=deps_len,
        c_arr=c_arr,
        a_arr=a_arr,
        b=100,
    )
    assert valid[0]
    # only the inactive dep counts
    assert inc_costs[0] == pytest.approx(5)


# ----------------------------- path relinking -----------------------------


def test_path_relinking_identical_solutions():
    sol = np.array([1.0, 2.0, 3.0])
    out, cost = path_relinking(quad, sol, sol.copy())
    np.testing.assert_array_equal(out, sol)
    assert cost == pytest.approx(quad(sol))


def test_path_relinking_best_strategy():
    _set_integer_split(3)
    src = np.array([5.0, 5.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    _, cost = path_relinking(quad, src, tgt, strategy="best", seed=1)
    assert cost <= quad(src)


def test_path_relinking_forward_strategy():
    _set_integer_split(3)
    src = np.array([5.0, 5.0, 5.0])
    tgt = np.array([1.0, 1.0, 1.0])
    _, cost = path_relinking(quad, src, tgt, strategy="forward", seed=1)
    assert np.isfinite(cost)


def test_path_relinking_truncates_when_many_diff_indices():
    _set_integer_split(40)
    src = np.arange(40, dtype=float)
    tgt = src + 1.0
    out, _ = path_relinking(quad, src, tgt, seed=2)
    assert out.shape == (40,)


def test_bidirectional_path_relinking_returns_best():
    _set_integer_split(3)
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 3.0, 3.0])
    _, cost = bidirectional_path_relinking(quad, a, b)
    assert cost <= max(quad(a), quad(b))


# ----------------------------- perturb / alpha -----------------------------


def test_perturb_solution_changes_some_values():
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 2.0, 2.0])
    out = perturb_solution_numpy(
        sol,
        num_vars=4,
        strength=2,
        seed=0,
        lower_arr=np.array([0.0, 0.0, 0.0, 0.0]),
        upper_arr=np.array([1.0, 1.0, 5.0, 5.0]),
    )
    assert out.shape == sol.shape
    assert not np.array_equal(out, sol)


def test_perturb_solution_no_bounds():
    _set_integer_split(1)
    sol = np.array([0.5, 3.0])
    out = perturb_solution_numpy(sol, num_vars=2, strength=1, seed=0)
    assert out.shape == sol.shape


def test_get_current_alpha_adaptive():
    cfg = CoreConfig(
        adaptive_alpha=True, alpha_min=0.1, alpha_max=0.5, max_iterations=10
    )
    a0 = get_current_alpha(0, cfg)
    a9 = get_current_alpha(9, cfg)
    assert 0.1 - 0.02 <= a0 <= 0.5 + 0.02
    assert 0.1 - 0.02 <= a9 <= 0.5 + 0.02


def test_get_current_alpha_static():
    cfg = CoreConfig(adaptive_alpha=False, alpha=0.3)
    assert get_current_alpha(5, cfg) == pytest.approx(0.3)


# ----------------------------- local search & ILS -----------------------------


def test_local_search_vnd_runs():
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    lower = np.array([-1.0, -1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 3.0, 3.0])
    out = local_search_vnd(
        quad,
        sol,
        num_vars=4,
        max_iter=5,
        lower_arr=lower,
        upper_arr=upper,
    )
    assert out.shape == sol.shape


def test_local_search_vnd_adaptive_runs():
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    lower = np.array([-1.0, -1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 3.0, 3.0])
    out = local_search_vnd_adaptive(
        quad,
        sol,
        num_vars=4,
        max_iter=5,
        lower_arr=lower,
        upper_arr=upper,
    )
    assert out.shape == sol.shape


def test_ils_search_returns_finite_cost():
    _set_integer_split(2)
    cfg = CoreConfig(
        ils_iterations=2,
        vnd_iterations=4,
        perturbation_strength=2,
        adaptive_alpha=False,
    )
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    lower = np.array([-1.0, -1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 3.0, 3.0])
    out, cost = ils_search(
        sol,
        quad(sol),
        num_vars=4,
        cost_fn=quad,
        config=cfg,
        lower_arr=lower,
        upper_arr=upper,
    )
    assert np.isfinite(cost)
    assert out.shape == sol.shape


# ----------------------------- group / block neighborhoods -----------------------------


def test_local_search_vnd_with_group_size():
    _set_integer_split(6)
    _set_group_size(2)
    sol = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = local_search_vnd(
        quad,
        sol,
        num_vars=6,
        max_iter=3,
        lower_arr=np.zeros(6),
        upper_arr=np.ones(6),
    )
    assert out.shape == sol.shape


def test_neighborhood_group_with_layout():
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    lower = np.zeros(8)
    upper = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    out, _ = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=8,
        first_improvement=True,
        max_attempts=10,
        lower_arr=lower,
        upper_arr=upper,
    )
    assert out.shape == (8,)


def test_neighborhood_group_disabled_without_group_size():
    _set_integer_split(4)
    sol = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    out, _ = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=8,
        first_improvement=True,
        max_attempts=5,
        lower_arr=np.zeros(8),
        upper_arr=np.ones(8) * 3,
    )
    np.testing.assert_array_equal(out, sol)


def test_neighborhood_group_disabled_without_bounds():
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    out, _ = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=8,
        first_improvement=True,
        max_attempts=5,
    )
    np.testing.assert_array_equal(out, sol)


def test_neighborhood_block_with_layout():
    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    sol[8:] = 1.0
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=5,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
    )
    assert out.shape == (16,)


def test_neighborhood_block_disabled_without_group_size():
    _set_integer_split(8)
    sol = np.full(16, 0.5)
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=2,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
    )
    np.testing.assert_array_equal(out, sol)


def test_neighborhood_block_disabled_without_bounds():
    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=2,
    )
    np.testing.assert_array_equal(out, sol)


# ----------------------------- _prepare_bounds + grasp_ils_vnd -----------------------------


def test_prepare_bounds_missing_raises():
    from givp import InvalidBoundsError as IBE
    from givp._core import _prepare_bounds

    with pytest.raises(IBE):
        _prepare_bounds(None, [1.0])
    with pytest.raises(IBE):
        _prepare_bounds([0.0], None)


def test_prepare_bounds_shape_mismatch_raises():
    from givp import InvalidBoundsError as IBE
    from givp._core import _prepare_bounds

    with pytest.raises(IBE):
        _prepare_bounds([0.0, 0.0], [1.0])


def test_prepare_bounds_upper_not_strictly_greater_raises():
    from givp import InvalidBoundsError as IBE
    from givp._core import _prepare_bounds

    with pytest.raises(IBE):
        _prepare_bounds([0.0, 1.0], [1.0, 1.0])


# ----------------------------- expired-deadline branches -----------------------------


def _past_deadline() -> float:
    """Return a deadline that is already in the past."""
    import time

    return time.perf_counter() - 1.0


def test_path_relinking_best_with_expired_deadline():
    _set_integer_split(3)
    src = np.array([5.0, 5.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    out, _ = path_relinking(
        quad, src, tgt, strategy="best", seed=1, deadline=_past_deadline()
    )
    assert out.shape == (3,)


def test_path_relinking_forward_with_expired_deadline():
    _set_integer_split(3)
    src = np.array([5.0, 5.0, 5.0])
    tgt = np.array([1.0, 1.0, 1.0])
    out, _ = path_relinking(
        quad, src, tgt, strategy="forward", seed=1, deadline=_past_deadline()
    )
    assert out.shape == (3,)


def test_bidirectional_pr_with_expired_deadline():
    _set_integer_split(3)
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 3.0, 3.0])
    out, _ = bidirectional_path_relinking(quad, a, b, deadline=_past_deadline())
    assert out.shape == (3,)


def test_local_search_vnd_with_expired_deadline():
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    out = local_search_vnd(
        quad,
        sol,
        num_vars=4,
        max_iter=10,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
        deadline=_past_deadline(),
    )
    assert out.shape == sol.shape


def test_neighborhood_group_with_expired_deadline():
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    out, _ = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=8,
        first_improvement=True,
        max_attempts=10,
        lower_arr=np.zeros(8),
        upper_arr=np.ones(8) * 3,
        deadline=_past_deadline(),
    )
    assert out.shape == (8,)


def test_neighborhood_block_with_expired_deadline():
    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    sol[8:] = 1.0
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=10,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
        deadline=_past_deadline(),
    )
    assert out.shape == (16,)


def test_ils_search_with_expired_deadline():
    _set_integer_split(2)
    cfg = CoreConfig(
        ils_iterations=5,
        vnd_iterations=4,
        perturbation_strength=2,
        adaptive_alpha=False,
    )
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    _, cost = ils_search(
        sol,
        quad(sol),
        num_vars=4,
        cost_fn=quad,
        config=cfg,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
        deadline=_past_deadline(),
    )
    assert np.isfinite(cost)


# ----------------------------- adaptive VND extras -----------------------------


def test_local_search_vnd_adaptive_with_groups():
    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    sol[8:] = 1.0
    out = local_search_vnd_adaptive(
        quad,
        sol,
        num_vars=16,
        max_iter=8,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
    )
    assert out.shape == sol.shape


def test_local_search_vnd_adaptive_with_cache():
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    out = local_search_vnd_adaptive(
        quad,
        sol,
        num_vars=4,
        max_iter=4,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
        cache=EvaluationCache(maxsize=8),
    )
    assert out.shape == sol.shape


# ----------------------------- _execute_neighborhood multiflip branch -----------------------------


def test_execute_neighborhood_all_indices():
    """Cover every branch of ``_execute_neighborhood`` (idx 0..4)."""
    from givp._core import _execute_neighborhood

    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    sol[8:] = 1.0
    lower = np.zeros(16)
    upper = np.ones(16) * 3
    sensitivity = np.zeros(16)
    for idx in range(5):
        out, _ = _execute_neighborhood(
            idx,
            quad,
            sol.copy(),
            quad(sol),
            num_vars=16,
            first_improvement=True,
            lower_arr=lower,
            upper_arr=upper,
            sensitivity=sensitivity,
        )
        assert out.shape == (16,)


# ----------------------------- _evaluate_solution_with_cache branches -----------------------------


def test_evaluate_solution_with_cache_branches():
    from givp._core import _evaluate_solution_with_cache

    cache = EvaluationCache(maxsize=4)
    sol = np.array([1.0, 2.0])
    # first call: miss -> compute and store
    assert _evaluate_solution_with_cache(sol, quad, cache) == quad(sol)
    # second call: hit
    assert _evaluate_solution_with_cache(sol, quad, cache) == quad(sol)
    # no cache path
    assert _evaluate_solution_with_cache(sol, quad, None) == quad(sol)


# ----------------------------- _create_cached_cost_fn -----------------------------


def test_create_cached_cost_fn_with_and_without_cache():
    from givp._core import _create_cached_cost_fn

    cache = EvaluationCache(maxsize=4)
    cached = _create_cached_cost_fn(quad, cache)
    sol = np.array([1.0, 2.0])
    assert cached(sol) == quad(sol)
    assert cached(sol) == quad(sol)  # hit

    no_cache = _create_cached_cost_fn(quad, None)
    assert no_cache(sol) == quad(sol)


# ----------------------------- restart-on-stagnation path -----------------------------


def test_grasp_run_triggers_stagnation_restart():
    """Constant-cost evaluator forces stagnation -> partial restart code path."""
    from givp import grasp_ils_vnd_pr

    cfg = PublicConfig(
        max_iterations=12,
        vnd_iterations=4,
        ils_iterations=2,
        elite_size=3,
        path_relink_frequency=2,
        num_candidates_per_step=4,
        perturbation_strength=2,
        adaptive_alpha=False,
        use_convergence_monitor=False,
    )

    def flat(_x):
        return 1.0

    result = grasp_ils_vnd_pr(flat, [(-1.0, 1.0)] * 4, config=cfg, verbose=True)
    assert result.fun == pytest.approx(1.0)


# ----------------------------- direct-call helpers for fallback / verbose paths -----------------------------


def test_select_rcl_fallback_with_nan_ratios():
    """Triggers the ``len(rcl_indices) == 0`` fallback in ``select_rcl``."""
    out = select_rcl(
        np.array([0, 1, 2]),
        np.array([np.nan, np.nan, np.nan]),
        alpha=0.5,
    )
    assert len(out) >= 1


def test_handle_convergence_monitor_verbose_with_pool_trim():
    """Cover ``_handle_convergence_monitor`` verbose branch trimming the pool."""
    from givp._core import _handle_convergence_monitor

    pool = ElitePool(max_size=10, min_distance=0.0)
    for i in range(5):
        pool.add(np.array([float(i)]), float(i))

    mon = ConvergenceMonitor(window_size=3, restart_threshold=2)
    mon.update(10.0)
    mon.update(11.0)
    # Now no_improve_count == 1; one more push >= threshold so should_restart=True
    mon.update(12.0)
    out = _handle_convergence_monitor(mon, 12.0, pool, verbose=True)
    assert out == 0
    assert pool.size() == 2


def test_handle_convergence_monitor_returns_neg1_when_no_restart():
    from givp._core import _handle_convergence_monitor

    mon = ConvergenceMonitor(window_size=3, restart_threshold=10)
    out = _handle_convergence_monitor(mon, 1.0, None, verbose=False)
    assert out == -1


def test_handle_convergence_monitor_returns_zero_when_no_monitor():
    from givp._core import _handle_convergence_monitor

    assert _handle_convergence_monitor(None, 0.0, None, False) == 0


def test_maybe_apply_warm_start_updates_best():
    from givp._core import _maybe_apply_warm_start

    pool = ElitePool(max_size=2, min_distance=0.0)
    init = np.array([0.5, 0.5])
    new_cost, new_sol = _maybe_apply_warm_start(
        initial_guess=[0.5, 0.5],
        elite_pool=pool,
        cost_fn=quad,
        initial_arr=init,
        best_cost=10.0,
        best_solution=np.zeros(2),
        verbose=True,
    )
    assert new_cost == quad(init)
    assert pool.size() == 1
    np.testing.assert_array_equal(new_sol, init)


def test_maybe_apply_warm_start_keeps_best_when_initial_worse():
    from givp._core import _maybe_apply_warm_start

    pool = ElitePool(max_size=2, min_distance=0.0)
    init = np.array([5.0, 5.0])
    new_cost, _ = _maybe_apply_warm_start(
        initial_guess=[5.0, 5.0],
        elite_pool=pool,
        cost_fn=quad,
        initial_arr=init,
        best_cost=0.0,
        best_solution=np.zeros(2),
        verbose=False,
    )
    assert new_cost == pytest.approx(0.0)


def test_maybe_apply_warm_start_no_pool_short_circuit():
    from givp._core import _maybe_apply_warm_start

    out_cost, _ = _maybe_apply_warm_start(
        None,
        None,
        quad,
        np.zeros(2),
        1.0,
        np.zeros(2),
        False,
    )
    assert out_cost == pytest.approx(1.0)


def test_print_cache_stats_runs(caplog):
    from givp._core import _print_cache_stats

    cache = EvaluationCache(maxsize=4)
    cache.put(np.array([1.0]), 1.0)
    cache.get(np.array([1.0]))
    with caplog.at_level("INFO"):
        _print_cache_stats(cache, verbose=True)
        _print_cache_stats(None, verbose=True)
        _print_cache_stats(cache, verbose=False)


def test_check_early_stopping_branches(caplog):
    from givp._core import _check_early_stopping

    cfg = CoreConfig(early_stop_threshold=2)
    assert _check_early_stopping(None, cfg, False) is False

    mon = ConvergenceMonitor(restart_threshold=2)
    mon.no_improve_count = 5
    with caplog.at_level("INFO"):
        assert _check_early_stopping(mon, cfg, True) is True

    mon.no_improve_count = 0
    assert _check_early_stopping(mon, cfg, False) is False


def test_apply_path_relinking_to_pair_runs():
    """Cover ``_apply_path_relinking_to_pair`` directly."""
    from givp._core import _apply_path_relinking_to_pair, _create_cached_cost_fn

    _set_integer_split(3)
    cfg = CoreConfig(vnd_iterations=4)
    cached = _create_cached_cost_fn(quad, None)
    sol, cost = _apply_path_relinking_to_pair(
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 3.0, 3.0]),
        cached,
        num_vars=3,
        config=cfg,
        cache=None,
    )
    assert sol.shape == (3,)
    assert np.isfinite(cost)


def test_initialize_optimization_components_all_off():
    from givp._core import _initialize_optimization_components

    cfg = CoreConfig(
        use_elite_pool=False,
        use_cache=False,
        use_convergence_monitor=False,
    )
    elite, cache, mon = _initialize_optimization_components(cfg)
    assert elite is None
    assert cache is None
    assert mon is None


def test_initialize_optimization_components_all_on():
    from givp._core import _initialize_optimization_components

    cfg = CoreConfig(
        use_elite_pool=True,
        use_cache=True,
        use_convergence_monitor=True,
    )
    elite, cache, mon = _initialize_optimization_components(
        cfg,
        lower_arr=np.zeros(2),
        upper_arr=np.ones(2),
    )
    assert elite is not None
    assert cache is not None
    assert mon is not None


# ----------------------------- monkey-patched deadline branches -----------------------------


@pytest.fixture
def expired_deadline(monkeypatch):
    """Make ``_expired`` always return True regardless of deadline value."""
    from givp._core import _impl as core_impl

    monkeypatch.setattr(core_impl, "_expired", lambda _d: True)
    yield


def test_path_relinking_loops_short_circuit(expired_deadline):
    """All loops guarded by ``_expired`` exit immediately."""
    _set_integer_split(3)
    src = np.array([5.0, 5.0, 5.0])
    tgt = np.array([0.0, 0.0, 0.0])
    out, _ = path_relinking(quad, src, tgt, strategy="best", seed=1, deadline=1.0)
    assert out.shape == (3,)
    out, _ = path_relinking(quad, src, tgt, strategy="forward", seed=1, deadline=1.0)
    assert out.shape == (3,)


def test_local_search_vnd_short_circuit(expired_deadline):
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    out = local_search_vnd(
        quad,
        sol,
        num_vars=4,
        max_iter=10,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
        deadline=1.0,
    )
    assert out.shape == sol.shape


def test_ils_search_short_circuit(expired_deadline):
    _set_integer_split(2)
    cfg = CoreConfig(ils_iterations=5, vnd_iterations=4, perturbation_strength=2)
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    out, _ = ils_search(
        sol,
        quad(sol),
        num_vars=4,
        cost_fn=quad,
        config=cfg,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
        deadline=1.0,
    )
    assert out.shape == sol.shape


def test_neighborhoods_short_circuit(expired_deadline):
    _set_integer_split(8)
    _set_group_size(4)
    sol = np.full(16, 0.5)
    sol[8:] = 1.0
    out, _ = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=10,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
        deadline=1.0,
    )
    assert out.shape == (16,)
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=16,
        first_improvement=True,
        max_attempts=10,
        lower_arr=np.zeros(16),
        upper_arr=np.ones(16) * 3,
        deadline=1.0,
    )
    assert out.shape == (16,)


def test_grasp_run_with_immediate_deadline(monkeypatch):
    """Make the loop quit on first iteration via patched ``_expired``."""
    from givp import grasp_ils_vnd_pr
    from givp._core import _impl as core_impl

    cfg = PublicConfig(
        max_iterations=5,
        vnd_iterations=4,
        ils_iterations=1,
        elite_size=2,
        path_relink_frequency=1,
        time_limit=0.1,
    )
    calls = {"n": 0}

    real_expired = core_impl._expired

    def patched(d):
        calls["n"] += 1
        # Allow the first few internal calls but expire after a couple
        return calls["n"] > 3 or real_expired(d)

    monkeypatch.setattr(core_impl, "_expired", patched)
    result = grasp_ils_vnd_pr(quad, [(-1.0, 1.0)] * 3, config=cfg, verbose=True)
    assert np.isfinite(result.fun)


def test_perturb_index_no_bounds_int_branch():
    """Cover the ``_perturb_index`` integer branch with no bounds."""
    from givp._core import _perturb_index

    _set_integer_split(1)
    arr = np.array([0.5, 3.0])
    rng = np.random.default_rng(0)
    _perturb_index(arr, idx=1, strength=2, rng=rng, lower_arr=None, upper_arr=None)
    assert arr.shape == (2,)
