"""Tests for ``givp.core`` internals: helpers, classes, neighborhoods, search."""

from __future__ import annotations

import contextlib
import logging
import time

import givp.core.pr as pr_module
import numpy as np
import pytest
from givp import EmptyPoolError, GIVPConfig, givp
from givp import InvalidBoundsError as IBE
from givp.config import GIVPConfig as PublicConfig
from givp.core import (
    ConvergenceMonitor,
    ElitePool,
    EvaluationCache,
    _apply_path_relinking_to_pair,
    _build_heuristic_candidate,
    _build_random_candidate,
    _check_early_stopping,
    _create_cached_cost_fn,
    _evaluate_solution_with_cache,
    _evaluate_with_cache,
    _execute_neighborhood,
    _get_group_size,
    _get_half,
    _handle_convergence_monitor,
    _initialize_optimization_components,
    _maybe_apply_warm_start,
    _neighborhood_block,
    _neighborhood_group,
    _perturb_index,
    _prepare_bounds,
    _print_cache_stats,
    _safe_evaluate,
    _sample_integer_from_bounds,
    _select_from_rcl,
    _set_group_size,
    _set_integer_split,
    bidirectional_path_relinking,
    construct_grasp,
    evaluate_candidates,
    get_current_alpha,
    ils_search,
    local_search_vnd,
    local_search_vnd_adaptive,
    path_relinking,
    perturb_solution_numpy,
    select_rcl,
)
from givp.core import (
    GIVPConfig as CoreConfig,
)
from givp.core import impl as core_impl
from givp.core import vnd as core_vnd
from givp.core.helpers import (
    _VERBOSE_HANDLER_ATTACHED,
    _ensure_verbose_handler,
    logger,
)
from givp.core.impl import _print_run_footer, _print_run_header


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

    def test_hash_solution_sha1_fallback(self, monkeypatch):
        """Line 59: force the sha1 fallback path by patching _FAST_HASH to False."""
        import givp.core.cache as cache_module

        monkeypatch.setattr(cache_module, "_FAST_HASH", False)
        cache = EvaluationCache(maxsize=4)
        sol = np.array([1.0, 2.0])
        cache.put(sol, 7.0)
        assert cache.get(sol) == pytest.approx(7.0)


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


def test_construct_grasp_returns_array():
    _set_integer_split(2)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 4.0, 4.0])
    sol = construct_grasp(
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


def test_construct_grasp_with_initial_guess():
    _set_integer_split(2)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 4.0, 4.0])
    init = np.array([0.5, 0.5, 1.0, 2.0])
    sol = construct_grasp(
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


def test_bidirectional_path_relinking_returns_first_when_cost1_wins(monkeypatch):
    """Covers the `if cost1 <= cost2: return best1, cost1` branch."""
    _set_integer_split(3)
    expected = np.array([1.0, 0.0, 0.0])
    call_count = 0

    def fake_path_relinking(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return expected, 0.5  # cost1 — lower, wins
        return expected, 2.0  # cost2 — higher

    monkeypatch.setattr(pr_module, "path_relinking", fake_path_relinking)
    _, result_cost = bidirectional_path_relinking(quad, expected, expected)
    assert result_cost == pytest.approx(0.5)


def test_bidirectional_path_relinking_returns_second_when_cost2_wins(monkeypatch):
    """Covers the `return best2, cost2` branch (cost2 < cost1)."""
    _set_integer_split(3)
    expected = np.array([1.0, 0.0, 0.0])
    call_count = 0

    def fake_path_relinking(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return expected, 2.0  # cost1 — higher
        return expected, 0.5  # cost2 — lower, wins

    monkeypatch.setattr(pr_module, "path_relinking", fake_path_relinking)
    _, result_cost = bidirectional_path_relinking(quad, expected, expected)
    assert result_cost == pytest.approx(0.5)


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
    with pytest.raises(IBE):
        _prepare_bounds(None, [1.0])
    with pytest.raises(IBE):
        _prepare_bounds([0.0], None)


def test_prepare_bounds_shape_mismatch_raises():
    with pytest.raises(IBE):
        _prepare_bounds([0.0, 0.0], [1.0])


def test_prepare_bounds_upper_not_strictly_greater_raises():
    with pytest.raises(IBE):
        _prepare_bounds([0.0, 1.0], [1.0, 1.0])


# ----------------------------- expired-deadline branches -----------------------------


def _past_deadline() -> float:
    """Return a deadline that is already in the past."""
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


def test_search_continuous_flip_break_and_try_neighborhoods_second_expired(monkeypatch):
    """Covers line 145 (break in _search_continuous_flip_module) and
    line 693 (second _expired guard in _try_neighborhoods).

    Strategy: monkeypatch _expired to return False on the first call
    (the initial guard in _try_neighborhoods) then True for all subsequent
    calls.  With all variables treated as continuous (integer_split = num_vars
    → int_indices empty) the integer loop never calls _expired, so the very
    next call is count=0 inside _search_continuous_flip_module → break (L145).
    Control then returns to _try_neighborhoods with no improvement, where the
    second _expired check fires (L693).
    """
    calls = [0]

    def fake_expired(_deadline: float) -> bool:
        calls[0] += 1
        return calls[0] > 1  # False first, True thereafter

    monkeypatch.setattr(core_vnd, "_expired", fake_expired)

    num_vars = 4
    _set_integer_split(num_vars)  # half = num_vars → all continuous, no integers

    sol = np.array([0.5, 0.5, 0.5, 0.5])
    cached = _create_cached_cost_fn(quad, None)

    _solution, _benefit, improved = core_vnd._try_neighborhoods(
        cached,
        sol,
        quad(sol),
        num_vars,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=5,
        lower_arr=np.zeros(num_vars),
        upper_arr=np.ones(num_vars),
        sensitivity=np.zeros(num_vars),
        deadline=1.0,
    )
    assert not improved
    assert calls[0] >= 2  # both _expired calls were made


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

    result = givp(flat, [(-1.0, 1.0)] * 4, config=cfg, verbose=True)
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
    mon = ConvergenceMonitor(window_size=3, restart_threshold=10)
    out = _handle_convergence_monitor(mon, 1.0, None, verbose=False)
    assert out == -1


def test_handle_convergence_monitor_returns_zero_when_no_monitor():
    assert _handle_convergence_monitor(None, 0.0, None, False) == 0


def test_maybe_apply_warm_start_updates_best():

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
    cache = EvaluationCache(maxsize=4)
    cache.put(np.array([1.0]), 1.0)
    cache.get(np.array([1.0]))
    with caplog.at_level("INFO"):
        _print_cache_stats(cache, verbose=True)
        _print_cache_stats(None, verbose=True)
        _print_cache_stats(cache, verbose=False)


def test_check_early_stopping_branches(caplog):
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
    monkeypatch.setattr(core_impl, "_expired", lambda _d: True)


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
    result = givp(quad, [(-1.0, 1.0)] * 3, config=cfg, verbose=True)
    assert np.isfinite(result.fun)


def test_perturb_index_no_bounds_int_branch():
    """Cover the ``_perturb_index`` integer branch with no bounds."""
    _set_integer_split(1)
    arr = np.array([0.5, 3.0])
    rng = np.random.default_rng(0)
    _perturb_index(arr, idx=1, strength=2, rng=rng, lower_arr=None, upper_arr=None)
    assert arr.shape == (2,)


# ----------------------------- verbose handler & run header/footer -----------------------------


class _ListHandler(logging.Handler):
    """Minimal in-memory log handler used by verbose tests."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        self.messages.append(self.format(record))


def _capture_logger(logger_name: str = "givp.core"):
    """Context manager that attaches a ``_ListHandler`` to *logger_name* and yields it."""

    @contextlib.contextmanager
    def _ctx():
        log = logging.getLogger(logger_name)
        h = _ListHandler()
        h.setLevel(logging.DEBUG)
        log.addHandler(h)
        try:
            yield h
        finally:
            log.removeHandler(h)

    return _ctx()


def test_ensure_verbose_handler_attaches_once():
    """``_ensure_verbose_handler`` adds a handler the first time and is idempotent."""
    # Force a clean state so we can exercise the "first call" branch.
    _VERBOSE_HANDLER_ATTACHED[0] = False
    original_handlers = logger.handlers[:]
    logger.handlers.clear()

    _ensure_verbose_handler()
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1
    assert _VERBOSE_HANDLER_ATTACHED[0] is True

    # Second call must be idempotent — handler count must not grow.
    handler_count_after_first = len(logger.handlers)
    _ensure_verbose_handler()
    assert len(logger.handlers) == handler_count_after_first

    # Restore original handlers so subsequent tests see a clean logger.
    logger.handlers.clear()
    for h in original_handlers:
        logger.addHandler(h)


def test_print_run_header_verbose_true():
    """``_print_run_header`` emits a log line when verbose=True."""
    cfg = CoreConfig(
        max_iterations=10,
        adaptive_alpha=True,
        alpha_min=0.08,
        alpha_max=0.18,
        use_elite_pool=True,
        elite_size=5,
        time_limit=0.0,
    )
    with _capture_logger() as h:
        _print_run_header(verbose=True, num_vars=4, config=cfg)
    assert any("GRASP-ILS-VND-PR start" in m for m in h.messages)


def test_print_run_header_verbose_false():
    """``_print_run_header`` emits nothing when verbose=False."""
    cfg = CoreConfig()
    with _capture_logger() as h:
        _print_run_header(verbose=False, num_vars=4, config=cfg)
    assert not any("GRASP-ILS-VND-PR start" in m for m in h.messages)


def test_print_run_header_non_adaptive_alpha():
    """Header uses fixed alpha values when adaptive_alpha=False."""
    cfg = CoreConfig(adaptive_alpha=False, alpha=0.15)
    with _capture_logger() as h:
        _print_run_header(verbose=True, num_vars=2, config=cfg)
    assert any("0.150" in m for m in h.messages)


def test_print_run_header_with_time_limit():
    """Header shows formatted time limit when time_limit > 0."""
    cfg = CoreConfig(time_limit=30.0)
    with _capture_logger() as h:
        _print_run_header(verbose=True, num_vars=2, config=cfg)
    assert any("30.0s" in m for m in h.messages)


def test_print_run_footer_verbose_true():
    """``_print_run_footer`` emits a log line when verbose=True."""
    start = time.monotonic()
    with _capture_logger() as h:
        _print_run_footer(verbose=True, best_cost=3.14, stagnation=7, start_time=start)
    assert any("GRASP-ILS-VND-PR end" in m for m in h.messages)
    assert any("3.1400" in m for m in h.messages)


def test_print_run_footer_verbose_false():
    """``_print_run_footer`` emits nothing when verbose=False."""
    start = time.monotonic()
    with _capture_logger() as h:
        _print_run_footer(verbose=False, best_cost=1.0, stagnation=0, start_time=start)
    assert not any("GRASP-ILS-VND-PR end" in m for m in h.messages)


def test_verbose_output_contains_iter_and_best():
    """End-to-end: verbose run emits header, per-iteration lines, and footer."""
    cfg = GIVPConfig(
        max_iterations=4,
        vnd_iterations=6,
        ils_iterations=2,
        elite_size=3,
        num_candidates_per_step=4,
        use_convergence_monitor=False,
    )
    with _capture_logger() as h:
        result = givp(
            lambda x: float(np.sum(x**2)),
            [(-2.0, 2.0)] * 3,
            config=cfg,
            seed=1,
            verbose=True,
        )
    assert any("GRASP-ILS-VND-PR start" in m for m in h.messages)
    assert any("iter" in m and "best=" in m for m in h.messages)
    assert any("GRASP-ILS-VND-PR end" in m for m in h.messages)
    assert np.isfinite(result.fun)


# ----------------------------- _sign_from_delta ---------------------------


def test_sign_from_delta_all_branches():
    """Lines 1234, 1244: cover negative (-1) and zero (0) return branches."""
    assert core_vnd._sign_from_delta(2.0) == 1
    assert core_vnd._sign_from_delta(-0.5) == -1
    assert core_vnd._sign_from_delta(0.0) == 0


# ----------------------------- _neighborhood_pair invalid split -----------


def test_neighborhood_swap_skips_when_half_equals_num_vars():
    """Line 1132: half >= num_vars triggers early return in _neighborhood_swap."""
    _set_integer_split(4)  # _get_half(4) = 4 >= num_vars=4 -> early return
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    out, ben = core_vnd._neighborhood_swap(quad, sol.copy(), quad(sol), num_vars=4)
    np.testing.assert_array_equal(out, sol)
    assert ben == pytest.approx(quad(sol))


# ----------------------------- _group_layout indivisible -----------------


def test_group_layout_returns_none_when_indivisible():
    """Lines 1162->1165: half not divisible by n_steps -> return None."""
    _set_integer_split(5)
    _set_group_size(3)  # 5 // 3 = 1; 1 * 3 = 3 != 5 -> None
    assert core_vnd._group_layout(10) is None


# ----------------------------- _modify_indices_for_multiflip no-bounds ---


def test_modify_indices_continuous_no_bounds():
    """Lines 622-623: continuous perturbation path without bounds supplied."""
    _set_integer_split(2)  # half=2; indices [0,1] are in the continuous region
    sol = np.array([0.5, 0.5, 1.0, 1.0])
    indices = np.array([0, 1])
    rng = np.random.default_rng(99)
    core_vnd._modify_indices_for_multiflip(
        sol, indices, rng, lower_arr=None, upper_arr=None
    )
    assert sol.shape == (4,)


# ----------------------------- _perturb_index continuous no bounds -------


def test_perturb_index_continuous_no_bounds():
    """Line 662: continuous variable with no bounds uses normal perturbation."""
    _set_integer_split(1)  # half=1; idx=0 is in the continuous region
    arr = np.array([1.5, 2.0])
    rng = np.random.default_rng(7)
    _perturb_index(arr, idx=0, strength=1, rng=rng, lower_arr=None, upper_arr=None)
    assert arr.shape == (2,)


# ----------------------------- grasp_ils_vnd config=None -----------------


def test_grasp_ils_vnd_config_none_uses_default():
    """Line 2316: grasp_ils_vnd auto-creates GIVPConfig when config=None."""
    _set_integer_split(2)
    _, cost = core_impl.grasp_ils_vnd(
        quad,
        num_vars=4,
        config=None,
        lower=[0.0, 0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 3.0, 3.0],
    )
    assert np.isfinite(cost)


# ----------------------------- _path_relinking_best lines 1452-1491 ------


def test_path_relinking_best_no_improving_move_breaks():
    """Line 1491->1475: break when _find_best_move returns None.
    Source is already optimal; every step toward target worsens cost."""
    _set_integer_split(3)
    src = np.array([0.0, 0.0, 0.0])
    tgt = np.array([2.0, 2.0, 2.0])
    out, cost = path_relinking(quad, src, tgt, strategy="best", seed=0)
    assert out.shape == (3,)
    assert np.isfinite(cost)


def test_path_relinking_best_processes_improving_moves():
    """Lines 1452, 1454: _find_best_move iterates indices and restores state."""
    _set_integer_split(3)
    src = np.array([3.0, 3.0, 3.0])
    tgt = np.array([0.0, 0.0, 0.0])
    out, cost = path_relinking(quad, src, tgt, strategy="best", seed=0)
    assert out.shape == (3,)
    assert cost <= 27.0


# ----------------------------- _find_best_move equal-index skip ----------


def test_find_best_move_skips_index_already_equal_to_target():
    """Line 1452: `continue` when current[idx] == target[idx]."""
    current = np.array([1.0, 2.0, 3.0], dtype=float)
    # idx=1: current[1] == target[1] == 2.0 -> the continue branch fires
    target = np.array([0.0, 2.0, 0.0], dtype=float)
    source = np.array([1.0, 2.0, 3.0], dtype=float)
    indices = np.array([0, 1, 2])
    diff_indices = np.array([0, 1, 2])
    best_idx, _ = pr_module._find_best_move(
        quad, current, target, indices, source, quad(current), diff_indices
    )
    assert best_idx != 1  # idx=1 was skipped; winner is 0 or 2


# ----------------------------- _safe_iteration_callback verbose=False ----


def test_safe_iteration_callback_exception_verbose_false():
    """Line 1727->exit: exception in callback with verbose=False skips logger.info."""

    def boom(_it, _cost, _sol):
        raise RuntimeError("oops")

    # Must not raise; the False branch of `if verbose:` is taken
    core_impl._safe_iteration_callback(
        boom, iter_idx=0, benefit=1.0, sol=np.zeros(2), verbose=False
    )


# ----------------------------- _search_*_module first_improvement --------


def test_search_integer_flip_module_first_improvement():
    """Line 515: early return when first_improvement=True and move improves cost."""
    _set_integer_split(2)
    # sol=[0,0,5,5], cost=50; moving idx=2 from 5->4 gives cost=41 < 50
    sol = np.array([0.0, 0.0, 5.0, 5.0])
    initial_cost = quad(sol)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 10.0, 10.0])
    out, cost = core_vnd._search_integer_flip_module(
        sol,
        initial_cost,
        np.arange(2, 4),  # integer indices
        quad,
        lower,
        upper,
        first_improvement=True,
    )
    assert out.shape == (4,)
    assert cost < initial_cost


def test_search_continuous_flip_module_first_improvement(monkeypatch):
    """Line 550: early return when first_improvement=True and move improves cost."""
    # Patch _try_continuous_move_module to guarantee an improvement on first call
    monkeypatch.setattr(
        core_vnd,
        "_try_continuous_move_module",
        lambda *_args, **_kwargs: (True, 0.0),
    )
    _set_integer_split(2)
    sol = np.array([3.0, 3.0, 0.0, 0.0])
    lower = np.array([-5.0, -5.0, 0.0, 0.0])
    upper = np.array([5.0, 5.0, 3.0, 3.0])
    rng = np.random.default_rng(0)
    out, cost = core_vnd._search_continuous_flip_module(
        sol,
        quad(sol),
        np.arange(0, 2),  # continuous indices
        quad,
        rng,
        lower,
        upper,
        first_improvement=True,
    )
    assert out.shape == sol.shape
    assert cost == pytest.approx(0.0)


# ----------------------------- _compute_ratios_numpy dependency branch ---


def test_evaluate_candidates_with_new_deps():
    """Line 146->148: branch where a package has new (inactive) dependencies."""
    # Package 0 depends on package 1; package 1 has no dependencies
    available = np.array([0, 1])
    deps_active = np.array([False, False])  # no dep active yet
    current_cost = 0
    deps_matrix = np.zeros((2, 2), dtype=int)
    deps_matrix[0, 0] = 1  # pkg 0's first dep is pkg 1
    deps_len = np.array([1, 0])  # pkg 0 has 1 dep; pkg 1 has 0
    c_arr = np.array([20, 15])  # package benefits
    a_arr = np.array([10, 5])  # dependency costs
    b = 100  # budget

    _, inc_costs, valid = evaluate_candidates(
        available,
        deps_active,
        current_cost,
        deps_matrix,
        deps_len,
        c_arr,
        a_arr,
        b,
    )
    # pkg 0: n_deps=1, dep 1 not active -> incremental_cost = a_arr[1] = 5
    assert valid[0]
    assert inc_costs[0] == 5


# ---- evaluate_candidates: all deps already active (146->148 False branch) ----


def test_evaluate_candidates_all_deps_already_active():
    """Line 146->148: n_deps > 0 but all deps already active -> incremental_cost = 0."""
    available = np.array([0])
    deps_active = np.array([False, True])  # dep pkg 1 is already active
    current_cost = 0
    deps_matrix = np.zeros((2, 2), dtype=int)
    deps_matrix[0, 0] = 1  # pkg 0 depends on pkg 1
    deps_len = np.array([1, 0])
    c_arr = np.array([20, 15])
    a_arr = np.array([10, 5])
    b = 100
    _, inc_costs, valid = evaluate_candidates(
        available, deps_active, current_cost, deps_matrix, deps_len, c_arr, a_arr, b
    )
    assert valid[0]
    assert inc_costs[0] == 0  # dep already active -> no incremental cost


# ---- _select_from_rcl: all-inf costs returns None (line 246) ----


def test_select_from_rcl_all_costs_infinite():
    """Line 246: return None when all candidate costs are infinite."""
    costs = np.array([np.inf, np.inf, np.inf])
    rng = np.random.default_rng(0)
    result = _select_from_rcl(costs, alpha=0.5, rng=rng)
    assert result is None


# ---- _select_from_rcl: empty RCL after filter falls back to valid_idx (line 254) ----


def test_select_from_rcl_empty_rcl_fallback():
    """Line 254: rcl_local = valid_idx when threshold filters out all candidates (alpha<0)."""
    costs = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    # alpha < 0 -> threshold < min_cost -> all valid_costs fail <= threshold -> empty RCL
    result = _select_from_rcl(costs, alpha=-0.5, rng=rng)
    assert result is not None
    assert 0 <= result < 3


# ---- _search_integer_flip_module: deadline break (line 515) ----


def test_search_integer_flip_module_deadline_break(monkeypatch):
    """Line 515: break fires when _expired returns True on first loop iteration."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: True)
    _set_integer_split(2)
    sol = np.array([0.0, 0.0, 5.0, 5.0])
    initial_cost = quad(sol)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 10.0, 10.0])
    out, cost = core_vnd._search_integer_flip_module(
        sol,
        initial_cost,
        np.arange(2, 4),
        quad,
        lower,
        upper,
        first_improvement=True,
        deadline=1.0,
    )
    np.testing.assert_array_equal(out, sol)
    assert cost == pytest.approx(initial_cost)


# ---- _search_integer_flip_module: improvement + first_improvement=False (522->513) ----


def test_search_integer_flip_module_no_first_improvement():
    """Line 522->513: improvement found but first_improvement=False continues loop."""
    _set_integer_split(2)
    sol = np.array([0.0, 0.0, 5.0, 5.0])
    initial_cost = quad(sol)
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 10.0, 10.0])
    _, cost = core_vnd._search_integer_flip_module(
        sol,
        initial_cost,
        np.arange(2, 4),
        quad,
        lower,
        upper,
        first_improvement=False,
    )
    assert cost < initial_cost


# ---- _search_continuous_flip_module: improvement + first_improvement=False (550->552) ----


def test_search_continuous_flip_module_no_first_improvement(monkeypatch):
    """Line 550->552: improvement found but first_improvement=False continues loop."""
    call_count = [0]

    def always_improve(*_args, **_kwargs):
        call_count[0] += 1
        return (True, 0.0)

    monkeypatch.setattr(core_vnd, "_try_continuous_move_module", always_improve)
    _set_integer_split(2)
    sol = np.array([3.0, 3.0, 0.0, 0.0])
    lower = np.array([-5.0, -5.0, 0.0, 0.0])
    upper = np.array([5.0, 5.0, 3.0, 3.0])
    rng = np.random.default_rng(0)
    _, cost = core_vnd._search_continuous_flip_module(
        sol,
        quad(sol),
        np.arange(0, 2),
        quad,
        rng,
        lower,
        upper,
        first_improvement=False,
    )
    assert call_count[0] >= 2  # called on both indices (no early return)
    assert cost == pytest.approx(0.0)


# ---- _modify_indices_for_multiflip: integer no-bounds path (572->576) ----


def test_modify_indices_integer_no_bounds():
    """Line 572->576: integer indices with lower_arr=None skips the clip block."""
    _set_integer_split(2)
    sol = np.array([0.5, 0.5, 3.0, 5.0])
    rng = np.random.default_rng(0)
    old_vals = core_vnd._modify_indices_for_multiflip(
        sol, indices=np.array([2, 3]), rng=rng, lower_arr=None, upper_arr=None
    )
    assert old_vals.shape == (2,)


# ---- _try_neighborhoods: expired at very start (line 662) ----


def test_try_neighborhoods_expired_immediately(monkeypatch):
    """Line 662: _expired True before first neighborhood -> immediately returns False."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: True)
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    _, cost, improved = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=3,
        lower_arr=None,
        upper_arr=None,
        deadline=1.0,
    )
    assert not improved
    assert cost == pytest.approx(2.0)


# ---- _try_neighborhoods: expired after swap no-improvement (line 696) ----


def test_try_neighborhoods_expired_after_swap(monkeypatch):
    """Line 696: _expired fires after swap neighborhood finds no improvement."""
    call_count = [0]

    def count_expired(d):
        call_count[0] += 1
        return call_count[0] >= 3

    monkeypatch.setattr(core_vnd, "_expired", count_expired)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_flip", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_swap", lambda *a, **kw: (a[1].copy(), a[2])
    )
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    _, _, improved = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=3,
        lower_arr=None,
        upper_arr=None,
        deadline=1.0,
    )
    assert not improved


# ---- _try_neighborhoods: group neighborhood improves (line 710) ----


def test_try_neighborhoods_group_neighborhood_improves(monkeypatch):
    """Line 710: _neighborhood_group returns an improvement -> early True return."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: False)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_flip", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_swap", lambda *a, **kw: (a[1].copy(), a[2])
    )
    improved_sol = np.zeros(2)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_group", lambda *a, **kw: (improved_sol, 0.0)
    )
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    _, cost, improved_flag = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=3,
        lower_arr=None,
        upper_arr=None,
    )
    assert improved_flag
    assert cost == pytest.approx(0.0)


# ---- _try_neighborhoods: expired after group no-improvement (line 713) ----


def test_try_neighborhoods_expired_after_group(monkeypatch):
    """Line 713: _expired fires after group neighborhood finds no improvement."""
    call_count = [0]

    def count_expired(d):
        call_count[0] += 1
        return call_count[0] >= 4

    monkeypatch.setattr(core_vnd, "_expired", count_expired)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_flip", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_swap", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_group", lambda *a, **kw: (a[1].copy(), a[2])
    )
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    _, _, improved = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=3,
        lower_arr=None,
        upper_arr=None,
        deadline=1.0,
    )
    assert not improved


# ---- _try_neighborhoods: block neighborhood improves (line 726) ----


def test_try_neighborhoods_block_neighborhood_improves(monkeypatch):
    """Line 726: _neighborhood_block returns an improvement -> early True return."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: False)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_flip", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_swap", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_group", lambda *a, **kw: (a[1].copy(), a[2])
    )
    improved_sol = np.zeros(2)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_block", lambda *a, **kw: (improved_sol, 0.0)
    )
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    _, cost, improved_flag = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=3,
        lower_arr=None,
        upper_arr=None,
    )
    assert improved_flag
    assert cost == pytest.approx(0.0)


# ---- _try_neighborhoods: expired in multiflip check (line 730) ----


def test_try_neighborhoods_expired_in_multiflip_check(monkeypatch):
    """Line 730: _expired fires inside `if iteration % limit == 0:` block."""
    call_count = [0]

    def count_expired(d):
        call_count[0] += 1
        return call_count[0] >= 5

    monkeypatch.setattr(core_vnd, "_expired", count_expired)
    monkeypatch.setattr(
        core_vnd, "_neighborhood_flip", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_swap", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_group", lambda *a, **kw: (a[1].copy(), a[2])
    )
    monkeypatch.setattr(
        core_vnd, "_neighborhood_block", lambda *a, **kw: (a[1].copy(), a[2])
    )
    _set_integer_split(1)
    sol = np.array([1.0, 1.0])
    # iteration=0, no_improve_flip_limit=1 -> 0 % 1 == 0 -> multiflip check fires
    _, _, improved = core_vnd._try_neighborhoods(
        quad,
        sol,
        2.0,
        num_vars=2,
        use_first_improvement=True,
        iteration=0,
        no_improve_flip_limit=1,
        lower_arr=None,
        upper_arr=None,
        deadline=1.0,
    )
    assert not improved


# ---- _neighborhood_swap: deadline break in loop (line 1132) ----


def test_neighborhood_swap_deadline_break(monkeypatch):
    """Line 1132: break fires when _expired True inside the for-loop."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: True)
    _set_integer_split(2)  # half=2 < num_vars=4 -> we enter the loop
    sol = np.array([1.0, 1.0, 2.0, 2.0])
    out, _ = core_vnd._neighborhood_swap(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=4,
        first_improvement=True,
        lower_arr=None,
        upper_arr=None,
        deadline=1.0,
    )
    np.testing.assert_array_equal(out, sol)


# ---- _neighborhood_swap: else branch (no bounds, lines 1155-1156) ----


def test_neighborhood_swap_no_bounds_else_branch():
    """Lines 1155-1156: else branch (no bounds) perturbs cont+int without clipping."""
    _set_integer_split(2)
    sol = np.array([1.0, 1.0, 2.0, 2.0])
    out, _ = core_vnd._neighborhood_swap(
        quad,
        sol.copy(),
        1000.0,
        num_vars=4,
        first_improvement=True,
        max_attempts=10,
        lower_arr=None,
        upper_arr=None,
    )
    assert out.shape == (4,)


# ---- _neighborhood_swap: improvement + first_improvement=False (1162->1165) ----


def test_neighborhood_swap_improvement_no_first_improvement():
    """Line 1162->1165: improvement found but first_improvement=False -> restore+continue."""
    _set_integer_split(2)
    sol = np.array([1.0, 1.0, 2.0, 2.0])
    _, cost = core_vnd._neighborhood_swap(
        lambda _: -1.0,
        sol.copy(),
        0.0,
        num_vars=4,
        first_improvement=False,
        max_attempts=5,
        lower_arr=np.zeros(4),
        upper_arr=np.ones(4) * 3,
    )
    assert cost == pytest.approx(-1.0)


# ---- _neighborhood_group: zero attempts (1340->1372) ----


def test_neighborhood_group_zero_attempts():
    """Line 1340->1372: for loop runs 0 times with max_attempts=0."""
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.ones(8)
    out, cost = _neighborhood_group(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=8,
        max_attempts=0,
        lower_arr=np.zeros(8),
        upper_arr=np.ones(8) * 3,
    )
    np.testing.assert_array_equal(out, sol)
    assert cost == pytest.approx(quad(sol))


# ---- _neighborhood_group: no improvement restores solution (lines 1369-1370) ----


def test_neighborhood_group_no_improvement_restores():
    """Lines 1369-1370: restore after perturbation doesn't improve cost."""
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.zeros(8)  # cost=0 is optimal -> any perturbation worsens
    _, cost = _neighborhood_group(
        quad,
        sol.copy(),
        0.0,
        num_vars=8,
        max_attempts=5,
        lower_arr=np.zeros(8),
        upper_arr=np.ones(8) * 3,
    )
    assert cost == pytest.approx(0.0)


# ---- _neighborhood_group: improvement + first_improvement=False (1366->1369) ----


def test_neighborhood_group_improvement_no_first_improvement():
    """Line 1366->1369: improvement found but first_improvement=False -> restore+continue."""
    _set_integer_split(4)
    _set_group_size(2)
    sol = np.ones(8)
    _, cost = _neighborhood_group(
        lambda _: -1.0,
        sol.copy(),
        0.0,
        num_vars=8,
        first_improvement=False,
        max_attempts=3,
        lower_arr=np.zeros(8),
        upper_arr=np.ones(8) * 3,
    )
    assert cost == pytest.approx(-1.0)


# ---- _neighborhood_block: zero attempts (1403->1435) ----


def test_neighborhood_block_zero_attempts():
    """Line 1403->1435: for loop runs 0 times with max_attempts=0."""
    _set_integer_split(5)
    _set_group_size(5)
    sol = np.ones(10)
    out, _ = _neighborhood_block(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=10,
        max_attempts=0,
        lower_arr=np.zeros(10),
        upper_arr=np.ones(10) * 3,
    )
    np.testing.assert_array_equal(out, sol)


# ---- _neighborhood_block: improvement + first_improvement=False (1430->1433) ----


def test_neighborhood_block_improvement_no_first_improvement():
    """Line 1430->1433: improvement found but first_improvement=False -> restore+continue."""
    _set_integer_split(5)
    _set_group_size(5)
    sol = np.ones(10)
    _, cost = _neighborhood_block(
        lambda _: -1.0,
        sol.copy(),
        0.0,
        num_vars=10,
        first_improvement=False,
        max_attempts=3,
        lower_arr=np.zeros(10),
        upper_arr=np.ones(10) * 3,
    )
    assert cost == pytest.approx(-1.0)


# ---- _find_best_move: deadline break (line 1452) ----


def test_find_best_move_deadline_break(monkeypatch):
    """Line 1452: break fires when _expired True in the for-loop."""
    monkeypatch.setattr(pr_module, "_expired", lambda _: True)
    current = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 1.0, 0.0])
    source = current.copy()
    indices = np.array([0, 1, 2])
    best_idx, _ = pr_module._find_best_move(
        quad, current, target, indices, source, quad(current), indices, deadline=1.0
    )
    assert best_idx is None  # break before any move was evaluated


# ---- _neighborhood_multiflip: deadline break in loop (line 1204) ----


def test_neighborhood_multiflip_deadline_break(monkeypatch):
    """Line 1204: break fires when _expired True inside the for-loop."""
    monkeypatch.setattr(core_vnd, "_expired", lambda _: True)
    _set_integer_split(2)
    sol = np.array([1.0, 1.0, 2.0, 2.0])
    out, _ = core_vnd._neighborhood_multiflip(
        quad,
        sol.copy(),
        quad(sol),
        num_vars=4,
        max_attempts=5,
        deadline=1.0,
    )
    np.testing.assert_array_equal(out, sol)


# ---- _handle_convergence_monitor: restart+verbose+no-pool (1746->1751) ----


def test_handle_convergence_monitor_restart_verbose_no_pool():
    """Line 1746->1751: restart+verbose but elite_pool is None -> skip to return 0."""
    monitor = ConvergenceMonitor(restart_threshold=1)
    monitor.update(10.0)
    monitor.update(15.0)  # worse -> no_improve=1 -> should_restart=True
    result = _handle_convergence_monitor(monitor, 15.0, elite_pool=None, verbose=True)
    assert result == 0


# ---- _process_path_relinking_pairs: expired break in inner j-loop (line 2011) ----


def test_process_path_relinking_pairs_expired_break(monkeypatch):
    """Line 2011: break in inner j-loop when _expired fires."""
    monkeypatch.setattr(core_impl, "_expired", lambda _: True)
    _set_integer_split(2)
    sol1 = np.array([1.0, 1.0, 0.0, 0.0])
    sol2 = np.array([0.0, 0.0, 1.0, 1.0])
    sol3 = np.array([0.5, 0.5, 0.5, 0.5])
    elite_solutions = [(sol1, 2.0), (sol2, 2.0), (sol3, 1.0)]
    cfg = CoreConfig(use_elite_pool=True, vnd_iterations=1)
    pool = ElitePool(max_size=5)
    for s, c in elite_solutions:
        pool.add(s, c)
    best_cost, _, _ = core_impl._process_path_relinking_pairs(
        elite_solutions,
        quad,
        num_vars=4,
        config=cfg,
        best_cost=2.0,
        best_solution=sol1.copy(),
        stagnation=0,
        elite_pool=pool,
        cache=None,
        deadline=1.0,
    )
    assert np.isfinite(best_cost)


# ---- grasp_ils_vnd: use_elite_pool=False covers 1885->1888 and 1961->1963 ----


def test_grasp_ils_vnd_no_elite_pool_stagnation_branches():
    """Lines 1885->1888, 1961->1963: use_elite_pool=False skips elite-pool adds.
    Also covers line 1923: verbose logger.info inside stagnation restart."""
    _set_integer_split(2)
    cfg = CoreConfig(
        max_iterations=3,
        vnd_iterations=1,
        ils_iterations=1,
        use_elite_pool=False,
        use_cache=False,
        use_convergence_monitor=True,  # must be True so stagnation isn't auto-reset to 0
        adaptive_alpha=False,
    )
    # Constant-cost function guarantees stagnation after iteration 1
    _, cost = core_impl.grasp_ils_vnd(
        lambda x: 1.0,
        num_vars=4,
        config=cfg,
        lower=[0.0, 0.0, 0.0, 0.0],
        upper=[2.0, 2.0, 2.0, 2.0],
        verbose=True,
    )
    assert np.isfinite(cost)


# ---- _path_relinking_best: move found but NOT better (1491->1475) ----


def test_path_relinking_best_move_found_but_not_better(monkeypatch):
    """Line 1491->1475: best_move_idx is not None but best_move_benefit >= best_benefit
    -> if-body skipped, loop continues to while check (the False branch of line 1491).
    """
    # Monkeypatch _find_best_move: first call returns (0, 100.0) meaning a move was
    # found but its cost is NOT less than best_benefit. Second call returns None -> break.
    call_count = [0]

    def fake_find_best_move(
        cost_fn,
        current,
        target,
        indices,
        source,
        best_benefit,
        diff_indices,
        deadline=0.0,
    ):
        call_count[0] += 1
        if call_count[0] == 1:
            # Pretend idx=0 was "best" but with cost equal to best_benefit (not better)
            current[0] = target[0]
            return 0, best_benefit  # NOT less than best_benefit
        return None, best_benefit  # break on second call

    monkeypatch.setattr(pr_module, "_find_best_move", fake_find_best_move)
    _set_integer_split(3)
    src = np.array([1.0, 2.0, 3.0])
    tgt = np.array([0.0, 0.0, 0.0])
    out, _ = path_relinking(quad, src, tgt, strategy="best", deadline=0.0)
    assert out.shape == (3,)
    assert call_count[0] == 2  # both fake calls were made
