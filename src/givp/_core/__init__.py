# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""``givp._core`` package — algorithm implementation.

Historically this was a single ``_core.py`` module. It is now a package whose
public surface is split across dedicated submodules:

* :mod:`._cache` — :class:`EvaluationCache` (LRU evaluation cache).
* :mod:`._convergence` — :class:`ConvergenceMonitor` (restart heuristics).
* :mod:`._elite` — :class:`ElitePool` (diversity-aware elite set).
* :mod:`._impl` — every other symbol (neighbourhoods, path relinking,
  GRASP/ILS/VND loop). To be split incrementally.

Both internal callers (``givp._api``, ``givp._config``) and the test suite
import symbols directly from ``givp._core``; this module re-exports the
relevant names so those imports keep working unchanged.
"""

from givp._core._cache import EvaluationCache
from givp._core._convergence import ConvergenceMonitor
from givp._core._elite import ElitePool

# Re-export everything else from the legacy implementation module.
from givp._core._impl import (
    GraspIlsVndConfig,
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
    _validate_bounds_and_initial,
    bidirectional_path_relinking,
    construct_solution_numpy,
    evaluate_candidates,
    get_current_alpha,
    grasp_ils_vnd,
    ils_search,
    local_search_vnd,
    local_search_vnd_adaptive,
    path_relinking,
    perturb_solution_numpy,
    select_rcl,
)

__all__ = [
    "ConvergenceMonitor",
    "ElitePool",
    "EvaluationCache",
    "GraspIlsVndConfig",
    "_apply_path_relinking_to_pair",
    "_build_heuristic_candidate",
    "_build_random_candidate",
    "_check_early_stopping",
    "_create_cached_cost_fn",
    "_evaluate_solution_with_cache",
    "_evaluate_with_cache",
    "_execute_neighborhood",
    "_get_group_size",
    "_get_half",
    "_handle_convergence_monitor",
    "_initialize_optimization_components",
    "_maybe_apply_warm_start",
    "_neighborhood_block",
    "_neighborhood_group",
    "_perturb_index",
    "_prepare_bounds",
    "_print_cache_stats",
    "_safe_evaluate",
    "_sample_integer_from_bounds",
    "_select_from_rcl",
    "_set_group_size",
    "_set_integer_split",
    "_validate_bounds_and_initial",
    "bidirectional_path_relinking",
    "construct_solution_numpy",
    "evaluate_candidates",
    "get_current_alpha",
    "grasp_ils_vnd",
    "ils_search",
    "local_search_vnd",
    "local_search_vnd_adaptive",
    "path_relinking",
    "perturb_solution_numpy",
    "select_rcl",
]
