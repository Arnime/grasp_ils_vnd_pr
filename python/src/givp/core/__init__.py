"""Compatibility package `givp.core` re-exporting the internal core submodules.

This module provides a compatibility layer: it ensures callers can import
either the submodules (``import givp.core.path_relinking as _pr_module``)
or the convenience callables/symbols (``from givp.core import path_relinking``).

Design goals:
- Import submodules as modules first so direct `import givp.core.<mod>`
  yields a real module object.
- Then import and re-export the canonical callables/symbols from those
  submodules so ``from givp.core import ...`` returns the expected API.
"""

from __future__ import annotations

# `import givp.core.path_relinking as _pr_module` and get the module object.
from givp.core import grasp, ils, impl, vnd

# Re-export key symbols from the canonical submodules (public API surface).
from givp.core.cache import EvaluationCache
from givp.core.convergence import ConvergenceMonitor
from givp.core.elite import ElitePool
from givp.core.grasp import (
    _build_heuristic_candidate,
    _build_random_candidate,
    _evaluate_with_cache,
    _sample_integer_from_bounds,
    _select_from_rcl,
    _validate_bounds_and_initial,
    construct_grasp,
    evaluate_candidates,
    get_current_alpha,
    select_rcl,
)
from givp.core.helpers import (
    _get_group_size,
    _get_half,
    _safe_evaluate,
    _set_group_size,
    _set_integer_split,
)
from givp.core.ils import ils_search, perturb_solution_numpy
from givp.core.impl import (
    _AlgorithmConfig as GIVPConfig,
)
from givp.core.impl import (
    _apply_path_relinking_to_pair,
    _check_early_stopping,
    _evaluate_solution_with_cache,
    _handle_convergence_monitor,
    _initialize_optimization_components,
    _maybe_apply_warm_start,
    _prepare_bounds,
    _print_cache_stats,
    grasp_ils_vnd,
)
from givp.core.pr import bidirectional_path_relinking, path_relinking
from givp.core.vnd import (
    _create_cached_cost_fn,
    _execute_neighborhood,
    _neighborhood_block,
    _neighborhood_group,
    local_search_vnd,
    local_search_vnd_adaptive,
)
from givp.core.vnd_moves import (
    _modify_indices_for_multiflip,
    _perturb_index,
)
from givp.core.vnd_neighborhoods import (
    _group_layout,
    _neighborhood_multiflip,
    _sign_from_delta,
)

__all__ = [
    "ConvergenceMonitor",
    "ElitePool",
    "EvaluationCache",
    "GIVPConfig",
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
    "_group_layout",
    "_handle_convergence_monitor",
    "_initialize_optimization_components",
    "_maybe_apply_warm_start",
    "_modify_indices_for_multiflip",
    "_neighborhood_block",
    "_neighborhood_group",
    "_neighborhood_multiflip",
    "_perturb_index",
    "_prepare_bounds",
    "_print_cache_stats",
    "_safe_evaluate",
    "_sample_integer_from_bounds",
    "_select_from_rcl",
    "_set_group_size",
    "_set_integer_split",
    "_sign_from_delta",
    "_validate_bounds_and_initial",
    "bidirectional_path_relinking",
    "construct_grasp",
    "evaluate_candidates",
    "get_current_alpha",
    "grasp",
    "grasp_ils_vnd",
    "ils",
    "ils_search",
    "impl",
    "local_search_vnd",
    "local_search_vnd_adaptive",
    "path_relinking",
    "perturb_solution_numpy",
    "select_rcl",
    "vnd",
]
