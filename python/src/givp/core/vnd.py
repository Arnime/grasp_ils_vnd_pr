# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""VND — Variable Neighborhood Descent (orchestration).

Public entry points:

- :func:`local_search_vnd`          — standard VND with fixed neighbourhood order
- :func:`local_search_vnd_adaptive` — VND with Adaptive Neighbourhood Selection (ANS)

Implementation is split into focused sub-modules:

- :mod:`givp.core.vnd_moves`         — atomic move helpers (try_*, modify, perturb)
- :mod:`givp.core.vnd_neighborhoods` — swap, multiflip, group, block neighbourhoods

``_search_integer_flip_module``, ``_search_continuous_flip_module``,
``_neighborhood_flip``, and ``_execute_neighborhood`` are defined here so that
``monkeypatch.setattr(core_vnd, "_try_*", mock)`` continues to work in tests.

All symbols are accessible via ``import givp.core.vnd as core_vnd``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.cache import EvaluationCache
from givp.core.helpers import (
    _expired,
    _get_half,
    _new_rng,
)

# ---------------------------------------------------------------------------
# Re-export atomic-move symbols for full backward compatibility
# ---------------------------------------------------------------------------
from givp.core.vnd_moves import (
    _modify_indices_for_multiflip,
    _try_continuous_move_module,
    _try_integer_moves_module,
)

# Re-export group/block neighbourhood symbols
from givp.core.vnd_neighborhoods import (
    _group_layout,
    _neighborhood_block,
    _neighborhood_group,
    _neighborhood_multiflip,
    _neighborhood_swap,
    _sign_from_delta,
)

# Symbols re-exported so ``import givp.core.vnd as core_vnd`` exposes them.
__all__ = [
    "_create_cached_cost_fn",
    "_execute_neighborhood",
    "_group_layout",
    "_modify_indices_for_multiflip",
    "_neighborhood_block",
    "_neighborhood_flip",
    "_neighborhood_group",
    "_neighborhood_multiflip",
    "_neighborhood_swap",
    "_search_continuous_flip_module",
    "_search_integer_flip_module",
    "_sign_from_delta",
    "local_search_vnd",
    "local_search_vnd_adaptive",
]

# ---------------------------------------------------------------------------
# Cached cost-function wrapper
# ---------------------------------------------------------------------------


def _create_cached_cost_fn(
    cost_fn: Callable, cache: EvaluationCache | None
) -> Callable:
    """Create a cost-function wrapper with evaluation caching."""

    def cached_cost_fn(sol):
        if cache is not None:
            cached = cache.get(sol)
            if cached is not None:
                return cached
        cost = cost_fn(sol)
        if cache is not None:
            cache.put(sol, cost)
        return cost

    return cached_cost_fn


# ---------------------------------------------------------------------------
# Flip-sweep helpers
# (defined here so monkeypatch.setattr(core_vnd, "_try_*", mock) works)
# ---------------------------------------------------------------------------


def _search_integer_flip_module(
    sol: np.ndarray,
    best_benefit: float,
    indices: np.ndarray,
    cost_fn: Callable,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    first_improvement: bool,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Sweep over *indices* (integer variables), applying 1-opt moves."""
    best_sol = sol.copy()
    best_ben = best_benefit
    for count, i in enumerate(indices):
        if count % 8 == 0 and _expired(deadline):
            break
        new_sol, new_ben, improved = _try_integer_moves_module(
            i, sol, best_ben, cost_fn, lower_arr, upper_arr
        )
        if improved:
            best_ben = new_ben
            best_sol = new_sol.copy()
            if first_improvement:
                return best_sol, best_ben
    return best_sol, best_ben


def _search_continuous_flip_module(
    sol: np.ndarray,
    best_benefit: float,
    indices: np.ndarray,
    cost_fn: Callable,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    first_improvement: bool,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Sweep over *indices* (continuous variables), applying small perturbations."""
    best_sol = sol.copy()
    best_ben = best_benefit
    for count, i in enumerate(indices):
        if count % 8 == 0 and _expired(deadline):
            break
        old_val = sol[i]
        changed, new_ben = _try_continuous_move_module(
            i, sol, best_ben, cost_fn, rng, lower_arr, upper_arr
        )
        if changed:
            best_ben = new_ben
            best_sol = sol.copy()
            if first_improvement:
                return best_sol, best_ben
        sol[i] = old_val
    return best_sol, best_ben


# ---------------------------------------------------------------------------
# 1-opt neighbourhood (defined here to call local _search_* references)
# ---------------------------------------------------------------------------


def _neighborhood_flip(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    sensitivity: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """1-opt neighbourhood: try modifying each variable individually.

    P9: When a ``sensitivity`` array is provided, variables with a higher
    historical impact on the cost are explored first, reducing wasted evaluations.
    """
    rng = _new_rng(seed)
    half = _get_half(num_vars)
    if sensitivity is not None and np.any(sensitivity > 0):
        noise = rng.uniform(0, 0.1, size=num_vars) * np.max(sensitivity)
        priority = sensitivity + noise
        indices = np.argsort(-priority)
    else:
        indices = rng.permutation(num_vars)

    int_indices = indices[indices >= half]
    cont_indices = indices[indices < half]

    best_solution = solution.copy()
    best_benefit = current_benefit

    int_best_sol, int_best_ben = _search_integer_flip_module(
        solution,
        best_benefit,
        int_indices,
        cost_fn,
        lower_arr,
        upper_arr,
        first_improvement,
        deadline=deadline,
    )
    if int_best_ben < best_benefit:
        best_solution, best_benefit = int_best_sol, int_best_ben

    cont_best_sol, cont_best_ben = _search_continuous_flip_module(
        solution,
        best_benefit,
        cont_indices,
        cost_fn,
        rng,
        lower_arr,
        upper_arr,
        first_improvement,
        deadline=deadline,
    )
    if cont_best_ben < best_benefit:
        best_solution, best_benefit = cont_best_sol, cont_best_ben

    return best_solution, best_benefit


# ---------------------------------------------------------------------------
# Neighbourhood dispatcher (defined here so all 5 neighbourhoods are resolved
# from vnd.py's namespace, avoiding any circular-import issues)
# ---------------------------------------------------------------------------


def _execute_neighborhood(
    idx: int,
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    sensitivity: np.ndarray | None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Dispatch to a specific neighbourhood by index (0=flip, 1=pair, 2=group, 3=block, 4=multiflip)."""
    if idx == 0:
        return _neighborhood_flip(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            seed=None,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            sensitivity=sensitivity,
            deadline=deadline,
        )
    if idx == 1:
        return _neighborhood_swap(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    if idx == 2:
        return _neighborhood_group(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    if idx == 3:
        return _neighborhood_block(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    return _neighborhood_multiflip(
        cost_fn,
        solution,
        current_benefit,
        num_vars,
        k=3,
        seed=None,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )


def _try_neighborhoods(
    cached_cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    use_first_improvement: bool,
    iteration: int,
    no_improve_flip_limit: int,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    sensitivity: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float, bool]:
    """Try neighbourhoods: 1-opt, pair, group, block, and multiflip.

    Returns (solution, benefit, improved).
    """
    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_flip(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        seed=None,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        sensitivity=sensitivity,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_swap(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_group(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_block(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if iteration % no_improve_flip_limit == 0:
        if _expired(deadline):
            return solution, current_benefit, False
        new_solution, new_benefit = _neighborhood_multiflip(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            k=no_improve_flip_limit,
            seed=None,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
        if new_benefit < current_benefit:
            return new_solution, new_benefit, True

    return solution, current_benefit, False


def local_search_vnd(
    cost_fn: Callable,
    solution: np.ndarray,
    num_vars: int,
    max_iter: int = 300,
    use_first_improvement: bool = True,
    no_improve_limit: int = 5,
    no_improve_flip_limit: int = 3,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: EvaluationCache | None = None,
    deadline: float = 0.0,
) -> np.ndarray:
    """Run Variable Neighborhood Descent (VND) local search.

    Alternates between 1-opt, pair, group, block, and multiflip neighbourhoods
    with an optional LRU evaluation cache.

    Args:
        cost_fn: Objective function (minimisation).
        solution: Starting solution.
        num_vars: Number of variables.
        max_iter: Maximum number of VND iterations.
        use_first_improvement: If True, accept the first improving move.
        no_improve_limit: Maximum consecutive iterations without improvement.
        no_improve_flip_limit: Multiflip trigger period.
        cache: Evaluation cache (optional).

    Returns:
        Improved solution vector.
    """
    solution = np.array(solution, dtype=float)
    cached_cost_fn = _create_cached_cost_fn(cost_fn, cache)
    current_benefit = cached_cost_fn(solution)

    # P9: rastrear sensibilidade das variáveis para priorização
    sensitivity = np.zeros(num_vars, dtype=float)

    iteration = 0
    no_improve_count = 0

    while iteration < max_iter and no_improve_count < no_improve_limit:
        if _expired(deadline):
            break
        iteration += 1
        old_benefit = current_benefit
        old_solution = solution.copy()
        solution, current_benefit, improved = _try_neighborhoods(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            use_first_improvement,
            iteration,
            no_improve_flip_limit,
            lower_arr,
            upper_arr,
            sensitivity=sensitivity,
            deadline=deadline,
        )

        if improved:
            no_improve_count = 0
            # P9: atualizar sensibilidade — variáveis que mudaram recebem crédito
            changed_mask = np.abs(solution - old_solution) > 1e-12
            improvement = old_benefit - current_benefit
            sensitivity[changed_mask] += improvement
            # Decaimento exponencial para não ficar preso em variáveis antigas
            sensitivity *= 0.9
        else:
            no_improve_count += 1

    return solution


def local_search_vnd_adaptive(
    cost_fn: Callable,
    solution: np.ndarray,
    num_vars: int,
    max_iter: int = 300,
    use_first_improvement: bool = True,
    no_improve_limit: int = 5,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: EvaluationCache | None = None,
    reward_factor: float = 1.0,
    decay_factor: float = 0.95,
    min_probability: float = 0.05,
) -> np.ndarray:
    """VND with Adaptive Neighbourhood Selection (ANS / roulette-wheel).

    Instead of a fixed neighbourhood order (flip->pair->group->block->multiflip),
    uses roulette-wheel selection proportional to each neighbourhood's historical
    success.  Neighbourhoods that yield improvements receive a reward; all scores
    decay exponentially to allow re-adaptation throughout the search.

    Args:
        cost_fn: Objective function (minimisation).
        solution: Starting solution.
        num_vars: Number of variables.
        max_iter: Maximum number of iterations.
        use_first_improvement: If True, accept the first improving move.
        no_improve_limit: Maximum consecutive iterations without improvement.
        lower_arr: Lower bounds for each variable.
        upper_arr: Upper bounds for each variable.
        cache: Evaluation cache (optional).
        reward_factor: Reward multiplier for a successful neighbourhood.
        decay_factor: Per-iteration score decay (0 < decay_factor < 1).
        min_probability: Minimum selection probability per neighbourhood.

    Returns:
        Improved solution vector.
    """
    solution = np.array(solution, dtype=float)
    cached_cost_fn = _create_cached_cost_fn(cost_fn, cache)
    current_benefit = cached_cost_fn(solution)

    # P9: rastrear sensibilidade das variáveis para priorização
    sensitivity = np.zeros(num_vars, dtype=float)

    # ANS: scores de sucesso por vizinhança
    n_neighborhoods = 5  # flip, pair, group, block, multiflip
    scores = np.ones(n_neighborhoods, dtype=float)

    rng = _new_rng()
    iteration = 0
    no_improve_count = 0

    while iteration < max_iter and no_improve_count < no_improve_limit:
        iteration += 1
        old_benefit = current_benefit
        old_solution = solution.copy()

        # ANS: calcular probabilidades via roulette-wheel
        probs = scores / scores.sum()
        probs = np.maximum(probs, min_probability)
        probs /= probs.sum()

        neighborhood_idx = int(rng.choice(n_neighborhoods, p=probs))

        new_solution, new_benefit = _execute_neighborhood(
            neighborhood_idx,
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            use_first_improvement,
            lower_arr,
            upper_arr,
            sensitivity,
        )

        if new_benefit < current_benefit:
            improvement = current_benefit - new_benefit
            solution = new_solution
            current_benefit = new_benefit

            # ANS: recompensar vizinhança bem-sucedida (normalizado pelo custo)
            scores[neighborhood_idx] += (
                reward_factor * improvement / max(abs(old_benefit), 1e-10)
            )

            # P9: atualizar sensibilidade
            changed_mask = np.abs(solution - old_solution) > 1e-12
            sensitivity[changed_mask] += improvement
            sensitivity *= 0.9

            no_improve_count = 0
        else:
            no_improve_count += 1

        # ANS: decaimento dos scores para permitir readaptação
        scores *= decay_factor
        scores = np.maximum(scores, 0.01)

    return solution
