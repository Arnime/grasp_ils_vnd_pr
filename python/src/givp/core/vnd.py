# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""VND — Variable Neighborhood Descent.

Implements the local-search phase of GRASP-ILS-VND-PR:
- Five neighbourhoods: flip (1-opt), swap (pair), group, block, multiflip (k-opt)
- Standard VND with fixed neighbourhood ordering
- Adaptive VND with Adaptive Neighbourhood Selection (ANS / roulette-wheel)
- Variable sensitivity tracking (P9)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.cache import EvaluationCache
from givp.core.helpers import (
    _expired,
    _get_group_size,
    _get_half,
    _new_rng,
)

# ---------------------------------------------------------------------------
# Cached cost-function wrapper
# ---------------------------------------------------------------------------


def _create_cached_cost_fn(
    cost_fn: Callable, cache: EvaluationCache | None
) -> Callable:
    """Cria wrapper de função de custo com cache."""

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
# Atomic move helpers
# ---------------------------------------------------------------------------


def _try_integer_moves_module(
    idx: int,
    sol: np.ndarray,
    best_benefit: float,
    cost_fn: Callable,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    old = sol[idx]
    base = int(np.rint(old))
    cand_vals = (base - 1, base, base + 1)
    lo = int(np.ceil(lower_arr[idx])) if (lower_arr is not None) else -np.inf
    hi = int(np.floor(upper_arr[idx])) if (upper_arr is not None) else np.inf
    for v in cand_vals:
        if v < lo or v > hi:
            continue
        sol[idx] = float(v)
        c = cost_fn(sol)
        if c < best_benefit:
            return sol.copy(), c, True
    sol[idx] = old
    return sol, best_benefit, False


def _try_continuous_move_module(
    idx: int,
    sol: np.ndarray,
    best_benefit: float,
    cost_fn: Callable,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    old = sol[idx]
    if lower_arr is not None and upper_arr is not None:
        span = upper_arr[idx] - lower_arr[idx]
        perturb = rng.uniform(-0.05 * span, 0.05 * span)
    else:
        perturb = rng.uniform(-0.1, 0.1)
    new_val = old + perturb
    if lower_arr is not None and upper_arr is not None:
        new_val = float(np.clip(new_val, lower_arr[idx], upper_arr[idx]))
    sol[idx] = new_val
    cost = cost_fn(sol)
    if cost < best_benefit:
        return True, cost
    sol[idx] = old
    return False, best_benefit


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


def _modify_indices_for_multiflip(
    solution: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    half = _get_half(solution.size)
    old_vals = solution[indices].copy()

    int_mask = indices >= half
    int_indices = indices[int_mask]
    if int_indices.size > 0:
        base = np.rint(solution[int_indices]).astype(int)
        deltas = rng.integers(-1, 2, size=int_indices.size)
        new_vals = base + deltas
        if lower_arr is not None and upper_arr is not None:
            lo = np.ceil(lower_arr[int_indices]).astype(int)
            hi = np.floor(upper_arr[int_indices]).astype(int)
            new_vals = np.clip(new_vals, lo, hi)
        solution[int_indices] = new_vals.astype(float)

    cont_mask = indices < half
    cont_indices = indices[cont_mask]
    if cont_indices.size > 0:
        if lower_arr is not None and upper_arr is not None:
            spans = upper_arr[cont_indices] - lower_arr[cont_indices]
            perturb = rng.uniform(-0.05, 0.05, size=cont_indices.size) * spans
        else:
            perturb = rng.uniform(-0.1, 0.1, size=cont_indices.size)
        new_vals = solution[cont_indices] + perturb
        if lower_arr is not None and upper_arr is not None:
            new_vals = np.clip(
                new_vals, lower_arr[cont_indices], upper_arr[cont_indices]
            )
        solution[cont_indices] = new_vals

    return old_vals


def _perturb_index(
    perturbed: np.ndarray,
    idx: int,
    strength: int,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    half = _get_half(perturbed.size)
    old = perturbed[idx]
    if idx >= half:
        lo = int(np.ceil(lower_arr[idx])) if lower_arr is not None else None
        hi = int(np.floor(upper_arr[idx])) if upper_arr is not None else None
        step = max(1, int(round(strength / 2)))
        delta = int(rng.integers(-step, step + 1))
        new_val = int(np.rint(old)) + delta
        if lo is not None and hi is not None:
            new_val = int(np.clip(new_val, lo, hi))
        perturbed[idx] = float(new_val)
    else:
        if lower_arr is not None and upper_arr is not None:
            span = upper_arr[idx] - lower_arr[idx]
            # P15: amplitude aumentada para ±15% para diversificação efetiva
            delta = rng.uniform(-0.15 * span, 0.15 * span)
            perturbed[idx] = float(np.clip(old + delta, lower_arr[idx], upper_arr[idx]))
        else:
            delta = rng.normal(scale=0.12 * (abs(old) + 1e-6))
            perturbed[idx] = old + delta


# ---------------------------------------------------------------------------
# Neighbourhood functions
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
    """
    Vizinhança 1-opt: Tenta inverter cada bit individualmente na solução.

    P9: Se sensitivity array fornecido, prioriza variáveis com maior sensibilidade
    (maior impacto histórico no custo), reduzindo avaliações desperdiçadas.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        first_improvement (bool): Se True, retorna na primeira melhoria.
        seed (int, optional): Semente aleatória.
        sensitivity (np.ndarray, optional): P9 — scores de sensibilidade por variável.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """
    rng = _new_rng(seed)
    half = _get_half(num_vars)
    if sensitivity is not None and np.any(sensitivity > 0):
        # P9: ordenar por sensibilidade (maior primeiro), com ruído para exploração
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


def _neighborhood_swap(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 50,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança de par: perturba uma variável contínua e sua correspondente inteira simultaneamente.

    Modifica conjuntamente a variável contínua e a inteira de mesmo índice,
    capturando correlações entre as duas metades que a busca 1-opt ignora.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        first_improvement (bool): Se True, retorna na primeira melhoria.
        max_attempts (int): Número máximo de tentativas.
        lower_arr (np.ndarray | None): Limites inferiores.
        upper_arr (np.ndarray | None): Limites superiores.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = _new_rng()
    half = _get_half(num_vars)
    if half <= 0 or half >= num_vars:
        # Neighborhood requires both continuous and integer halves.
        return best_solution, best_benefit
    for _ in range(max_attempts):
        if _expired(deadline):
            break
        cont_idx = rng.integers(0, half)
        int_idx = cont_idx + half

        old_cont = solution[cont_idx]
        old_int = solution[int_idx]

        if lower_arr is not None and upper_arr is not None:
            span_cont = upper_arr[cont_idx] - lower_arr[cont_idx]
            solution[cont_idx] = float(
                np.clip(
                    old_cont + rng.uniform(-0.08 * span_cont, 0.08 * span_cont),
                    lower_arr[cont_idx],
                    upper_arr[cont_idx],
                )
            )
            lo_int = int(np.ceil(lower_arr[int_idx]))
            hi_int = int(np.floor(upper_arr[int_idx]))
            new_int = int(np.rint(old_int)) + int(rng.integers(-1, 2))
            solution[int_idx] = float(np.clip(new_int, lo_int, hi_int))
        else:
            solution[cont_idx] = old_cont + rng.uniform(-0.1, 0.1)
            solution[int_idx] = float(int(np.rint(old_int)) + int(rng.integers(-1, 2)))

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        solution[cont_idx] = old_cont
        solution[int_idx] = old_int

    return best_solution, best_benefit


def _neighborhood_multiflip(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    k: int = 3,
    max_attempts: int = 50,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança k-opt: Inverte k bits simultaneamente na solução.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        k (int): Número de bits a inverter.
        max_attempts (int): Número máximo de tentativas.
        seed (int, optional): Semente aleatória.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = _new_rng(seed)

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        indices = rng.choice(num_vars, size=min(k, num_vars), replace=False)
        old_vals = _modify_indices_for_multiflip(
            solution, indices, rng, lower_arr, upper_arr
        )
        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
        solution[indices] = old_vals

    return best_solution, best_benefit


# ---------------------------------------------------------------------------
# Group / block neighbourhoods
# ---------------------------------------------------------------------------


def _group_layout(num_vars: int) -> tuple[int, int, int] | None:
    """Infer grouped variable layout (half, n_groups, n_steps) when valid."""
    half = _get_half(num_vars)
    if half <= 0 or half >= num_vars:
        return None
    n_steps = _get_group_size()
    if n_steps is None or n_steps < 1:
        return None
    n_groups = half // n_steps
    if n_groups < 1 or n_groups * n_steps != half:
        return None
    return half, n_groups, n_steps


def _sign_from_delta(delta: float) -> int:
    """Return discrete direction sign from continuous delta."""
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _apply_group_perturbation(
    solution: np.ndarray,
    old_cont: np.ndarray,
    old_int: np.ndarray,
    start: int,
    half: int,
    n_steps: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Apply a correlated perturbation to all steps of one group."""
    end = start + n_steps
    span = upper_arr[start:end] - lower_arr[start:end]
    base_delta = rng.uniform(-0.05, 0.05)
    noise = rng.uniform(-0.02, 0.02, size=n_steps)
    solution[start:end] = np.clip(
        old_cont + (base_delta + noise) * span,
        lower_arr[start:end],
        upper_arr[start:end],
    )
    delta_int = _sign_from_delta(base_delta)
    int_start = half + start
    for step_idx in range(n_steps):
        lo = int(np.ceil(lower_arr[int_start + step_idx]))
        hi = int(np.floor(upper_arr[int_start + step_idx]))
        new_val = int(np.rint(old_int[step_idx])) + delta_int
        solution[int_start + step_idx] = float(np.clip(new_val, lo, hi))


def _apply_block_perturbation(
    solution: np.ndarray,
    old_vals: np.ndarray,
    half: int,
    n_groups: int,
    n_steps: int,
    block_start: int,
    block_end: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    base_delta: float,
) -> None:
    """Apply a coordinated perturbation to a contiguous step-block across all groups."""
    int_delta = _sign_from_delta(base_delta)
    for group_idx in range(n_groups):
        offset = group_idx * n_steps
        for step_idx in range(block_start, block_end):
            cont_idx = offset + step_idx
            int_idx = half + cont_idx
            span = upper_arr[cont_idx] - lower_arr[cont_idx]
            solution[cont_idx] = float(
                np.clip(
                    old_vals[cont_idx] + base_delta * span,
                    lower_arr[cont_idx],
                    upper_arr[cont_idx],
                )
            )
            lo = int(np.ceil(lower_arr[int_idx]))
            hi = int(np.floor(upper_arr[int_idx]))
            solution[int_idx] = float(
                np.clip(int(np.rint(old_vals[int_idx])) + int_delta, lo, hi)
            )


def _neighborhood_group(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 30,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança de grupo: perturba todos os passos de um mesmo grupo simultaneamente.

    Requer que a metade contínua esteja organizada como
    [group0_step0, group0_step1, ..., group1_step0, ...] com grupos de
    tamanho ``config.group_size``.  Desativada automaticamente se
    ``group_size`` não estiver configurado.
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = _new_rng()
    layout = _group_layout(num_vars)
    if layout is None:
        return best_solution, best_benefit
    half, n_groups, n_steps = layout
    if lower_arr is None or upper_arr is None:
        return best_solution, best_benefit

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        group_idx = rng.integers(0, n_groups)
        start = group_idx * n_steps
        end = start + n_steps

        old_cont = solution[start:end].copy()
        old_int = solution[half + start : half + end].copy()

        _apply_group_perturbation(
            solution,
            old_cont,
            old_int,
            int(start),
            half,
            n_steps,
            lower_arr,
            upper_arr,
            rng,
        )

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        solution[start:end] = old_cont
        solution[half + start : half + end] = old_int

    return best_solution, best_benefit


def _neighborhood_block(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 30,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança de bloco: perturba um bloco contíguo de passos em todos os grupos.

    Seleciona aleatoriamente um subintervalo de 3–6 passos e aplica uma
    perturbação coordenada em todos os grupos simultaneamente.  Desativada
    automaticamente se ``group_size`` não estiver configurado.
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = _new_rng()
    layout = _group_layout(num_vars)
    if layout is None:
        return best_solution, best_benefit
    half, n_groups, n_steps = layout
    if lower_arr is None or upper_arr is None:
        return best_solution, best_benefit

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        block_size = rng.integers(3, min(7, n_steps + 1))
        b_start = rng.integers(0, n_steps - block_size + 1)
        b_end = b_start + block_size

        old_vals = solution.copy()

        base_delta = rng.uniform(-0.04, 0.04)
        _apply_block_perturbation(
            solution,
            old_vals,
            half,
            n_groups,
            n_steps,
            int(b_start),
            int(b_end),
            lower_arr,
            upper_arr,
            base_delta,
        )

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        np.copyto(solution, old_vals)

    return best_solution, best_benefit


# ---------------------------------------------------------------------------
# VND orchestration
# ---------------------------------------------------------------------------


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
    """Tenta vizinhanças 1-opt, pair, group, block e multiflip.

    Retorna (solution, benefit, improved).
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
    """Executa uma vizinhança específica pelo índice.

    Args:
        idx: Índice da vizinhança (0=flip, 1=pair, 2=group, 3=block, 4=multiflip).
        cost_fn: Função de custo.
        solution: Solução atual.
        current_benefit: Custo atual.
        num_vars: Número de variáveis.
        first_improvement: Se True, aceita primeira melhoria.
        lower_arr: Limites inferiores.
        upper_arr: Limites superiores.
        sensitivity: Scores de sensibilidade por variável (P9).

    Returns:
        tuple: (melhor_solução, melhor_benefício).
    """
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
    """
    Executa busca local usando Variable Neighborhood Descent (VND),
    alternando entre vizinhanças 1-opt, 2-opt e multi-flip com cache.

    Args:
        cost_fn (Callable): Função de custo a maximizar.
        solution (np.ndarray): Solução inicial.
        num_vars (int): Número de variáveis.
        max_iter (int): Máximo de iterações.
        use_first_improvement (bool): Se True, aceita primeira melhoria encontrada.
        no_improve_limit (int): Limite de iterações sem melhoria.
        no_improve_flip_limit (int): Frequência para multi-flip.
        cache: Cache de avaliações (opcional).

    Returns:
        np.ndarray: Solução melhorada.
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
    """
    VND com Seleção Adaptativa de Vizinhança (Adaptive Neighborhood Selection — ANS).

    Em vez de aplicar vizinhanças em ordem fixa (flip→pair→group→block→multiflip),
    usa roulette-wheel proporcional ao sucesso histórico de cada vizinhança.
    Vizinhanças que geram melhorias recebem recompensa; todas sofrem decaimento
    exponencial para permitir readaptação ao longo da busca.

    Args:
        cost_fn: Função de custo (minimização).
        solution: Solução inicial.
        num_vars: Número de variáveis.
        max_iter: Máximo de iterações.
        use_first_improvement: Se True, aceita primeira melhoria encontrada.
        no_improve_limit: Limite de iterações sem melhoria para parar.
        lower_arr: Limites inferiores das variáveis.
        upper_arr: Limites superiores das variáveis.
        cache: Cache de avaliações (opcional).
        reward_factor: Fator de recompensa para vizinhança bem-sucedida.
        decay_factor: Fator de decaimento dos scores por iteração (0 < d < 1).
        min_probability: Probabilidade mínima de seleção por vizinhança.

    Returns:
        np.ndarray: Solução melhorada.
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
