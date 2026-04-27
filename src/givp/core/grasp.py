# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""GRASP — Greedy Randomized Adaptive Search Procedure.

Implements the construction phase of GRASP:
- Candidate-list evaluation for legacy discrete packing
- RCL (Restricted Candidate List) selection
- Latin-style randomised solution construction for continuous/mixed spaces
- Adaptive alpha scheduling
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from givp.core.cache import EvaluationCache
from givp.core.helpers import (
    EvaluatorFn,
    _get_half,
    _new_rng,
    _safe_evaluate,
)
from givp.exceptions import (
    InvalidBoundsError,
    InvalidInitialGuessError,
)

# ---------------------------------------------------------------------------
# Legacy discrete-packing helpers
# ---------------------------------------------------------------------------


def evaluate_candidates(
    available: np.ndarray,
    deps_active: np.ndarray,
    current_cost: int,
    deps_matrix: np.ndarray,
    deps_len: np.ndarray,
    c_arr: np.ndarray,
    a_arr: np.ndarray,
    b: int,
):
    """
    Avalia candidatos com base em informações de dependência e custos incrementais.

    NOTE: This helper is retained for compatibility with legacy discrete packing
    use-cases; the main GRASP flow in this module works in continuous space via
    `construct_grasp` and an external `evaluator`.

    Args:
        available (np.ndarray): Índices dos pacotes disponíveis.
        deps_active (np.ndarray): Array booleano indicando dependências já ativas.
        current_cost (int): Custo total atual (legacy semantics).
        deps_matrix (np.ndarray): Matriz de dependências dos pacotes.
        deps_len (np.ndarray): Array de quantidade de dependências por pacote.
        c_arr (np.ndarray): Array de benefícios dos pacotes.
        a_arr (np.ndarray): Array de custos das dependências.
        b (int): Limite de orçamento.

    Returns:
        tuple: (ratios, incremental_costs, valid_mask)
            ratios (np.ndarray): Razão benefício/custo incremental (legacy).
            incremental_costs (np.ndarray): Custos incrementais dos candidatos.
            valid_mask (np.ndarray): Máscara booleana de candidatos válidos.
    """
    n_available = len(available)
    ratios = np.full(n_available, -np.inf, dtype=np.float32)
    inc_costs = np.zeros(n_available, dtype=np.int32)
    valid = np.zeros(n_available, dtype=np.bool_)
    for idx, i in enumerate(available):
        n_deps = deps_len[i]
        incremental_cost = 0
        if n_deps > 0:
            pkg_deps = deps_matrix[i, :n_deps]
            new_deps_mask = ~deps_active[pkg_deps]
            new_deps = pkg_deps[new_deps_mask]
            if len(new_deps) > 0:
                incremental_cost = a_arr[new_deps].sum()
        if current_cost + incremental_cost <= b:
            valid[idx] = True
            inc_costs[idx] = incremental_cost
            ratios[idx] = c_arr[i] / incremental_cost if incremental_cost > 0 else 1e9
    return ratios, inc_costs, valid


def select_rcl(
    valid_indices: np.ndarray, valid_ratios: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Seleciona a Restricted Candidate List (RCL) com base nas razões benefício/custo e
    parâmetro alpha.

    Args:
        valid_indices (np.ndarray): Índices dos candidatos válidos.
        valid_ratios (np.ndarray): Razões benefício/custo dos candidatos válidos.
        alpha (float): Parâmetro de randomização (0 = guloso, 1 = aleatório).

    Returns:
        np.ndarray: Índices dos candidatos na RCL.
    """
    max_ratio = valid_ratios.max()
    min_ratio = valid_ratios.min()
    threshold = max_ratio - alpha * (max_ratio - min_ratio)
    rcl_mask = valid_ratios >= threshold
    rcl_indices = valid_indices[rcl_mask]
    if len(rcl_indices) == 0:
        n_top = max(1, int(len(valid_indices) * 0.3))
        top_idx = np.argpartition(valid_ratios, -n_top)[-n_top:]
        rcl_indices = valid_indices[top_idx]
    return np.asarray(rcl_indices)


# ---------------------------------------------------------------------------
# Bounds and initial guess validation
# ---------------------------------------------------------------------------


def _validate_bounds_and_initial(
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    initial_guess: np.ndarray | None,
    num_vars: int,
):
    if lower_arr.shape[0] != num_vars or upper_arr.shape[0] != num_vars:
        raise InvalidBoundsError(
            f"lower (len={lower_arr.shape[0]}) and upper (len={upper_arr.shape[0]}) "
            f"must both have length num_vars={num_vars}"
        )
    if initial_guess is not None:
        if initial_guess.shape[0] != num_vars:
            raise InvalidInitialGuessError(
                f"initial_guess has length {initial_guess.shape[0]}, expected {num_vars}"
            )
        if np.any(initial_guess <= lower_arr) or np.any(initial_guess >= upper_arr):
            bad = np.nonzero(
                (initial_guess <= lower_arr) | (initial_guess >= upper_arr)
            )[0]
            raise InvalidInitialGuessError(
                f"initial_guess values must be strictly between lower and upper; "
                f"violating indices: {bad.tolist()[:10]}"
            )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _seed_from_initial(
    chute: np.ndarray,
    num_vars: int,
    evaluator: EvaluatorFn,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
) -> np.ndarray:
    cost = _safe_evaluate(evaluator, chute)
    if np.isfinite(cost):
        copy: np.ndarray = chute.copy()
        return copy
    rng = _new_rng()
    result: np.ndarray = np.asarray(
        lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)
    )
    return result


def _evaluate_with_cache(
    cand: np.ndarray, evaluator: EvaluatorFn, cache: EvaluationCache | None
) -> float:
    """Avalia candidato usando cache se disponível."""
    if cache is not None:
        cached_cost = cache.get(cand)
        if cached_cost is not None:
            return cached_cost

    cost = _safe_evaluate(evaluator, cand)
    if cache is not None and np.isfinite(cost):
        cache.put(cand, cost)
    return cost


def _select_from_rcl(
    costs: np.ndarray, alpha: float, rng: np.random.Generator
) -> int | None:
    valid_mask = np.isfinite(costs)
    if not np.any(valid_mask):
        return None
    valid_idx = np.nonzero(valid_mask)[0]
    valid_costs = costs[valid_idx]
    min_cost = valid_costs.min()
    max_cost = valid_costs.max()
    threshold = min_cost + alpha * (max_cost - min_cost)
    rcl_local = valid_idx[valid_costs <= threshold]
    if rcl_local.size == 0:
        rcl_local = valid_idx
    return int(rng.choice(rcl_local))


def _normalize_integer_tail(sol: np.ndarray, half: int) -> None:
    """Round integer-part variables in-place."""
    for idx in range(half, sol.size):
        sol[idx] = float(int(np.rint(sol[idx])))


def _build_seed_candidate(
    initial_guess: np.ndarray | None,
    num_vars: int,
    evaluator: Callable,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    cache: EvaluationCache | None,
) -> tuple[np.ndarray, float] | None:
    """Build and evaluate candidate from initial guess when available."""
    if initial_guess is None:
        return None
    seed_candidate = _seed_from_initial(
        initial_guess, num_vars, evaluator, lower_arr, upper_arr
    )
    _normalize_integer_tail(seed_candidate, _get_half(num_vars))
    seed_cost = _evaluate_with_cache(seed_candidate, evaluator, cache)
    return seed_candidate, seed_cost


def _sample_integer_from_bounds(
    lower: float, upper: float, rng: np.random.Generator
) -> float:
    """Sample one integer variable respecting numeric bounds."""
    lo = int(np.ceil(lower))
    hi = int(np.floor(upper))
    if hi >= lo:
        return float(rng.integers(lo, hi + 1))
    return float(int(np.rint((lower + upper) / 2.0)))


def _build_heuristic_candidate(
    num_vars: int,
    half: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one heuristic candidate using proportional dispatch idea."""
    sol = np.empty(num_vars, dtype=float)
    mid = (lower_arr[:half] + upper_arr[:half]) / 2.0
    span = upper_arr[:half] - lower_arr[:half]
    noise = rng.uniform(-0.15, 0.15, size=half)
    sol[:half] = np.clip(mid + noise * span, lower_arr[:half], upper_arr[:half])

    for idx in range(half, num_vars):
        lo = int(np.ceil(lower_arr[idx]))
        hi = int(np.floor(upper_arr[idx]))
        cont_idx = idx - half
        if hi > lo and span[cont_idx] > 0:
            frac = (sol[cont_idx] - lower_arr[cont_idx]) / span[cont_idx]
            target = lo + frac * (hi - lo)
            sol[idx] = float(int(np.clip(np.rint(target), lo, hi)))
        else:
            sol[idx] = float(
                hi
                if hi >= lo
                else int(np.rint((lower_arr[idx] + upper_arr[idx]) / 2.0))
            )

    return sol


def _build_random_candidate(
    num_vars: int,
    half: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one random mixed continuous/integer candidate."""
    sol = np.empty(num_vars, dtype=float)
    sol[:half] = lower_arr[:half] + (upper_arr[:half] - lower_arr[:half]) * rng.random(
        half
    )
    for idx in range(half, num_vars):
        sol[idx] = _sample_integer_from_bounds(lower_arr[idx], upper_arr[idx], rng)
    return sol


def _evaluate_candidates_batch(
    candidates: list[np.ndarray],
    evaluated_count: int,
    evaluator: Callable,
    cache: EvaluationCache | None,
    n_workers: int,
) -> list[float]:
    """Evaluate candidates not yet evaluated (optionally in parallel)."""
    unevaluated = candidates[evaluated_count:]
    if n_workers > 1 and len(unevaluated) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            return list(
                executor.map(
                    lambda s: _evaluate_with_cache(s, evaluator, cache),
                    unevaluated,
                )
            )
    candidates_batch = [
        _evaluate_with_cache(sol, evaluator, cache) for sol in unevaluated
    ]
    return candidates_batch


# ---------------------------------------------------------------------------
# Public construction entry-point
# ---------------------------------------------------------------------------


def construct_grasp(
    num_vars,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    evaluator: Callable,
    initial_guess: np.ndarray | None,
    alpha: float,
    seed: int | None = None,
    num_candidates_per_step: int | None = None,
    cache: EvaluationCache | None = None,
    n_workers: int = 1,
):
    """
    Constrói uma solução inicial por amostragem aleatória (Latin-style) + avaliação.

    Gera N soluções aleatórias respeitando limites, avalia todas, e seleciona
    a melhor via RCL (Restricted Candidate List). Muito mais eficiente que a
    construção greedy coordenada-a-coordenada para espaço contínuo.

    Args:
        num_vars (int): Número de variáveis.
        lower_arr (np.ndarray): Limites inferiores.
        upper_arr (np.ndarray): Limites superiores.
        evaluator (Callable): Função de avaliação.
        initial_guess (np.ndarray | None): Solução inicial sugerida.
        alpha (float): Parâmetro de randomização para RCL.
        seed (int, optional): Semente para aleatoriedade.
        num_candidates_per_step (int, optional): Número de soluções candidatas a gerar.
        cache (EvaluationCache | None): Cache de avaliações.

    Returns:
        np.ndarray: Melhor solução construída.
    """
    rng = _new_rng(seed)
    _validate_bounds_and_initial(lower_arr, upper_arr, initial_guess, num_vars)

    half = _get_half(num_vars)
    n_candidates = max(num_candidates_per_step or 10, 5)
    candidates: list[np.ndarray] = []
    costs: list[float] = []

    seed_data = _build_seed_candidate(
        initial_guess,
        num_vars,
        evaluator,
        lower_arr,
        upper_arr,
        cache,
    )
    if seed_data is not None:
        seed_candidate, seed_cost = seed_data
        candidates.append(seed_candidate)
        costs.append(seed_cost)
        n_candidates -= 1

    n_heuristic = max(1, n_candidates // 2)
    n_random = n_candidates - n_heuristic

    for _ in range(n_heuristic):
        candidates.append(
            _build_heuristic_candidate(num_vars, half, lower_arr, upper_arr, rng)
        )
    for _ in range(n_random):
        candidates.append(
            _build_random_candidate(num_vars, half, lower_arr, upper_arr, rng)
        )

    costs.extend(
        _evaluate_candidates_batch(
            candidates,
            evaluated_count=len(costs),
            evaluator=evaluator,
            cache=cache,
            n_workers=n_workers,
        )
    )

    costs_arr = np.array(costs)
    chosen = _select_from_rcl(costs_arr, alpha, rng)
    if chosen is None:
        return candidates[0]
    return candidates[chosen]


# ---------------------------------------------------------------------------
# Adaptive alpha
# ---------------------------------------------------------------------------


def get_current_alpha(iteration: int, config) -> float:
    """Calcula alpha adaptativo baseado no progresso das iterações."""
    if config.adaptive_alpha:
        progress = iteration / max(1, config.max_iterations - 1)
        current_alpha = (
            config.alpha_min + (config.alpha_max - config.alpha_min) * progress
        )
        current_alpha += float(_new_rng().uniform(-0.02, 0.02))
        return float(np.clip(current_alpha, config.alpha_min, config.alpha_max))
    return config.alpha
