# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""ILS — Iterated Local Search.

Implements the ILS phase of GRASP-ILS-VND-PR:
- Solution perturbation with adaptive strength
- ILS loop with optional acceptance of worse solutions for diversification
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.cache import EvaluationCache
from givp.core.helpers import (
    _expired,
    _new_rng,
)
from givp.core.vnd import _perturb_index, local_search_vnd

# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def perturb_solution_numpy(
    solution: np.ndarray,
    num_vars: int,
    strength: int = 4,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aplica perturbação à solução, invertendo 'strength' bits aleatórios.

    Args:
        solution (np.ndarray): Solução original.
        num_vars (int): Número de variáveis.
        strength (int): Intensidade da perturbação (quantos bits inverter).
        seed (int, optional): Semente aleatória.

    Returns:
        np.ndarray: Solução perturbada.
    """
    perturbed: np.ndarray = solution.copy().astype(float)
    rng = _new_rng(seed)
    # P15: perturbação mais agressiva — num_vars//5 variáveis para escapar ótimos locais
    n_perturb = min(max(strength, num_vars // 5), num_vars)
    indices = rng.choice(num_vars, size=n_perturb, replace=False)
    for idx in indices:
        _perturb_index(perturbed, idx, strength, rng, lower_arr, upper_arr)
    return perturbed


# ---------------------------------------------------------------------------
# ILS search loop
# ---------------------------------------------------------------------------


def ils_search(
    solution: np.ndarray,
    current_cost: float,
    num_vars: int,
    cost_fn: Callable,
    config,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: EvaluationCache | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Iterated Local Search (ILS) sobre a solução, aplicando perturbações e busca local.

    Args:
        solution (np.ndarray): Solução inicial.
        current_cost (float): Custo atual (minimização).
        num_vars (int): Número de variáveis.
        cost_fn (Callable): Função de custo.
        config: Configuração do algoritmo (GIVPConfig).
        lower_arr (np.ndarray, optional): Limites inferiores das variáveis.
        upper_arr (np.ndarray, optional): Limites superiores das variáveis.
        cache (EvaluationCache, optional): Cache de avaliações.

    Returns:
        tuple: (solução final, custo final)
    """
    best_solution = solution.copy()
    best_cost = current_cost
    for ils_iter in range(config.ils_iterations):
        if _expired(deadline):
            break
        # P12: amplitude progressiva suave (5%→12%) em vez de dobrar na metade
        progress = ils_iter / max(1, config.ils_iterations - 1)
        adaptive_strength = max(
            config.perturbation_strength,
            int(config.perturbation_strength * (1.0 + progress)),
        )
        perturbed = perturb_solution_numpy(
            solution,
            num_vars,
            strength=adaptive_strength,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
        )
        perturbed = local_search_vnd(
            cost_fn,
            perturbed,
            num_vars,
            max_iter=config.vnd_iterations // 2,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            cache=cache,
            deadline=deadline,
        )
        perturbed_cost = cost_fn(perturbed)
        # P15: aceitar pior com probabilidade maior (max 25%) para escapar ótimos locais
        temperature = 1.0 - progress
        accept_worse = _new_rng().random() < temperature * 0.25
        if perturbed_cost < current_cost or accept_worse:
            solution = perturbed
            current_cost = perturbed_cost
        # Sempre manter a melhor solução global
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = solution.copy()
    # Retornar a melhor encontrada, não a última aceita
    return best_solution, best_cost
