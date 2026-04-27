# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Path Relinking.

Implements the path-relinking intensification phase:
- Forward path relinking (greedy step-by-step)
- Best-move path relinking (select best move at each step)
- Bidirectional path relinking (explore both directions)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.helpers import (
    _expired,
    _new_rng,
)

# ---------------------------------------------------------------------------
# Internal move helpers
# ---------------------------------------------------------------------------


def _find_best_move(
    cost_fn,
    current,
    target,
    indices,
    source,
    best_benefit,
    diff_indices,
    deadline=0.0,
):
    best_move_idx = None
    best_move_benefit = best_benefit
    for count, idx in enumerate(indices):
        if count % 5 == 0 and _expired(deadline):
            break
        if current[idx] == target[idx]:
            continue
        current[idx] = target[idx]
        cost = cost_fn(current)
        if cost < best_move_benefit:
            best_move_benefit = cost
            best_move_idx = idx
        current[idx] = source[idx] if idx in diff_indices else current[idx]
    return best_move_idx, best_move_benefit


def _path_relinking_best(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    diff_indices: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    current = source.copy().astype(float)
    best_solution = current.copy()
    best_benefit = cost_fn(current)
    indices = diff_indices.copy()
    while len(indices) > 0:
        if _expired(deadline):
            break
        best_move_idx, best_move_benefit = _find_best_move(
            cost_fn,
            current,
            target,
            indices,
            source,
            best_benefit,
            diff_indices,
            deadline=deadline,
        )
        if best_move_idx is not None:
            current[best_move_idx] = target[best_move_idx]
            indices = indices[indices != best_move_idx]
            if best_move_benefit < best_benefit:
                best_benefit = best_move_benefit
                best_solution = current.copy()
        else:
            break
    return best_solution, best_benefit


def _path_relinking_forward(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    diff_indices: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    current = source.copy().astype(float)
    best_solution = current.copy()
    best_benefit = cost_fn(current)
    for count, idx in enumerate(diff_indices):
        if count % 5 == 0 and _expired(deadline):
            break
        current[idx] = target[idx]
        cost = cost_fn(current)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = current.copy()
    return best_solution, best_benefit


# ---------------------------------------------------------------------------
# Public path relinking entry-points
# ---------------------------------------------------------------------------


def path_relinking(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    strategy: str = "best",
    seed: int | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Path Relinking entre duas soluções, modificando um bit por vez na direção da
    solução destino.

    Args:
        cost_fn (Callable): Função de custo.
        source (np.ndarray): Solução origem.
        target (np.ndarray): Solução destino.
        strategy (str): 'best' (melhor a cada passo) ou 'forward' (todos os passos).
        seed (int, optional): Semente aleatória.

    Returns:
        tuple: (melhor_solução_no_caminho, melhor_benefício)
    """
    source = np.array(source, dtype=float)
    target = np.array(target, dtype=float)
    diff_indices = np.nonzero(np.abs(source - target) > 1e-9)[0]
    if len(diff_indices) == 0:
        return source.copy(), cost_fn(source)

    # Limitar a top-K variáveis mais diferentes para evitar O(n²) no path relinking
    max_pr_vars = 25
    if len(diff_indices) > max_pr_vars:
        diffs = np.abs(source[diff_indices] - target[diff_indices])
        top_k_local = np.argpartition(diffs, -max_pr_vars)[-max_pr_vars:]
        diff_indices = diff_indices[top_k_local]

    rng = _new_rng(seed)
    rng.shuffle(diff_indices)
    if strategy == "best":
        return _path_relinking_best(
            cost_fn, source, target, diff_indices, deadline=deadline
        )
    return _path_relinking_forward(
        cost_fn, source, target, diff_indices, deadline=deadline
    )


def bidirectional_path_relinking(
    cost_fn: Callable,
    sol1: np.ndarray,
    sol2: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Path Relinking bidirecional entre duas soluções, explorando ambos os caminhos.

    Args:
        cost_fn (Callable): Função de custo.
        sol1 (np.ndarray): Primeira solução.
        sol2 (np.ndarray): Segunda solução.

    Returns:
        tuple: Melhor solução encontrada e seu benefício.
    """
    best1, cost1 = path_relinking(
        cost_fn, sol1, sol2, strategy="forward", deadline=deadline
    )

    if _expired(deadline):
        return best1, cost1

    best2, cost2 = path_relinking(
        cost_fn, sol2, sol1, strategy="forward", deadline=deadline
    )

    if cost1 <= cost2:
        return best1, cost1
    return best2, cost2
