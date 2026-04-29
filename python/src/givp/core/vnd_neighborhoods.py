# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""VND — Neighbourhood functions.

Implements four of the five neighbourhoods used by the VND local search:

- :func:`_neighborhood_swap`     — pair: jointly perturb a continuous+integer pair
- :func:`_neighborhood_multiflip`— k-opt: simultaneously modify k variables
- :func:`_neighborhood_group`    — group: perturb all steps of one structured group
- :func:`_neighborhood_block`    — block: perturb a step-range across all groups

``_neighborhood_flip`` and ``_execute_neighborhood`` live in ``vnd.py`` so
that ``monkeypatch.setattr(core_vnd, "_try_*", mock)`` continues to work in
tests (``_neighborhood_flip`` calls the patchable ``_search_*`` sweep helpers
that are also defined in ``vnd.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.helpers import (
    _expired,
    _get_group_size,
    _get_half,
    _new_rng,
)
from givp.core.vnd_moves import (
    _modify_indices_for_multiflip,
)

__all__ = [
    "_apply_block_perturbation",
    "_apply_group_perturbation",
    "_group_layout",
    "_neighborhood_block",
    "_neighborhood_group",
    "_neighborhood_multiflip",
    "_neighborhood_swap",
    "_sign_from_delta",
]


# ---------------------------------------------------------------------------
# Group / block layout helpers
# ---------------------------------------------------------------------------


def _group_layout(num_vars: int) -> tuple[int, int, int] | None:
    """Infer grouped variable layout ``(half, n_groups, n_steps)`` when valid."""
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


# ---------------------------------------------------------------------------
# Neighbourhood functions
# ---------------------------------------------------------------------------


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
    """Pair neighbourhood: simultaneously perturb a continuous variable and its paired integer.

    Jointly modifies the continuous variable and the integer at the same index,
    capturing correlations between the two halves that 1-opt misses.

    Args:
        cost_fn: Objective function.
        solution: Current solution.
        current_benefit: Current objective value.
        num_vars: Number of variables.
        first_improvement: If True, return on the first improvement found.
        max_attempts: Maximum number of perturbation attempts.
        lower_arr: Lower bounds.
        upper_arr: Upper bounds.

    Returns:
        Tuple ``(best_solution, best_value)``.
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
    """k-opt neighbourhood: simultaneously modify *k* variables.

    Args:
        cost_fn: Objective function.
        solution: Current solution.
        current_benefit: Current objective value.
        num_vars: Number of variables.
        k: Number of variables to modify simultaneously.
        max_attempts: Maximum number of perturbation attempts.
        seed: Random seed.

    Returns:
        Tuple ``(best_solution, best_value)``.
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
    """Group neighbourhood: perturb all steps of one group simultaneously.

    Requires the continuous half to be laid out as
    ``[group0_step0, group0_step1, ..., group1_step0, ...]`` with groups of
    size ``config.group_size``.  Automatically disabled when ``group_size``
    is not configured.
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
    """Block neighbourhood: perturb a contiguous step-block across all groups.

    Selects a random subinterval of 3–6 steps and applies a coordinated
    perturbation to all groups simultaneously.  Automatically disabled when
    ``group_size`` is not configured.
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
