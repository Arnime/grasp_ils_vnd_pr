# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""VND — Atomic move helpers for integer and continuous variables.

Lowest-level move operators used by the neighbourhood functions:

- :func:`_try_integer_moves_module`   -- 1-opt candidates for one integer variable
- :func:`_try_continuous_move_module` -- small perturbation for one continuous variable
- :func:`_modify_indices_for_multiflip`  -- simultaneous k-opt index modifier
- :func:`_perturb_index`              -- single-index perturbation (ILS helper)

Note: the sweep functions ``_search_integer_flip_module`` /
``_search_continuous_flip_module`` live in ``vnd.py`` so that
``monkeypatch.setattr(core_vnd, "_try_*", mock)`` works in tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from givp.core.helpers import _get_half

__all__ = [
    "_modify_indices_for_multiflip",
    "_perturb_index",
    "_try_continuous_move_module",
    "_try_integer_moves_module",
]


def _try_integer_moves_module(
    idx: int,
    sol: np.ndarray,
    best_benefit: float,
    cost_fn: Callable,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    """Try +/-1 integer moves at *idx*; return first improvement or original."""
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
    """Perturb *sol[idx]* by +/-5% of its span; accept if strictly better."""
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


def _modify_indices_for_multiflip(
    solution: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    """Apply simultaneous perturbations to *indices* for k-opt; return old values."""
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
    """Perturb a single variable in-place (integer +/-step, continuous +/-15% span)."""
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
            # P15: amplitude aumentada para +/-15% para diversificacao efetiva
            delta = rng.uniform(-0.15 * span, 0.15 * span)
            perturbed[idx] = float(np.clip(old + delta, lower_arr[idx], upper_arr[idx]))
        else:
            delta = rng.normal(scale=0.12 * (abs(old) + 1e-6))
            perturbed[idx] = old + delta