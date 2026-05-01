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

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pickle import PicklingError

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

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCL selection — two variants for different problem types
# ---------------------------------------------------------------------------


def select_rcl(
    valid_indices: np.ndarray, valid_ratios: np.ndarray, alpha: float
) -> np.ndarray:
    """Build the Restricted Candidate List (RCL) using benefit/cost ratios and alpha.

    This is the **ratio-based** RCL selector, suited for problems where
    candidates are ranked by a benefit/cost ratio (e.g. discrete or
    mixed-integer knapsack-style formulations where *higher* ratios are
    better).  The threshold is computed as::

        threshold = max_ratio - alpha * (max_ratio - min_ratio)

    Candidates with ``ratio >= threshold`` enter the RCL.

    For **continuous minimisation** problems (where lower cost is better)
    use the internal :func:`_select_from_rcl`, which applies the symmetric
    cost-based threshold::

        threshold = min_cost + alpha * (max_cost - min_cost)

    Both functions implement the same alpha semantics:
    ``alpha = 0`` → purely greedy, ``alpha = 1`` → uniformly random.

    Args:
        valid_indices: Indices of feasible candidates.
        valid_ratios: Benefit/cost ratios for the feasible candidates
            (higher is better).
        alpha: Randomisation parameter in ``[0, 1]``.  ``0`` is purely
            greedy (only the best ratio), ``1`` admits all candidates.

    Returns:
        Array of indices of candidates selected into the RCL.
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
) -> None:
    """Validate bounds and initial guess."""
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
    """Build a candidate from the initial guess, rounding integer variables and evaluating."""
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
    """Evaluate a candidate using the cache when available."""
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
    """Select an index from the restricted candidate list (RCL) based on costs and alpha."""
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
    """Round integer-part variables in-place (vectorised NumPy operation)."""
    if half < sol.size:
        sol[half:] = np.rint(sol[half:])


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


def _parallel_worker(args: tuple[np.ndarray, Callable]) -> float:
    """Module-level worker for ProcessPoolExecutor (standard pickle).

    Must be a top-level (non-closure) function to remain picklable.
    Closures and lambdas cause a ``PicklingError`` — handled in
    ``_evaluate_candidates_batch`` by trying ``_cloudpickle_worker`` first,
    then falling back to ``ThreadPoolExecutor``.
    """
    s, evaluator = args
    try:
        v = float(evaluator(s))
        return v if np.isfinite(v) else float("inf")
    except Exception:  # pylint: disable=broad-exception-caught
        return float("inf")


def _cloudpickle_worker(args: tuple[np.ndarray, bytes]) -> float:
    """Module-level worker for ProcessPoolExecutor using cloudpickle serialisation.

    Deserialises the evaluator from ``cloudpickle.dumps`` bytes, enabling
    closures and locally-defined functions (which are not picklable with
    standard ``pickle``) to still run in separate processes and bypass the GIL.

    Requires the optional ``cloudpickle`` package::

        pip install "givp[parallel]"
    """
    # Optional dependency for closure/lambda serialisation:
    # pip install "givp[parallel]"
    import cloudpickle  # type: ignore[import-not-found,import-untyped]

    s, serialized = args
    evaluator = cloudpickle.loads(serialized)
    try:
        v = float(evaluator(s))
        return v if np.isfinite(v) else float("inf")
    except Exception:  # pylint: disable=broad-exception-caught
        return float("inf")


def _try_standard_process_pool(
    unevaluated: list[np.ndarray],
    evaluator: Callable,
    n_workers: int,
) -> tuple[list[float] | None, Exception | None]:
    """Attempt candidate evaluation using standard ProcessPoolExecutor."""
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return (
                list(
                    pool.map(
                        _parallel_worker,
                        ((s, evaluator) for s in unevaluated),
                    )
                ),
                None,
            )
    except (PicklingError, AttributeError, TypeError, OSError) as exc:
        return None, exc


def _try_cloudpickle_process_pool(
    unevaluated: list[np.ndarray],
    evaluator: Callable,
    n_workers: int,
) -> tuple[list[float] | None, Exception | None]:
    """Attempt candidate evaluation using cloudpickle + ProcessPoolExecutor."""
    try:
        # Optional dependency for closure/lambda serialisation:
        # pip install "givp[parallel]"
        import cloudpickle  # type: ignore[import-not-found,import-untyped]

        serialized = cloudpickle.dumps(evaluator)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return (
                list(
                    pool.map(
                        _cloudpickle_worker,
                        ((s, serialized) for s in unevaluated),
                    )
                ),
                None,
            )
    except ImportError as exc:
        return None, exc
    except (PicklingError, AttributeError, TypeError, OSError) as exc:
        return None, exc


def _evaluate_candidates_batch(
    candidates: list[np.ndarray],
    evaluated_count: int,
    evaluator: Callable,
    cache: EvaluationCache | None,
    n_workers: int,
) -> list[float]:
    """Evaluate candidates not yet evaluated (optionally in parallel).

    When *n_workers* > 1 and the cache is disabled, the following strategy is
    applied to bypass the GIL and achieve true multi-core speedup:

    1. **Standard ProcessPoolExecutor**: used when the evaluator is picklable
       (module-level functions, classes). Bypasses the GIL entirely.
    2. **cloudpickle ProcessPoolExecutor**: if the evaluator is a closure or
       lambda (not picklable with ``pickle``), a second attempt serialises it
       with ``cloudpickle`` (optional dep: ``pip install "givp[parallel]"``).
       This enables process-based parallelism for *any* Python callable.
    3. **ThreadPoolExecutor fallback**: used only when both serialisation
       strategies fail (e.g. cloudpickle is not installed).  Thread-based
       execution still benefits objectives that release the GIL (NumPy-heavy,
       Cython, Numba-compiled functions).

    When the cache is enabled, always uses ``ThreadPoolExecutor`` so that
    the in-process cache is shared across workers without IPC overhead.
    """
    unevaluated = candidates[evaluated_count:]
    if n_workers <= 1 or len(unevaluated) <= 1:
        return [_evaluate_with_cache(sol, evaluator, cache) for sol in unevaluated]

    if cache is None:
        process_results, process_exc = _try_standard_process_pool(
            unevaluated,
            evaluator,
            n_workers,
        )
        if process_results is not None:
            return process_results

        cloudpickle_results, cloudpickle_exc = _try_cloudpickle_process_pool(
            unevaluated,
            evaluator,
            n_workers,
        )
        if cloudpickle_results is not None:
            return cloudpickle_results

        if isinstance(cloudpickle_exc, ImportError):
            _log.info(
                "Objective is not picklable with standard pickle (%s). "
                "Install cloudpickle for process-based parallelism with closures: "
                'pip install "givp[parallel]". '
                "Falling back to ThreadPoolExecutor (GIL-limited).",
                process_exc,
            )
        elif cloudpickle_exc is not None:
            _log.info(
                "cloudpickle serialisation failed (%s). "
                "Falling back to ThreadPoolExecutor (GIL-limited).",
                cloudpickle_exc,
            )

    # Thread fallback: shared in-process state; benefits GIL-releasing objectives.
    if cache is not None:
        _log.warning(
            "n_workers=%d requested but use_cache=True forces ThreadPoolExecutor "
            "(GIL-limited). For true multi-core speedup disable the cache: "
            "GIVPConfig(use_cache=False, n_workers=%d).",
            n_workers,
            n_workers,
        )
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        return list(
            executor.map(
                lambda s: _evaluate_with_cache(s, evaluator, cache),
                unevaluated,
            )
        )


# ---------------------------------------------------------------------------
# Public construction entry-point
# ---------------------------------------------------------------------------


def construct_grasp(
    num_vars: int,
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
    """Build an initial solution via randomised Latin-style sampling and RCL selection.

    Generates ``num_candidates_per_step`` random solutions within the bounds,
    evaluates them all, and selects the best one via the Restricted Candidate
    List (RCL).  More efficient than coordinate-by-coordinate greedy
    construction for continuous and mixed-variable spaces.

    Args:
        num_vars: Number of decision variables.
        lower_arr: Lower bounds vector.
        upper_arr: Upper bounds vector.
        evaluator: Objective function to minimise.
        initial_guess: Optional warm-start vector; seeded as one candidate.
        alpha: RCL randomisation parameter (0 = greedy, 1 = uniform random).
        seed: Optional RNG seed.
        num_candidates_per_step: Number of candidate solutions to generate.
        cache: Optional evaluation cache.
        n_workers: Processes (or threads) used for parallel candidate evaluation.
            See :func:`_evaluate_candidates_batch` for the parallelism strategy.

    Returns:
        Best candidate solution array found.
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
        _log.warning("All candidate costs are non-finite; returning first candidate.")
        return candidates[0]
    return candidates[chosen]


# ---------------------------------------------------------------------------
# Adaptive alpha
# ---------------------------------------------------------------------------


def get_current_alpha(iteration: int, config) -> float:
    """Return the adaptive alpha value based on iteration progress."""
    if config.adaptive_alpha:
        progress = iteration / max(1, config.max_iterations - 1)
        current_alpha = (
            config.alpha_min + (config.alpha_max - config.alpha_min) * progress
        )
        return float(np.clip(current_alpha, config.alpha_min, config.alpha_max))
    return float(config.alpha)
