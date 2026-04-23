# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""
Public API for the ``givp`` library.

Exposes a SciPy-style functional entry point ``grasp_ils_vnd_pr`` and an
sklearn-style class ``GraspOptimizer``. Both wrap the internal minimizer in
``givp._core`` and add direction-agnostic objective handling
(``direction='minimize' | 'maximize'``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from givp import _core
from givp._config import GraspIlsVndConfig
from givp._core._helpers import _set_seed
from givp._result import OptimizeResult

BoundsLike = Sequence[tuple[float, float]] | tuple[Sequence[float], Sequence[float]]
ObjectiveFn = Callable[[NDArray[np.float64]], float]
IterationCallback = Callable[[int, float, NDArray[np.float64]], None]


def _normalize_bounds(
    bounds: BoundsLike, num_vars: int | None
) -> tuple[list[float], list[float], int]:
    """Accept SciPy-style list of (low, high) pairs or a (lower, upper) tuple."""
    if bounds is None:
        raise ValueError("bounds must be provided")

    arr_pairs: list[tuple[float, float]]
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and isinstance(bounds[0], Iterable)
        and not isinstance(bounds[0], (str, bytes))
        and isinstance(bounds[1], Iterable)
        and not isinstance(bounds[1], (str, bytes))
        and len(list(bounds[0])) == len(list(bounds[1]))
        and (num_vars is None or len(list(bounds[0])) == num_vars)
    ):
        lower = [float(v) for v in bounds[0]]
        upper = [float(v) for v in bounds[1]]
    else:
        arr_pairs = [(float(b[0]), float(b[1])) for b in bounds]  # type: ignore[index]
        lower = [lo for lo, _ in arr_pairs]
        upper = [hi for _, hi in arr_pairs]

    n = len(lower)
    if num_vars is not None and n != num_vars:
        raise ValueError(f"bounds length ({n}) does not match num_vars ({num_vars})")
    return lower, upper, n


def _wrap_objective(
    func: ObjectiveFn, direction: str, counter: list[int]
) -> ObjectiveFn:
    """Wrap user objective so the core always sees a *minimization* problem.

    Increments ``counter[0]`` on every call to track ``nfev``. Non-finite values
    are coerced to ``+inf`` so the algorithm always sees a comparable cost.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError("direction must be 'minimize' or 'maximize'")

    sign = -1.0 if direction == "maximize" else 1.0

    def wrapped(x: NDArray[np.float64]) -> float:
        counter[0] += 1
        try:
            value = float(func(np.asarray(x, dtype=float)))
        except (ValueError, RuntimeError, FloatingPointError):
            return float("inf")
        if not np.isfinite(value):
            return float("inf")
        return sign * value

    return wrapped


def _resolve_direction(
    minimize: bool | None, direction: str | None, default: str = "minimize"
) -> str:
    """Reconcile the boolean ``minimize`` flag with the string ``direction``.

    Rules:
        * Both ``None`` -> ``default``.
        * Only ``minimize`` set -> map to direction.
        * Only ``direction`` set -> validate and return it.
        * Both set -> values must agree, otherwise ``ValueError``.
    """
    if direction is not None and direction not in ("minimize", "maximize"):
        raise ValueError(
            f"direction must be 'minimize' or 'maximize', got {direction!r}"
        )
    if minimize is None and direction is None:
        return default
    if minimize is None:
        return direction  # type: ignore[return-value]
    derived = "minimize" if minimize else "maximize"
    if direction is not None and direction != derived:
        raise ValueError(
            "`minimize` and `direction` disagree: "
            f"minimize={minimize} implies '{derived}', got direction={direction!r}"
        )
    return derived


def grasp_ils_vnd_pr(
    func: ObjectiveFn,
    bounds: BoundsLike,
    *,
    num_vars: int | None = None,
    minimize: bool | None = None,
    direction: str | None = None,
    config: GraspIlsVndConfig | None = None,
    initial_guess: Sequence[float] | None = None,
    iteration_callback: IterationCallback | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> OptimizeResult:
    """
    Minimize (or maximize) a scalar function with GRASP-ILS-VND-PR.

    Args:
        func: Objective callable mapping a 1-D ``np.ndarray`` to a scalar.
        bounds: Either a sequence of ``(low, high)`` pairs (SciPy style) or a
            ``(lower, upper)`` tuple of two equally-sized sequences.
        num_vars: Optional explicit number of variables. Inferred from
            ``bounds`` when omitted.
        minimize: Boolean flag for the optimization sense. ``True`` minimizes,
            ``False`` maximizes. Preferred over ``direction`` for new code.
        direction: ``'minimize'`` or ``'maximize'`` (SciPy/Optuna style). Kept
            for backward compatibility. Ignored when ``minimize`` is given.
            Defaults to ``'minimize'`` when neither flag is supplied.
        config: Algorithm hyper-parameters. ``GraspIlsVndConfig()`` is used
            when omitted. Any sense field on ``config`` is overridden by the
            explicit ``minimize``/``direction`` kwargs when provided.
        initial_guess: Optional warm-start vector, evaluated and inserted in
            the elite pool before the first iteration.
        iteration_callback: Optional callable invoked once per outer iteration
            with ``(iteration, best_cost_in_core_sign, best_solution)``.
        seed: Optional integer seed for full reproducibility. When given,
            every internal RNG is derived deterministically from this seed,
            so two calls with the same inputs return the same result.
        verbose: If True, prints progress information to stdout.

    Returns:
        OptimizeResult: Dataclass with ``x`` (best solution), ``fun`` (best
        objective value in the **user's original sign**) and metadata.

    Raises:
        ValueError: If both ``minimize`` and ``direction`` are passed with
            conflicting values, or if ``direction`` is not one of
            ``'minimize'`` / ``'maximize'``.
    """
    resolved_direction = _resolve_direction(minimize, direction)
    cfg = config or GraspIlsVndConfig()
    cfg = GraspIlsVndConfig(
        **{**cfg.__dict__, "minimize": resolved_direction == "minimize"}
    )

    # Pin the master RNG when a seed was supplied so all internal helpers
    # derive deterministic child seeds. ``None`` restores the default
    # non-deterministic behaviour.
    _set_seed(seed)

    lower, upper, n = _normalize_bounds(bounds, num_vars)

    # Default to a fully continuous problem when the user did not specify the
    # split. Callers with mixed continuous/integer models must set
    # ``config.integer_split`` explicitly (or use the legacy adapter).
    if cfg.integer_split is None:
        cfg = GraspIlsVndConfig(**{**cfg.__dict__, "integer_split": n})
    nfev_counter = [0]
    wrapped = _wrap_objective(func, cfg.direction, nfev_counter)

    sol_list, core_value = _core.grasp_ils_vnd(
        wrapped,
        n,
        cfg.as_core_config(),
        verbose=verbose,
        iteration_callback=iteration_callback,
        lower=lower,
        upper=upper,
        initial_guess=list(initial_guess) if initial_guess is not None else None,
    )

    x = np.asarray(sol_list, dtype=float)
    sign = -1.0 if cfg.direction == "maximize" else 1.0
    fun_value = sign * float(core_value)
    success = np.isfinite(fun_value)

    return OptimizeResult(
        x=x,
        fun=fun_value,
        nit=cfg.max_iterations,
        nfev=nfev_counter[0],
        success=success,
        message="ok" if success else "no finite solution found",
        direction=cfg.direction,
    )


class GraspOptimizer:
    """
    Object-oriented wrapper around :func:`grasp_ils_vnd_pr`.

    Holds configuration and bounds, exposes a ``run()`` method that returns an
    :class:`OptimizeResult`. The instance also caches the best solution across
    repeated ``run()`` calls (useful for multi-start strategies).
    """

    def __init__(
        self,
        func: ObjectiveFn,
        bounds: BoundsLike,
        *,
        num_vars: int | None = None,
        minimize: bool | None = None,
        direction: str | None = None,
        config: GraspIlsVndConfig | None = None,
        initial_guess: Sequence[float] | None = None,
        iteration_callback: IterationCallback | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.func = func
        self.bounds = bounds
        self.num_vars = num_vars
        self.direction = _resolve_direction(minimize, direction)
        self.minimize = self.direction == "minimize"
        self.config = config or GraspIlsVndConfig()
        self.initial_guess = initial_guess
        self.iteration_callback = iteration_callback
        self.seed = seed
        self.verbose = verbose

        self.best_x: NDArray[np.float64] | None = None
        self.best_fun: float = (
            float("-inf") if self.direction == "maximize" else float("inf")
        )
        self.history: list[OptimizeResult] = []

    def _is_better(self, candidate: float) -> bool:
        if self.direction == "maximize":
            return candidate > self.best_fun
        return candidate < self.best_fun

    def run(self) -> OptimizeResult:
        """Execute one optimization round and update the historical best."""
        result = grasp_ils_vnd_pr(
            self.func,
            self.bounds,
            num_vars=self.num_vars,
            minimize=self.minimize,
            config=self.config,
            initial_guess=self.initial_guess,
            iteration_callback=self.iteration_callback,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.history.append(result)
        if self.best_x is None or self._is_better(result.fun):
            self.best_x = result.x
            self.best_fun = result.fun
        return result
