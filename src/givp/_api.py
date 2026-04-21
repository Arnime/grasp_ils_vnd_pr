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

from givp import _core
from givp._config import GraspIlsVndConfig
from givp._result import OptimizeResult

BoundsLike = Sequence[tuple[float, float]] | tuple[Sequence[float], Sequence[float]]


def _normalize_bounds(bounds: BoundsLike, num_vars: int | None) -> tuple[list[float], list[float], int]:
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
        arr_pairs = [tuple(b) for b in bounds]  # type: ignore[arg-type]
        lower = [float(lo) for lo, _ in arr_pairs]
        upper = [float(hi) for _, hi in arr_pairs]

    n = len(lower)
    if num_vars is not None and n != num_vars:
        raise ValueError(f"bounds length ({n}) does not match num_vars ({num_vars})")
    return lower, upper, n


def _wrap_objective(func: Callable[[np.ndarray], float], direction: str, counter: list[int]) -> Callable[[np.ndarray], float]:
    """Wrap user objective so the core always sees a *minimization* problem.

    Increments ``counter[0]`` on every call to track ``nfev``. Non-finite values
    are coerced to ``+inf`` so the algorithm always sees a comparable cost.
    """
    if direction not in ("minimize", "maximize"):
        raise ValueError("direction must be 'minimize' or 'maximize'")

    sign = -1.0 if direction == "maximize" else 1.0

    def wrapped(x: np.ndarray) -> float:
        counter[0] += 1
        try:
            value = float(func(np.asarray(x, dtype=float)))
        except (ValueError, RuntimeError, FloatingPointError):
            return float("inf")
        if not np.isfinite(value):
            return float("inf")
        return sign * value

    return wrapped


def grasp_ils_vnd_pr(
    func: Callable[[np.ndarray], float],
    bounds: BoundsLike,
    *,
    num_vars: int | None = None,
    direction: str = "minimize",
    config: GraspIlsVndConfig | None = None,
    initial_guess: Sequence[float] | None = None,
    iteration_callback: Callable[[int, float, np.ndarray], None] | None = None,
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
        direction: ``'minimize'`` (default) or ``'maximize'``.
        config: Algorithm hyper-parameters. ``GraspIlsVndConfig()`` is used
            when omitted. ``config.direction`` is ignored if ``direction`` is
            also passed explicitly here (the explicit kwarg wins).
        initial_guess: Optional warm-start vector, evaluated and inserted in
            the elite pool before the first iteration.
        iteration_callback: Optional callable invoked once per outer iteration
            with ``(iteration, best_cost_in_core_sign, best_solution)``.
        verbose: If True, prints progress information to stdout.

    Returns:
        OptimizeResult: Dataclass with ``x`` (best solution), ``fun`` (best
        objective value in the **user's original sign**) and metadata.
    """
    cfg = config or GraspIlsVndConfig()
    if direction is not None:
        cfg = GraspIlsVndConfig(
            **{**cfg.__dict__, "direction": direction}
        )

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
        func: Callable[[np.ndarray], float],
        bounds: BoundsLike,
        *,
        num_vars: int | None = None,
        direction: str = "minimize",
        config: GraspIlsVndConfig | None = None,
        initial_guess: Sequence[float] | None = None,
        iteration_callback: Callable[[int, float, np.ndarray], None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.func = func
        self.bounds = bounds
        self.num_vars = num_vars
        self.direction = direction
        self.config = config or GraspIlsVndConfig()
        self.initial_guess = initial_guess
        self.iteration_callback = iteration_callback
        self.verbose = verbose

        self.best_x: np.ndarray | None = None
        self.best_fun: float = float("-inf") if direction == "maximize" else float("inf")
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
            direction=self.direction,
            config=self.config,
            initial_guess=self.initial_guess,
            iteration_callback=self.iteration_callback,
            verbose=self.verbose,
        )
        self.history.append(result)
        if self.best_x is None or self._is_better(result.fun):
            self.best_x = result.x
            self.best_fun = result.fun
        return result
