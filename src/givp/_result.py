"""Result container returned by the public optimizer API."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizeResult:
    """
    Container for optimization output, modeled after ``scipy.optimize.OptimizeResult``.

    Attributes:
        x: Best solution vector found, in the user's variable space.
        fun: Objective value at ``x``, in the user's original sign
            (i.e., already restored when ``direction='maximize'``).
        nit: Number of GRASP outer iterations executed.
        nfev: Number of objective function evaluations.
        success: Whether at least one feasible solution was produced.
        message: Human-readable termination reason.
        direction: ``'minimize'`` or ``'maximize'``.
        meta: Additional algorithm-specific information (cache stats, etc.).
    """

    x: np.ndarray
    fun: float
    nit: int = 0
    nfev: int = 0
    success: bool = True
    message: str = ""
    direction: str = "minimize"
    meta: dict = field(default_factory=dict)

    def __iter__(self):
        # Allows tuple unpacking ``x, fun = result`` for compatibility with
        # callers expecting the legacy ``(solution, value)`` 2-tuple.
        yield self.x
        yield self.fun
