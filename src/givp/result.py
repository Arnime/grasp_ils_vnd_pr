# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Result container returned by the public optimizer API."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class TerminationReason(str, Enum):
    """Closed set of termination reasons — safe to pass to LLM agents."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    EARLY_STOP = "early_stop"
    NO_FEASIBLE = "no_feasible"
    UNKNOWN = "unknown"

    @classmethod
    def from_message(cls, message: str) -> TerminationReason:
        """Map a free-form termination message to the nearest enum value."""
        lower = message.lower()
        if "converge" in lower:
            return cls.CONVERGED
        if "time" in lower:
            return cls.TIME_LIMIT
        if "early" in lower or "threshold" in lower:
            return cls.EARLY_STOP
        if "feasible" in lower or "no solution" in lower:
            return cls.NO_FEASIBLE
        if "iteration" in lower or "max" in lower:
            return cls.MAX_ITERATIONS
        return cls.UNKNOWN


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

    x: NDArray[np.float64]
    fun: float
    nit: int = 0
    nfev: int = 0
    success: bool = True
    message: str = ""
    direction: str = "minimize"
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        # Allows tuple unpacking ``x, fun = result`` for compatibility with
        # callers expecting the legacy ``(solution, value)`` 2-tuple.
        yield self.x
        yield self.fun

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with typed, schema-validated fields.

        All values are Python primitives — no numpy types. The ``termination``
        field is a ``TerminationReason`` enum value (closed set), which prevents
        prompt-injection via free-form ``message`` strings when passed to LLM agents.
        """
        return {
            "x": self.x.tolist(),
            "fun": float(self.fun),
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "success": bool(self.success),
            "termination": TerminationReason.from_message(self.message).value,
            "direction": self.direction,
        }
