# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""
givp — GRASP-ILS-VND with Path Relinking optimizer.

Public API:
    givp(func, bounds, direction='minimize', ...) -> OptimizeResult
    GIVPOptimizer(...).run() -> OptimizeResult
    GIVPConfig          (algorithm hyper-parameters)
    OptimizeResult             (result dataclass, scipy-like)

The algorithm is direction-agnostic: pass ``direction='minimize'`` (default) or
``direction='maximize'`` and the library will internally normalize the objective
while always returning the value in the user's original sign.
"""

from givp.api import GIVPOptimizer, givp
from givp.config import GIVPConfig
from givp.exceptions import (
    EmptyPoolError,
    EvaluatorError,
    GivpError,
    InvalidBoundsError,
    InvalidConfigError,
    InvalidInitialGuessError,
)
from givp.result import OptimizeResult, TerminationReason

__all__ = [
    "EmptyPoolError",
    "EvaluatorError",
    "GIVPConfig",
    "GIVPOptimizer",
    "GivpError",
    "InvalidBoundsError",
    "InvalidConfigError",
    "InvalidInitialGuessError",
    "OptimizeResult",
    "TerminationReason",
    "givp",
]

__version__ = "0.5.1"
