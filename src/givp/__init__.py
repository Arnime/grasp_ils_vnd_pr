# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""
givp — GRASP-ILS-VND with Path Relinking optimizer.

Public API:
    grasp_ils_vnd_pr(func, bounds, direction='minimize', ...) -> OptimizeResult
    GraspOptimizer(...).run() -> OptimizeResult
    GraspIlsVndConfig          (algorithm hyper-parameters)
    OptimizeResult             (result dataclass, scipy-like)

The algorithm is direction-agnostic: pass ``direction='minimize'`` (default) or
``direction='maximize'`` and the library will internally normalize the objective
while always returning the value in the user's original sign.
"""

from givp._api import GraspOptimizer, grasp_ils_vnd_pr
from givp._config import GraspIlsVndConfig
from givp._exceptions import (
    EmptyPoolError,
    EvaluatorError,
    GivpError,
    InvalidBoundsError,
    InvalidConfigError,
    InvalidInitialGuessError,
)
from givp._result import OptimizeResult

__all__ = [
    "EmptyPoolError",
    "EvaluatorError",
    "GivpError",
    "GraspIlsVndConfig",
    "GraspOptimizer",
    "InvalidBoundsError",
    "InvalidConfigError",
    "InvalidInitialGuessError",
    "OptimizeResult",
    "grasp_ils_vnd_pr",
]

__version__ = "0.3.0"
