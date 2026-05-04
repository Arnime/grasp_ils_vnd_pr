# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Compatibility package `givp.core` re-exporting the internal core submodules.

This module provides a compatibility layer: it ensures callers can import
either the submodules (``import givp.core.path_relinking as _pr_module``)
or the convenience callables/symbols (``from givp.core import path_relinking``).

Design goals:
- Import submodules as modules first so direct `import givp.core.<mod>`
  yields a real module object.
- Then import and re-export the canonical callables/symbols from those
  submodules so ``from givp.core import ...`` returns the expected API.

Private symbols (names beginning with ``_``) are **not** re-exported here.
Import them directly from their defining submodule when needed, e.g.::

    from givp.core.grasp import _validate_bounds_and_initial
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from givp.config import GIVPConfig

# `import givp.core.path_relinking as _pr_module` and get the module object.
from givp.core import grasp, ils, impl, vnd

# Re-export key symbols from the canonical submodules (public API surface).
from givp.core.cache import EvaluationCache
from givp.core.convergence import ConvergenceMonitor
from givp.core.elite import ElitePool
from givp.core.grasp import (
    construct_grasp,
    get_current_alpha,
    select_rcl,
)
from givp.core.ils import ils_search, perturb_solution_numpy
from givp.core.impl import grasp_ils_vnd
from givp.core.pr import bidirectional_path_relinking, path_relinking
from givp.core.vnd import (
    local_search_vnd,
    local_search_vnd_adaptive,
)
from givp.legacy.sog2 import evaluate_candidates

if TYPE_CHECKING:
    # Declare for static analyzers; runtime access remains lazy via __getattr__.
    import givp.core.legacy_sog2 as legacy_sog2

__all__ = [
    # Public classes
    "ConvergenceMonitor",
    "ElitePool",
    "EvaluationCache",
    "GIVPConfig",
    # Public functions
    "bidirectional_path_relinking",
    "construct_grasp",
    "evaluate_candidates",
    "get_current_alpha",
    # Sub-module references (for ``import givp.core.grasp`` style access)
    "grasp",
    "grasp_ils_vnd",
    "ils",
    "ils_search",
    "impl",
    "legacy_sog2",
    "local_search_vnd",
    "local_search_vnd_adaptive",
    "path_relinking",
    "perturb_solution_numpy",
    "select_rcl",
    "vnd",
]


def __getattr__(name: str) -> object:
    """Provide lazy compatibility import for the deprecated legacy module."""
    if name == "legacy_sog2":
        return importlib.import_module("givp.core.legacy_sog2")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
