# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Legacy discrete-packing helpers from the original SOG2 problem formulation.

These functions implement the GRASP construction for the legacy discrete
knapsack-with-dependencies problem (SOG2).  They are retained for backward
compatibility only.

For general RCL selection (continuous *and* discrete problems) use the public
functions in :mod:`givp.core.grasp`:

- :func:`givp.core.grasp.select_rcl` — ratio-based RCL (discrete/mixed,
  higher-is-better objectives).
- :func:`givp.core.grasp.construct_grasp` — full GRASP construction for
  continuous and mixed-integer spaces.

.. deprecated::
    :func:`evaluate_candidates` will be removed in a future version.
"""

from __future__ import annotations

import warnings

import numpy as np


def evaluate_candidates(
    available: np.ndarray,
    deps_active: np.ndarray,
    current_cost: int,
    deps_matrix: np.ndarray,
    deps_len: np.ndarray,
    c_arr: np.ndarray,
    a_arr: np.ndarray,
    b: int,
):
    """Evaluate candidates based on dependency information and incremental costs.

    .. deprecated::
        This helper is retained for compatibility with legacy discrete-packing
        use-cases (SOG2 knapsack formulation).  Use
        :func:`givp.core.grasp.construct_grasp` for continuous and
        mixed-integer optimisation.

    Args:
        available: Indices of available packages.
        deps_active: Boolean array indicating already-active dependencies.
        current_cost: Current total cost (legacy semantics).
        deps_matrix: Dependency matrix per package.
        deps_len: Number of dependencies per package.
        c_arr: Benefit array per package.
        a_arr: Dependency cost array.
        b: Budget limit.

    Returns:
        Tuple ``(ratios, incremental_costs, valid_mask)``:
            ratios: Benefit/cost ratio per candidate.
            incremental_costs: Incremental cost per candidate.
            valid_mask: Boolean mask of feasible candidates.
    """
    warnings.warn(
        "evaluate_candidates is a legacy discrete-packing helper and will be removed "
        "in a future version. Use construct_grasp for continuous/mixed-integer optimization.",
        DeprecationWarning,
        stacklevel=2,
    )
    n_available = len(available)
    ratios = np.full(n_available, -np.inf, dtype=np.float32)
    inc_costs = np.zeros(n_available, dtype=np.int32)
    valid = np.zeros(n_available, dtype=np.bool_)
    for idx, i in enumerate(available):
        n_deps = deps_len[i]
        incremental_cost = 0
        if n_deps > 0:
            pkg_deps = deps_matrix[i, :n_deps]
            new_deps_mask = ~deps_active[pkg_deps]
            new_deps = pkg_deps[new_deps_mask]
            if len(new_deps) > 0:
                incremental_cost = a_arr[new_deps].sum()
        if current_cost + incremental_cost <= b:
            valid[idx] = True
            inc_costs[idx] = incremental_cost
            ratios[idx] = c_arr[i] / incremental_cost if incremental_cost > 0 else 1e9
    return ratios, inc_costs, valid
