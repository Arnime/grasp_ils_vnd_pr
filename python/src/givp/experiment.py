# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Experimental utilities for reproducible multi-seed benchmarking.

Provides :func:`seed_sweep` to run an objective function over a set of seeds
and return per-seed metrics, and :func:`sweep_summary` to aggregate them into
``mean ± std`` statistics suitable for academic reporting.

Quick start::

    from givp import GIVPConfig, givp
    from givp.experiment import seed_sweep, sweep_summary
    from givp.benchmarks import sphere

    bounds = [(-5.12, 5.12)] * 10
    df = seed_sweep(sphere, bounds, seeds=30)
    print(sweep_summary(df))

Optional dependency: install ``pandas`` to get a ``DataFrame`` back.
Without pandas, a plain ``list[dict]`` is returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from givp.api import givp
from givp.config import GIVPConfig

BoundsLike = Sequence[tuple[float, float]] | tuple[Sequence[float], Sequence[float]]


def seed_sweep(
    func: Callable[[np.ndarray], float],
    bounds: BoundsLike,
    seeds: int | Sequence[int] = 30,
    *,
    config: GIVPConfig | None = None,
    direction: str = "minimize",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run the optimizer for multiple independent seeds and collect metrics.

    Each seed produces one independent run.  Results are collected into a list
    of dicts (one per seed) with the following keys:

    - ``seed``    — the integer seed used for this run
    - ``fun``     — best objective value found (in the user's original sign)
    - ``nit``     — number of GRASP outer iterations executed
    - ``nfev``    — number of objective evaluations
    - ``time_s``  — wall-clock seconds for this run
    - ``success`` — whether the run terminated successfully
    - ``message`` — termination message

    The return value is always a `list[dict]` with one row per seed.
    (This is internally constructed from a DataFrame for convenience when pandas is available.)

    Args:
        func: Objective function ``f(x: np.ndarray) -> float``.
        bounds: Variable bounds — list of ``(low, high)`` pairs, or a
            ``(lower_vec, upper_vec)`` tuple (SciPy style).
        seeds: Either an integer ``N`` (uses seeds ``0 … N-1``) or an explicit
            sequence of seed integers.
        config: Algorithm configuration.  Defaults to :class:`GIVPConfig` with
            sensible defaults.  The ``seed`` field is overridden per run.
        direction: ``"minimize"`` (default) or ``"maximize"``.
        verbose: Pass ``True`` to enable per-iteration logging for every run.

    Returns:
        `list[dict]` with one row per seed, containing the metrics above.
    """
    seed_list: Sequence[int] = (
        list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    )

    cfg = config if config is not None else GIVPConfig()
    rows: list[dict[str, Any]] = []

    for s in seed_list:
        t0 = time.monotonic()
        result = givp(
            func,
            bounds,
            direction=direction,
            config=cfg,
            seed=s,
            verbose=verbose,
        )
        elapsed = time.monotonic() - t0
        rows.append(
            {
                "seed": s,
                "fun": result.fun,
                "nit": result.nit,
                "nfev": result.nfev,
                "time_s": elapsed,
                "success": result.success,
                "message": result.message,
            }
        )

    try:
        import pandas as pd  # type: ignore[import-not-found,import-untyped]

        return pd.DataFrame(rows).to_dict(orient="records")  # type: ignore[no-any-return]
    except ImportError:
        return rows


def sweep_summary(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate seed-sweep results into ``mean ± std`` statistics.

    Args:
        results: Output of :func:`seed_sweep` — list of per-seed dicts
            (or a ``pd.DataFrame``).

    Returns:
        A dict mapping each numeric metric (``fun``, ``nit``, ``nfev``,
        ``time_s``) to a sub-dict ``{"mean": …, "std": …, "min": …, "max": …}``.

    Example output::

        {
            "fun":    {"mean": 1.23e-5, "std": 4.5e-6, "min": 3.2e-7, "max": 2.1e-4},
            "nit":    {"mean": 87.3, "std": 12.1, "min": 60, "max": 100},
            "nfev":   {"mean": 43100.0, "std": 6200.0, "min": 29000, "max": 52000},
            "time_s": {"mean": 0.45, "std": 0.08, "min": 0.33, "max": 0.61},
        }
    """
    # Accept both plain list[dict] and pd.DataFrame
    try:
        import pandas as pd  # type: ignore[import-not-found,import-untyped]

        if isinstance(results, pd.DataFrame):
            rows = [
                {str(key): value for key, value in record.items()}
                for record in results.to_dict("records")
            ]
        else:
            rows = list(results)
    except ImportError:
        rows = list(results)

    metrics = ("fun", "nit", "nfev", "time_s")
    summary: dict[str, dict[str, float]] = {}
    for key in metrics:
        vals = np.array([float(r[key]) for r in rows], dtype=float)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


__all__ = ["seed_sweep", "sweep_summary"]
