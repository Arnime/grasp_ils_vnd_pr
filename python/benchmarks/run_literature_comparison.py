# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Reproducible multi-run benchmark experiment for GIVP vs literature baselines.

Runs GIVP-full, GRASP-only, and (optionally) scipy baselines — Differential
Evolution and Dual Annealing — on six standard continuous optimisation
benchmark functions widely used in the metaheuristic literature.

Each (algorithm, function, seed) triple is an independent run; results are
written to a JSON file suitable for analysis by ``generate_report.py``.

Usage
-----
    # Default: 30 runs × 10-D × GIVP-full + GRASP-only on all 6 functions
    python run_literature_comparison.py

    # Wider comparison including scipy baselines (requires: pip install scipy)
    python run_literature_comparison.py --algorithms GIVP-full GRASP-only DE SA

    # Higher dimensionality, fewer runs, custom output
    python run_literature_comparison.py --dims 30 --n-runs 10 --output results_30d.json

    # Capture per-iteration convergence traces (GIVP algorithms only)
    python run_literature_comparison.py --traces --n-runs 5

References
----------
- De Jong, K.A. (1975). Analysis of the Behaviour of a Class of Genetic
  Adaptive Systems. PhD thesis, University of Michigan.
- Rosenbrock, H.H. (1960). An Automatic Method for Finding the Greatest or
  Least Value of a Function. The Computer Journal, 3(3), 175-184.
- Rastrigin, L.A. (1974). Systems of Extremal Control. Nauka, Moscow.
- Ackley, D.H. (1987). A Connectionist Machine for Genetic Hillclimbing.
  Kluwer Academic Publishers.
- Griewank, A.O. (1981). Generalized descent for global optimization.
  Journal of Optimization Theory and Applications, 34(1), 11-39.
- Schwefel, H.P. (1981). Numerical Optimization of Computer Models. Wiley.
- Feo, T.A. & Resende, M.G.C. (1995). Greedy randomized adaptive search
  procedures. Journal of Global Optimization, 6, 109-133.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import scipy.optimize as _scipy_optimize  # type: ignore[import-untyped]

    _SCIPY_OPTIMIZE_OK = True
except ImportError:
    _scipy_optimize = None  # type: ignore[assignment]
    _SCIPY_OPTIMIZE_OK = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Verify givp installation
# ---------------------------------------------------------------------------

try:
    import givp as _givp_mod
    import givp.benchmarks as bm
    GIVPConfig = _givp_mod.GIVPConfig
    givp = _givp_mod.givp
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"[error] givp not installed: {exc}\n"
        "  From the python/ directory run:  pip install -e .[dev]"
    )

_GIVP_VERSION: str = getattr(_givp_mod, "__version__", "unknown")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark problem registry
# ---------------------------------------------------------------------------
#
# Each entry defines:
#   func            — callable(x: NDArray) -> float; global min = optimum
#   bounds_factory  — (n: int) -> list[tuple[float, float]]
#   optimum         — known global minimum value
#   reference       — canonical citation used in the paper/report
#
PROBLEM_REGISTRY: dict[str, dict] = {
    "Sphere": {
        "func": bm.sphere,
        "bounds_factory": lambda n: [(-5.12, 5.12)] * n,
        "optimum": 0.0,
        "reference": "De Jong (1975)",
    },
    "Rosenbrock": {
        "func": bm.rosenbrock,
        "bounds_factory": lambda n: [(-5.0, 10.0)] * n,
        "optimum": 0.0,
        "reference": "Rosenbrock (1960)",
    },
    "Rastrigin": {
        "func": bm.rastrigin,
        "bounds_factory": lambda n: [(-5.12, 5.12)] * n,
        "optimum": 0.0,
        "reference": "Rastrigin (1974); Mühlenbein et al. (1991)",
    },
    "Ackley": {
        "func": bm.ackley,
        "bounds_factory": lambda n: [(-32.768, 32.768)] * n,
        "optimum": 0.0,
        "reference": "Ackley (1987)",
    },
    "Griewank": {
        "func": bm.griewank,
        "bounds_factory": lambda n: [(-600.0, 600.0)] * n,
        "optimum": 0.0,
        "reference": "Griewank (1981)",
    },
    "Schwefel": {
        "func": bm.schwefel,
        "bounds_factory": lambda n: [(-500.0, 500.0)] * n,
        "optimum": 0.0,
        "reference": "Schwefel (1981)",
    },
}

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGO_DESCRIPTIONS: dict[str, str] = {
    "GIVP-full": "GRASP-ILS-VND-PR -- full hybrid pipeline (this work)",
    "GIVP-tuned": "GRASP-ILS-VND-PR -- Optuna-tuned hyperparameters (tune_hyperparams.py)",
    "GRASP-only": "GRASP-only baseline (Feo & Resende 1995)",
    "DE": "Differential Evolution -- scipy.optimize (Storn & Price 1997)",
    "SA": "Dual Annealing -- scipy.optimize (Xiang et al. 1997)",
}

def _config_givp_full(max_iter: int, time_limit: float) -> GIVPConfig:
    """Full GIVP pipeline: adaptive alpha, ILS, VND, elite pool, path relinking."""
    return GIVPConfig(
        max_iterations=max_iter,
        alpha=0.12,
        adaptive_alpha=True,
        alpha_min=0.08,
        alpha_max=0.18,
        vnd_iterations=200,
        ils_iterations=10,
        perturbation_strength=4,
        use_elite_pool=True,
        elite_size=7,
        path_relink_frequency=8,
        use_cache=True,
        cache_size=10_000,
        early_stop_threshold=80,
        use_convergence_monitor=True,
        time_limit=time_limit,
    )


def _config_grasp_only(max_iter: int, time_limit: float) -> GIVPConfig:
    """GRASP-only baseline (Feo & Resende, 1995).

    Disables ILS, VND depth, elite pool, convergence monitor and path
    relinking to reproduce the plain GRASP construction + trivial descent.
    """
    return GIVPConfig(
        max_iterations=max_iter,
        alpha=0.12,
        adaptive_alpha=False,
        vnd_iterations=1,
        ils_iterations=1,
        perturbation_strength=0,
        use_elite_pool=False,
        use_convergence_monitor=False,
        use_cache=True,
        cache_size=10_000,
        early_stop_threshold=max_iter,
        time_limit=time_limit,
    )


def _config_givp_tuned(
    base: GIVPConfig,
    max_iter: int,
    time_limit: float,
) -> GIVPConfig:
    """Return a copy of *base* with max_iterations and time_limit overridden."""
    import dataclasses

    return dataclasses.replace(base, max_iterations=max_iter, time_limit=time_limit)


# ---------------------------------------------------------------------------
# Scipy helpers (optional dependency)
# ---------------------------------------------------------------------------


def _scipy_de(
    func,
    bounds: list[tuple[float, float]],
    seed: int,
    max_iter: int,
) -> tuple[float, int, int]:
    """Run scipy.optimize.differential_evolution. Returns (fun, nit, nfev)."""
    if not _SCIPY_OPTIMIZE_OK or _scipy_optimize is None:  # pragma: no cover
        raise RuntimeError("scipy is not installed.  Run:  pip install scipy")
    res = _scipy_optimize.differential_evolution(
        func, bounds, seed=seed, maxiter=max_iter, tol=1e-12, workers=1  # type: ignore[call-arg]
        # scipy stubs unavailable
    )
    return float(res.fun), int(res.nit), int(res.nfev)


def _scipy_sa(
    func,
    bounds: list[tuple[float, float]],
    seed: int,
    max_iter: int,
) -> tuple[float, int, int]:
    """Run scipy.optimize.dual_annealing. Returns (fun, nit, nfev)."""
    if not _SCIPY_OPTIMIZE_OK or _scipy_optimize is None:  # pragma: no cover
        raise RuntimeError("scipy is not installed.  Run:  pip install scipy")
    rng = np.random.default_rng(seed)
    x0 = np.array([lo + rng.random() * (hi - lo) for lo, hi in bounds])
    res = _scipy_optimize.dual_annealing(
        func, bounds, seed=seed, maxiter=max_iter * 100, x0=x0  # type: ignore[call-arg]  # scipy stubs unavailable
    )
    return float(res.fun), int(res.nit), int(res.nfev)


# ----------------------------------------------------------------


def _run_givp(
    cfg,
    func,
    bounds: list[tuple[float, float]],
    seed: int,
    capture_trace: bool,
) -> tuple[float, int, int, list[float] | None]:
    """Execute one GIVP run (full, tuned, or GRASP-only). Returns (fun, nit, nfev, trace)."""
    trace: list[float] | None = None
    if capture_trace:
        history: list[float] = []
        best_ref: list[float] = [float("inf")]

        def _cb(iteration: int, cost: float, solution: NDArray) -> None:
            if cost < best_ref[0]:
                best_ref[0] = cost
            history.append(best_ref[0])

        res = givp(func, bounds, config=cfg, seed=seed, iteration_callback=_cb)
        trace = history
    else:
        res = givp(func, bounds, config=cfg, seed=seed)
    return float(res.fun), int(res.nit), int(res.nfev), trace


def _run_single(
    algo: str,
    func,
    bounds: list[tuple[float, float]],
    seed: int,
    max_iter: int,
    time_limit: float,
    capture_trace: bool,
    givp_tuned_config: GIVPConfig | None = None,
) -> dict:
    """Execute one (algorithm, function, seed) combination.

    Returns a flat dict with keys:
        algorithm, seed, fun, nit, nfev, time_s, trace (list[float] or None)
    """
    trace: list[float] | None = None
    t0 = time.perf_counter()

    if algo in ("GIVP-full", "GRASP-only", "GIVP-tuned"):
        if algo == "GIVP-full":
            cfg = _config_givp_full(max_iter, time_limit)
        elif algo == "GIVP-tuned":
            if givp_tuned_config is None:  # pragma: no cover
                raise RuntimeError(
                    "GIVP-tuned requires --tune-config PATH (output of tune_hyperparams.py).\n"
                    "  python benchmarks/tune_hyperparams.py --output best_config.json"
                )
            cfg = _config_givp_tuned(givp_tuned_config, max_iter, time_limit)
        else:
            cfg = _config_grasp_only(max_iter, time_limit)
        fun, nit, nfev, trace = _run_givp(cfg, func, bounds, seed, capture_trace)

    elif algo == "DE":
        fun, nit, nfev = _scipy_de(func, bounds, seed, max_iter)
    elif algo == "SA":
        fun, nit, nfev = _scipy_sa(func, bounds, seed, max_iter)
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")

    elapsed = time.perf_counter() - t0
    return {
        "algorithm": algo,
        "seed": seed,
        "fun": fun,
        "nit": nit,
        "nfev": nfev,
        "time_s": round(elapsed, 4),
        "trace": trace,
    }


def _build_summary_rows(
    raw: dict[str, list[dict]],
    functions: list[str],
    algorithms: list[str],
) -> list[dict]:
    """Compute per-(function, algorithm) descriptive statistics from raw records."""
    summary: list[dict] = []
    for fn_name in functions:
        for algo in algorithms:
            values = [r["fun"] for r in raw[fn_name] if r["algorithm"] == algo]
            arr = np.asarray(values, dtype=float)
            nfev_arr = np.asarray(
                [r["nfev"] for r in raw[fn_name] if r["algorithm"] == algo],
                dtype=float,
            )
            summary.append(
                {
                    "function": fn_name,
                    "algorithm": algo,
                    "n_runs": len(values),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "best": float(arr.min()),
                    "median": float(np.median(arr)),
                    "worst": float(arr.max()),
                    "nfev_mean": float(nfev_arr.mean()),
                }
            )
    return summary


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def _load_checkpoint(
    checkpoint_path: Path,
    raw: dict[str, list[dict]],
) -> tuple[set[str], int]:
    """Load completed function records from an existing checkpoint file.

    Returns (completed_functions, done_count).
    """
    completed: set[str] = set()
    done = 0
    ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for fn_name, records in ckpt.get("records", {}).items():
        if fn_name in raw and records:
            raw[fn_name] = records
            completed.add(fn_name)
            done += len(records)
    if completed:
        _log.info("[resume] Skipping %s (already in checkpoint)", sorted(completed))
    return completed, done


def _save_checkpoint(
    checkpoint_path: Path,
    raw: dict[str, list[dict]],
    completed_functions: set[str],
    fn_name: str,
    algorithms: list[str],
    functions: list[str],
    dims: int,
    n_runs: int,
    seed_start: int,
    seeds: list[int],
    max_iter: int,
    time_limit: float,
) -> None:
    """Persist a partial checkpoint JSON after completing one function."""
    partial_summary = _build_summary_rows(
        raw, [fn for fn in functions if raw[fn]], algorithms
    )
    partial_payload = {
        "metadata": {
            "givp_version": _GIVP_VERSION,
            "dims": dims,
            "n_runs": n_runs,
            "seed_start": seed_start,
            "seeds": seeds,
            "max_iter": max_iter,
            "time_limit": time_limit,
            "algorithms": algorithms,
            "functions": functions,
            "checkpoint": True,
            "completed_functions": sorted(completed_functions | {fn_name}),
            "problem_references": {
                fn: PROBLEM_REGISTRY[fn]["reference"] for fn in functions
            },
            "algo_descriptions": {a: ALGO_DESCRIPTIONS[a] for a in algorithms},
        },
        "summary": partial_summary,
        "records": raw,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")
    _log.debug("  [checkpoint] saved after %s → %s", fn_name, checkpoint_path)


def _run_function_seeds(
    fn_name: str,
    algorithms: list[str],
    seeds: list[int],
    dims: int,
    max_iter: int,
    time_limit: float,
    capture_traces: bool,
    givp_tuned_config: GIVPConfig | None,
    done_offset: int,
    total: int,
) -> tuple[list[dict], int]:
    """Run all (algo, seed) pairs for a single benchmark function.

    Returns (records, done_count) where done_count starts from *done_offset*.
    """
    spec = PROBLEM_REGISTRY[fn_name]
    bounds = spec["bounds_factory"](dims)
    func = spec["func"]
    records: list[dict] = []
    done = done_offset

    for algo in algorithms:
        for seed in seeds:
            rec = _run_single(
                algo, func, bounds, seed, max_iter, time_limit, capture_traces,
                givp_tuned_config=givp_tuned_config,
            )
            records.append(rec)
            done += 1
            trace_flag = " [+trace]" if rec["trace"] else ""
            _log.debug(
                "  [%4d/%d] %-12s %-12s seed=%3d  fun=%12.4e  nfev=%7d  t=%.2fs%s",
                done, total, fn_name, algo, seed,
                rec["fun"], rec["nfev"], rec["time_s"], trace_flag,
            )
    return records, done


def run_experiment(
    algorithms: list[str],
    functions: list[str],
    dims: int,
    n_runs: int,
    seed_start: int,
    max_iter: int,
    time_limit: float,
    capture_traces: bool,
    givp_tuned_config: GIVPConfig | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict:
    """Run the full experiment matrix and return the result payload.

    Parameters
    ----------
    algorithms:
        Names from ALGO_DESCRIPTIONS to execute.
    functions:
        Names from PROBLEM_REGISTRY to benchmark.
    dims:
        Problem dimensionality (number of decision variables).
    n_runs:
        Number of independent runs per (algorithm, function) pair.
        Must be ≥ 2 for statistical tests; ≥ 30 recommended for publication.
    seed_start:
        First seed value; runs use seeds [seed_start, seed_start + n_runs).
    max_iter:
        Maximum GIVP iterations (or equivalent budget for scipy methods).
    time_limit:
        Per-run wall-clock budget in seconds.
    capture_traces:
        If True, store per-iteration best-value history (GIVP algorithms only).
        Increases output size significantly.
    givp_tuned_config:
        Pre-built GIVPConfig for the GIVP-tuned algorithm.  Required when
        ``"GIVP-tuned"`` is in *algorithms*.
    checkpoint_path:
        If set, the result JSON is written after each function completes,
        allowing ``--resume`` to skip already-finished functions.
    resume:
        If True and *checkpoint_path* exists, load previously completed
        function results and skip those functions.

    Returns
    -------
    dict
        Payload with keys: ``metadata``, ``summary``, ``records``.
    """
    seeds = list(range(seed_start, seed_start + n_runs))
    total = len(algorithms) * len(functions) * n_runs

    # --- resume: load already-computed function results from checkpoint ---
    raw: dict[str, list[dict]] = {fn: [] for fn in functions}
    completed_functions: set[str] = set()
    done = 0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        completed_functions, done = _load_checkpoint(checkpoint_path, raw)

    for fn_name in functions:
        if fn_name in completed_functions:
            continue

        fn_records, done = _run_function_seeds(
            fn_name, algorithms, seeds, dims, max_iter, time_limit,
            capture_traces, givp_tuned_config, done, total,
        )
        raw[fn_name] = fn_records

        # --- checkpoint: persist after each function ---
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path, raw, completed_functions, fn_name,
                algorithms, functions, dims, n_runs, seed_start, seeds,
                max_iter, time_limit,
            )

    # Summary statistics (per function x algorithm)
    summary = _build_summary_rows(raw, functions, algorithms)

    return {
        "metadata": {
            "givp_version": _GIVP_VERSION,
            "dims": dims,
            "n_runs": n_runs,
            "seed_start": seed_start,
            "seeds": seeds,
            "max_iter": max_iter,
            "time_limit": time_limit,
            "algorithms": algorithms,
            "functions": functions,
            "problem_references": {
                fn: PROBLEM_REGISTRY[fn]["reference"] for fn in functions
            },
            "algo_descriptions": {a: ALGO_DESCRIPTIONS[a] for a in algorithms},
        },
        "summary": summary,
        "records": raw,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_literature_comparison",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Output JSON is consumed by generate_report.py to produce\n"
            "Markdown/LaTeX tables, boxplots, and Wilcoxon test results."
        ),
    )
    p.add_argument(
        "--dims",
        type=int,
        default=10,
        metavar="N",
        help="Problem dimensionality, i.e. number of decision variables (default: 10).",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Independent runs per (algorithm, function) pair (default: 30). "
            "≥30 is recommended for publication-quality statistics."
        ),
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        metavar="N",
        help="First seed; runs use seeds [N, N + n_runs) (default: 0).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=200,
        metavar="N",
        help="Max iterations per run (default: 200).",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Per-run wall-clock time limit in seconds (default: 30.0).",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["GIVP-full", "GRASP-only"],
        choices=list(ALGO_DESCRIPTIONS),
        metavar="ALGO",
        help=(
            "Algorithms to include.  Choices: "
            f"{list(ALGO_DESCRIPTIONS)}.  "
            "DE and SA require scipy (pip install scipy).  "
            "Default: GIVP-full GRASP-only."
        ),
    )
    p.add_argument(
        "--functions",
        nargs="+",
        default=list(PROBLEM_REGISTRY),
        choices=list(PROBLEM_REGISTRY),
        metavar="FUNC",
        help=(
            f"Benchmark functions.  Choices: {list(PROBLEM_REGISTRY)}.  "
            "Default: all six functions."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_results.json"),
        metavar="PATH",
        help="Output JSON file path (default: experiment_results.json).",
    )
    p.add_argument(
        "--tune-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to JSON produced by tune_hyperparams.py.  "
            "Required when GIVP-tuned is in --algorithms."
        ),
    )
    p.add_argument(
        "--traces",
        action="store_true",
        help=(
            "Capture per-iteration best-value history for GIVP algorithms. "
            "Enables convergence plots in generate_report.py.  "
            "Increases output file size."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing --output file: skip functions that already "
            "have complete results in the checkpoint and continue from where "
            "the experiment left off."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress after each run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments, run experiment, save JSON, print summary."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    givp_tuned_config: GIVPConfig | None = None
    if "GIVP-tuned" in args.algorithms:
        if args.tune_config is None:
            _log.error(
                "[error] --algorithms GIVP-tuned requires --tune-config PATH\n"
                "        Run tune_hyperparams.py first to generate a config file."
            )
            return 1
        tune_data = json.loads(args.tune_config.read_text(encoding="utf-8"))
        params = tune_data.get("best_params", tune_data)
        givp_tuned_config = GIVPConfig(**params)

    n_total = len(args.algorithms) * len(args.functions) * args.n_runs
    est_min = n_total * 0.5 / 60  # rough estimate assuming ~0.5s per run

    _log.info("=" * 60)
    _log.info("GIVP -- Literature Comparison Benchmark")
    _log.info("=" * 60)
    _log.info("  givp version  : %s", _GIVP_VERSION)
    _log.info("  dims          : %s", args.dims)
    _log.info(
        "  n_runs        : %d  (seeds %d-%d)",
        args.n_runs, args.seed_start, args.seed_start + args.n_runs - 1,
    )
    _log.info("  max_iter      : %s", args.max_iter)
    _log.info("  time_limit    : %ss per run", args.time_limit)
    _log.info("  algorithms    : %s", args.algorithms)
    _log.info("  functions     : %s", args.functions)
    _log.info("  capture traces: %s", args.traces)
    _log.info("  total runs    : %d  (~%d min estimated)", n_total, int(est_min))
    _log.info("  output        : %s", args.output)
    _log.info("")

    t_wall = time.perf_counter()
    payload = run_experiment(
        algorithms=args.algorithms,
        functions=args.functions,
        dims=args.dims,
        n_runs=args.n_runs,
        seed_start=args.seed_start,
        max_iter=args.max_iter,
        time_limit=args.time_limit,
        capture_traces=args.traces,
        givp_tuned_config=givp_tuned_config,
        checkpoint_path=args.output,
        resume=args.resume,
    )
    elapsed = time.perf_counter() - t_wall

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log.info("\nFinished in %.1fs  →  %s", elapsed, args.output.resolve())
    _log.info("")

    # Quick console summary
    col_w = 14
    header = (
        f"{'Function':<{col_w}} {'Algorithm':<{col_w}} "
        f"{'Mean':>12} {'Std':>12} {'Best':>12} {'Median':>12}"
    )
    _log.info(header)
    _log.info("-" * len(header))
    for row in payload["summary"]:
        _log.info(
            "%s %s %12.4e %12.4e %12.4e %12.4e",
            f"{row['function']:<{col_w}}", f"{row['algorithm']:<{col_w}}",
            row["mean"], row["std"], row["best"], row["median"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
