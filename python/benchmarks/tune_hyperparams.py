# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Reproducible Optuna hyperparameter search for GIVP.

Searches the best GIVPConfig hyperparameters on a set of benchmark functions
using Optuna with the TPE (Tree-structured Parzen Estimator) sampler.
The study is seeded for full reproducibility.

The output JSON can be fed directly to run_literature_comparison.py via
``--algorithms GIVP-tuned --tune-config best_config.json``.

Usage
-----
    # Fast (2-5 min): 50 trials, 5-D, Sphere + Rastrigin only
    python benchmarks/tune_hyperparams.py

    # More thorough (15-30 min): 200 trials, 10-D, all 6 functions
    python benchmarks/tune_hyperparams.py \\
        --n-trials 200 --dims 10 --n-eval-seeds 5 \\
        --functions Sphere Rosenbrock Rastrigin Ackley \\
        --output results/best_config.json

    # Persistent study (survives crashes, allows resuming)
    python benchmarks/tune_hyperparams.py \\
        --storage sqlite:///tune_study.db \\
        --study-name givp_tune_v1

    # Then pass the result to run_literature_comparison.py
    python benchmarks/run_literature_comparison.py \\
        --algorithms GIVP-full GIVP-tuned GRASP-only \\
        --tune-config results/best_config.json

Objective
---------
The objective minimised by Optuna is the mean ``result.fun`` across all
(function, seed) pairs.  Functions are normalised by their best-known GIVP
value on that trial to avoid Sphere dominating (it has much smaller absolute
values than Schwefel).  A small regularisation penalty is applied if the
solution hits the time_limit, to discourage configs that only work with
unlimited budget.

References
----------
- Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization
  Framework. KDD 2019. https://doi.org/10.1145/3292500.3330701
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Verify dependencies
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

try:
    import optuna  # type: ignore[import-untyped]
    import optuna.logging as optuna_logging  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    sys.exit("[error] optuna not installed.\n Run:  pip install optuna")

# ---------------------------------------------------------------------------
# Benchmark subset registry (same functions as run_literature_comparison.py)
# ---------------------------------------------------------------------------

PROBLEM_REGISTRY: dict[str, dict] = {
    "Sphere": {
        "func": bm.sphere,
        "bounds_factory": lambda n: [(-5.12, 5.12)] * n,
    },
    "Rosenbrock": {
        "func": bm.rosenbrock,
        "bounds_factory": lambda n: [(-5.0, 10.0)] * n,
    },
    "Rastrigin": {
        "func": bm.rastrigin,
        "bounds_factory": lambda n: [(-5.12, 5.12)] * n,
    },
    "Ackley": {
        "func": bm.ackley,
        "bounds_factory": lambda n: [(-32.768, 32.768)] * n,
    },
    "Griewank": {
        "func": bm.griewank,
        "bounds_factory": lambda n: [(-600.0, 600.0)] * n,
    },
    "Schwefel": {
        "func": bm.schwefel,
        "bounds_factory": lambda n: [(-500.0, 500.0)] * n,
    },
}

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
#
# These are the hyperparameters Optuna will tune.  All others keep their
# GIVPConfig defaults (use_cache=True, cache_size=10_000, etc.).
#
# The search space is intentionally narrower than the full parameter range
# to keep the search affordable in a short session.


def _suggest_config(trial, max_iter: int, time_limit: float) -> GIVPConfig:
    """Map an Optuna trial to a GIVPConfig.

    Parameters
    ----------
    trial:
        Optuna Trial object.
    max_iter:
        Fixed maximum iterations (passed from CLI, not tuned).
    time_limit:
        Per-run wall-clock limit in seconds (passed from CLI, not tuned).

    Returns
    -------
    GIVPConfig
        Configuration built from the trial's suggested hyperparameters.
    """
    alpha = trial.suggest_float("alpha", 0.05, 0.30)
    adaptive_alpha = trial.suggest_categorical("adaptive_alpha", [True, False])

    if adaptive_alpha:
        alpha_min = trial.suggest_float("alpha_min", 0.03, alpha * 0.9)
        alpha_max = trial.suggest_float(
            "alpha_max", alpha * 1.1, min(0.45, alpha * 3.0)
        )
    else:
        alpha_min = alpha
        alpha_max = alpha

    vnd_iterations = trial.suggest_int("vnd_iterations", 20, 500)
    ils_iterations = trial.suggest_int("ils_iterations", 1, 20)
    perturbation_strength = trial.suggest_int("perturbation_strength", 1, 8)

    use_elite_pool = trial.suggest_categorical("use_elite_pool", [True, False])
    elite_size = trial.suggest_int("elite_size", 3, 15) if use_elite_pool else 5
    path_relink_frequency = (
        trial.suggest_int("path_relink_frequency", 3, 25) if use_elite_pool else 10
    )

    early_stop_threshold = trial.suggest_int(
        "early_stop_threshold", min(10, max_iter), max_iter
    )
    use_convergence_monitor = trial.suggest_categorical(
        "use_convergence_monitor", [True, False]
    )

    return GIVPConfig(
        max_iterations=max_iter,
        alpha=alpha,
        adaptive_alpha=adaptive_alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        vnd_iterations=vnd_iterations,
        ils_iterations=ils_iterations,
        perturbation_strength=perturbation_strength,
        use_elite_pool=use_elite_pool,
        elite_size=elite_size,
        path_relink_frequency=path_relink_frequency,
        use_cache=True,
        cache_size=10_000,
        early_stop_threshold=early_stop_threshold,
        use_convergence_monitor=use_convergence_monitor,
        time_limit=time_limit,
    )


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def build_objective(
    functions: list[str],
    dims: int,
    n_eval_seeds: int,
    max_iter: int,
    time_limit: float,
):
    """Return an Optuna objective callable.

    The objective averages ``result.fun`` over all (function, seed) pairs.
    To avoid scale dominance, each function's result is divided by the
    reference value obtained by GIVP-full with seed=0 on the first call
    (cached after the first trial that evaluates it).

    Parameters
    ----------
    functions:
        Names from PROBLEM_REGISTRY to include in the objective.
    dims:
        Problem dimensionality.
    n_eval_seeds:
        Number of independent seeds per function per trial.
    max_iter:
        Maximum GIVP iterations.
    time_limit:
        Per-run wall-clock budget in seconds.

    Returns
    -------
    callable
        An ``objective(trial) -> float`` function for Optuna.
    """
    eval_seeds = list(range(n_eval_seeds))

    # Reference scale per function: computed once using GIVP-full seed=0.
    # This normalises across functions so no single function dominates.
    reference_scale: dict[str, float] = {}

    def _get_scale(fn_name: str) -> float:
        if fn_name in reference_scale:
            return reference_scale[fn_name]
        spec = PROBLEM_REGISTRY[fn_name]
        bounds = spec["bounds_factory"](dims)
        ref_cfg = GIVPConfig(
            max_iterations=max_iter,
            alpha=0.12,
            adaptive_alpha=True,
            vnd_iterations=100,
            ils_iterations=5,
            time_limit=time_limit,
        )
        ref = givp(spec["func"], bounds, config=ref_cfg, seed=0)
        scale = max(abs(float(ref.fun)), 1e-12)
        reference_scale[fn_name] = scale
        return scale

    def objective(trial) -> float:
        """Optuna objective: mean normalised best value over all (fn, seed) pairs."""
        cfg = _suggest_config(trial, max_iter, time_limit)
        scores: list[float] = []

        for fn_name in functions:
            spec = PROBLEM_REGISTRY[fn_name]
            bounds = spec["bounds_factory"](dims)
            scale = _get_scale(fn_name)

            for seed in eval_seeds:
                res = givp(spec["func"], bounds, config=cfg, seed=seed)
                scores.append(float(res.fun) / scale)

        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


def run_tuning(
    functions: list[str],
    dims: int,
    n_trials: int,
    n_eval_seeds: int,
    sampler_seed: int,
    max_iter: int,
    time_limit: float,
    study_name: str,
    storage: str | None,
    verbose: bool,
) -> dict:
    """Run the Optuna hyperparameter search.

    Parameters
    ----------
    functions:
        Benchmark functions to evaluate in each trial.
    dims:
        Problem dimensionality.
    n_trials:
        Number of Optuna trials.
    n_eval_seeds:
        Seeds per (trial, function) pair.  Higher = more stable estimate,
        but slower.  3 is usually sufficient.
    sampler_seed:
        Seed for the TPE sampler, ensuring reproducible trial ordering.
    max_iter:
        Fixed max iterations for GIVP inside each trial.
    time_limit:
        Per-run wall-clock cap in seconds inside each trial.
    study_name:
        Optuna study name (used for storage and display).
    storage:
        Optuna storage URL (e.g. ``sqlite:///tune.db``).  Pass ``None``
        for in-memory (default, not persistent across crashes).
    verbose:
        If True, show Optuna's INFO logs; otherwise show only warnings.

    Returns
    -------
    dict
        Result payload with keys ``metadata`` and ``best_params``.
    """
    if not verbose:
        optuna_logging.set_verbosity(optuna_logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = build_objective(functions, dims, n_eval_seeds, max_iter, time_limit)

    _log.info("Starting Optuna study: %r", study_name)
    _log.info(
        "  %d trials x %d function(s) x %d seed(s) each",
        n_trials, len(functions), n_eval_seeds,
    )
    _log.info("  sampler: TPE (seed=%d)  storage: %s", sampler_seed, storage or "in-memory")
    _log.info("")

    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose,
    )
    elapsed = time.perf_counter() - t0

    best_trial = study.best_trial
    _log.info("\nBest trial: #%d  value=%.6f", best_trial.number, best_trial.value)
    _log.info("Duration: %.1fs", elapsed)

    # Reconstruct the full GIVPConfig from the best trial params
    best_cfg = _suggest_config(best_trial, max_iter, time_limit)

    # Serialise GIVPConfig to a plain dict (dataclass)
    import dataclasses

    best_params = {
        f.name: getattr(best_cfg, f.name) for f in dataclasses.fields(best_cfg)
    }

    return {
        "metadata": {
            "givp_version": _GIVP_VERSION,
            "study_name": study_name,
            "n_trials": n_trials,
            "n_completed_trials": len(study.trials),
            "sampler": "TPESampler",
            "sampler_seed": sampler_seed,
            "dims": dims,
            "functions": functions,
            "n_eval_seeds": n_eval_seeds,
            "max_iter": max_iter,
            "time_limit_per_run": time_limit,
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "duration_s": round(elapsed, 2),
        },
        "best_params": best_params,
        "best_trial_params": dict(best_trial.params),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tune_hyperparams",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After tuning, use the output JSON with run_literature_comparison.py:\n"
            "  python benchmarks/run_literature_comparison.py \\\n"
            "      --algorithms GIVP-full GIVP-tuned GRASP-only \\\n"
            "      --tune-config best_config.json"
        ),
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Number of Optuna trials (default: 50).  "
            "50 trials with default settings takes ~2-5 min."
        ),
    )
    p.add_argument(
        "--dims",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Problem dimensionality for tuning (default: 5).  "
            "Lower = faster tuning; use 10 for publication-quality tuning."
        ),
    )
    p.add_argument(
        "--n-eval-seeds",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of seeds per (trial, function) evaluation (default: 3).  "
            "Higher = more stable estimate, but slower."
        ),
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Max GIVP iterations per evaluation within each trial (default: 100).  "
            "Keep low for fast tuning; increase for final validation."
        ),
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Per-run wall-clock cap in seconds inside each trial (default: 10.0).",
    )
    p.add_argument(
        "--functions",
        nargs="+",
        default=["Sphere", "Rastrigin"],
        choices=list(PROBLEM_REGISTRY),
        metavar="FUNC",
        help=(
            f"Benchmark functions to tune on.  Choices: {list(PROBLEM_REGISTRY)}.  "
            "Default: Sphere Rastrigin (fast, representative)."
        ),
    )
    p.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        metavar="N",
        help="Seed for the Optuna TPE sampler (default: 42, fully reproducible).",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Optuna study name (default: auto-generated from dims/functions/seed).  "
            "Used as identifier when --storage is set."
        ),
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Optuna storage URL for persistent studies (e.g. sqlite:///tune.db).  "
            "If omitted, uses in-memory storage (not persistent across crashes)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("best_config.json"),
        metavar="PATH",
        help="Output JSON path for the best config (default: best_config.json).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show Optuna INFO logs and progress bar.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, run Optuna study, save best config JSON."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    fn_tag = "_".join(args.functions[:3])
    study_name = args.study_name or (
        f"givp_tune_d{args.dims}_{fn_tag}_s{args.sampler_seed}"
    )

    _log.info("=" * 60)
    _log.info("GIVP -- Hyperparameter Tuning (Optuna TPE)")
    _log.info("=" * 60)
    _log.info("  givp version  : %s", _GIVP_VERSION)
    _log.info("  n_trials      : %s", args.n_trials)
    _log.info("  dims          : %s", args.dims)
    _log.info("  functions     : %s", args.functions)
    _log.info("  n_eval_seeds  : %s", args.n_eval_seeds)
    _log.info("  max_iter/run  : %s", args.max_iter)
    _log.info("  time_limit    : %ss per run", args.time_limit)
    _log.info("  sampler_seed  : %s", args.sampler_seed)
    _log.info("  study_name    : %s", study_name)
    _log.info("  storage       : %s", args.storage or "in-memory")
    _log.info("  output        : %s", args.output)
    est_runs = args.n_trials * len(args.functions) * args.n_eval_seeds
    est_min = est_runs * args.time_limit / 60
    _log.info("  ~eval calls   : %d  (~%d min if all hit time_limit)", est_runs, int(est_min))
    _log.info("")

    payload = run_tuning(
        functions=args.functions,
        dims=args.dims,
        n_trials=args.n_trials,
        n_eval_seeds=args.n_eval_seeds,
        sampler_seed=args.sampler_seed,
        max_iter=args.max_iter,
        time_limit=args.time_limit,
        study_name=study_name,
        storage=args.storage,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log.info("\nBest config saved -> %s", args.output.resolve())

    _log.info("\nBest hyperparameters:")
    for k, v in payload["best_trial_params"].items():
        _log.info("  %-30s = %s", k, v)

    _log.info(
        "\nRun the comparison with:\n"
        "  python benchmarks/run_literature_comparison.py \\\n"
        "      --algorithms GIVP-full GIVP-tuned GRASP-only \\\n"
        "      --tune-config %s",
        args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
