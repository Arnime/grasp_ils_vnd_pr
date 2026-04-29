# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Statistical analysis and report generation from a GIVP benchmark experiment.

Reads the JSON produced by ``run_literature_comparison.py`` and generates:

    - Console summary table
    - Wilcoxon signed-rank test table (non-parametric, α = 0.05)
    - Markdown tables   — paste directly into README or paper supplementary
    - LaTeX tables      — booktabs format, ready for SBPO / BRACIS / journals
    - Boxplot PNG       — objective value distribution per function × algorithm
    - Convergence PNG   — best-so-far per iteration (only if JSON has traces)

Usage
-----
    # Console + Markdown + plots
    python generate_report.py --input experiment_results.json

    # LaTeX tables only (no plots)
    python generate_report.py --input results.json --format latex --no-plots

    # Compare all algorithms against Differential Evolution as the reference
    python generate_report.py --input results.json --reference DE

    # Save everything to a specific directory
    python generate_report.py --input results.json --output-dir paper/tables/

Statistical method
------------------
We apply the two-sided Wilcoxon signed-rank test (Wilcoxon, 1945) to compare
each non-reference algorithm against the reference on matched pairs of 30 runs
(same seeds).  p-values below α = 0.05 indicate that the reference algorithm
achieves significantly different objective values.  Effect size is reported via
the rank-biserial correlation r = 1 - 2W / (n(n+1)/2).

Reference: Wilcoxon, F. (1945). Individual comparisons by ranking methods.
Biometrics Bulletin, 1(6), 80-83.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies with graceful fallback
# ---------------------------------------------------------------------------

try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import matplotlib  # type: ignore[import-untyped]

    matplotlib.use("Agg")  # non-interactive backend safe for scripts
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _MPL_OK = True
except ImportError:
    _MPL_OK = False

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class RunRecord(NamedTuple):
    """Immutable record for a single (algorithm, function, seed) run result."""

    algorithm: str
    function: str
    seed: int
    fun: float
    nit: int
    nfev: int
    time_s: float
    trace: list[float] | None


class SummaryRow(NamedTuple):
    """Pre-computed descriptive statistics for one (function, algorithm) cell."""

    function: str
    algorithm: str
    n_runs: int
    mean: float
    std: float
    best: float
    median: float
    worst: float
    nfev_mean: float


class WilcoxonRow(NamedTuple):
    """Result of a Wilcoxon signed-rank test for one (challenger, function) pair."""

    function: str
    algorithm: str  # the challenger
    reference: str  # the reference algorithm
    stat: float
    pvalue: float
    effect_r: float  # rank-biserial correlation
    significant: bool  # p < alpha


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_results(path: Path) -> tuple[dict, list[RunRecord], list[SummaryRow]]:
    """Parse the JSON output of run_literature_comparison.py.

    Returns
    -------
    (metadata, records, summary)
        - metadata : dict with experimental settings
        - records  : flat list of RunRecord (one per run)
        - summary  : pre-computed statistics per (function, algorithm)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = raw["metadata"]

    records: list[RunRecord] = []
    for fn_name, fn_recs in raw["records"].items():
        for r in fn_recs:
            records.append(
                RunRecord(
                    algorithm=r["algorithm"],
                    function=fn_name,
                    seed=r["seed"],
                    fun=r["fun"],
                    nit=r.get("nit", 0),
                    nfev=r.get("nfev", 0),
                    time_s=r.get("time_s", 0.0),
                    trace=r.get("trace"),
                )
            )

    summary: list[SummaryRow] = [SummaryRow(**row) for row in raw.get("summary", [])]

    return meta, records, summary


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def wilcoxon_table(
    records: list[RunRecord],
    reference: str,
    alpha: float = 0.05,
) -> list[WilcoxonRow]:
    """Compute Wilcoxon signed-rank tests for each algorithm vs *reference*.

    Only algorithms that share the same seeds as *reference* are compared
    (matched-pairs requirement).  The alternative hypothesis is two-sided:
    the two algorithms produce different objective values.

    Parameters
    ----------
    records:
        Raw per-run records (all algorithms).
    reference:
        Algorithm name to use as the baseline in pairwise comparisons.
    alpha:
        Significance level (default: 0.05).

    Returns
    -------
    list[WilcoxonRow]
        One row per (challenger, function) pair.  Excludes reference vs itself.
    """
    if not _SCIPY_OK:
        _log.warning(
            "[warning] scipy not installed — Wilcoxon tests skipped.\n"
            "          Run:  pip install scipy"
        )
        return []

    # Group values by (function, algorithm, seed) for matched pairs
    idx: dict[tuple[str, str, int], float] = {}
    for r in records:
        idx[(r.function, r.algorithm, r.seed)] = r.fun

    functions = sorted({r.function for r in records})
    algorithms = sorted({r.algorithm for r in records})
    challengers = [a for a in algorithms if a != reference]

    rows: list[WilcoxonRow] = []
    for fn_name in functions:
        ref_seeds = sorted(
            s for (fn, algo, s) in idx if fn == fn_name and algo == reference
        )
        if not ref_seeds:
            continue

        for challenger in challengers:
            ref_vals = [
                idx[(fn_name, reference, s)]
                for s in ref_seeds
                if (fn_name, challenger, s) in idx
            ]
            chal_vals = [
                idx[(fn_name, challenger, s)]
                for s in ref_seeds
                if (fn_name, challenger, s) in idx
            ]
            if len(ref_vals) < 2:
                continue

            a = np.asarray(ref_vals, dtype=float)
            b = np.asarray(chal_vals, dtype=float)

            # scipy ≥1.7 supports zero_method="wilcox" and nan_policy
            # Unpack as tuple to avoid Pylance attribute-access false positives
            # on the conditionally-imported _scipy_stats module.
            _stat, _pvalue = _scipy_stats.wilcoxon(a, b, alternative="two-sided")
            stat = float(_stat)  # type: ignore[arg-type]  # scipy stubs unavailable
            pvalue = float(_pvalue)  # type: ignore[arg-type]  # scipy stubs unavailable
            n = len(a)
            # Rank-biserial correlation: effect size for Wilcoxon
            effect_r = float(1.0 - (2.0 * stat) / (n * (n + 1) / 2.0))

            rows.append(
                WilcoxonRow(
                    function=fn_name,
                    algorithm=challenger,
                    reference=reference,
                    stat=stat,
                    pvalue=pvalue,
                    effect_r=effect_r,
                    significant=pvalue < alpha,
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Console / Markdown / LaTeX table generators
# ---------------------------------------------------------------------------


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format mean ± std in consistent scientific notation."""
    return f"{mean:.4e} ± {std:.4e}"


def print_console_summary(
    summary: list[SummaryRow], wilcoxon: list[WilcoxonRow]
) -> None:
    """Print a human-readable summary table to stdout."""
    pval_idx: dict[tuple[str, str], float] = {
        (r.function, r.algorithm): r.pvalue for r in wilcoxon
    }
    sig_idx: dict[tuple[str, str], bool] = {
        (r.function, r.algorithm): r.significant for r in wilcoxon
    }

    functions = list(dict.fromkeys(r.function for r in summary))
    col = 16

    print()
    for fn_name in functions:
        print(f"  ─── {fn_name} ───")
        print(
            f"  {'Algorithm':<{col}} {'Mean':>14} {'Std':>14} "
            f"{'Best':>14} {'Median':>14} {'p-value':>10} {'Sig':>5}"
        )
        print("  " + "─" * (col + 14 * 4 + 10 + 5 + 6))
        for row in summary:
            if row.function != fn_name:
                continue
            pval_str = (
                f"{pval_idx[(fn_name, row.algorithm)]:.4f}"
                if (fn_name, row.algorithm) in pval_idx
                else "  ref"
            )
            sig_str = "★" if sig_idx.get((fn_name, row.algorithm)) else "—"
            print(
                f"  {row.algorithm:<{col}} "
                f"{row.mean:>14.4e} {row.std:>14.4e} "
                f"{row.best:>14.4e} {row.median:>14.4e} "
                f"{pval_str:>10} {sig_str:>5}"
            )
        print()


def to_markdown(
    summary: list[SummaryRow],
    wilcoxon: list[WilcoxonRow],
    meta: dict,
) -> str:
    """Return a Markdown string with one table per benchmark function.

    Suitable for pasting into README.md or paper supplementary material.
    """
    pval_idx = {(r.function, r.algorithm): r.pvalue for r in wilcoxon}
    sig_idx = {(r.function, r.algorithm): r.significant for r in wilcoxon}

    lines: list[str] = []
    dims = meta.get("dims", "?")
    n_runs = meta.get("n_runs", "?")
    givp_ver = meta.get("givp_version", "?")

    lines.append(
        f"<!-- generated by generate_report.py — "
        f"givp {givp_ver}, n={dims}, {n_runs} runs -->"
    )
    lines.append("")

    functions = list(dict.fromkeys(r.function for r in summary))
    algo_descriptions = meta.get("algo_descriptions", {})

    for fn_name in functions:
        ref = meta.get("reference_algorithm", "GIVP-full")
        ref_note = meta.get("problem_references", {}).get(fn_name, "")
        lines.append(f"### {fn_name}")
        if ref_note:
            lines.append(f"*Reference: {ref_note}*  ")
        lines.append("")
        lines.append(
            "| Algorithm | Mean ± Std | Best | Median | Worst | "
            "NFev (mean) | p-value | Sig |"
        )
        lines.append(
            "|-----------|------------|------|--------|-------|"
            "------------|---------|-----|"
        )
        for row in summary:
            if row.function != fn_name:
                continue
            pval_str = (
                f"{pval_idx[(fn_name, row.algorithm)]:.4f}"
                if (fn_name, row.algorithm) in pval_idx
                else "*(ref)*"
            )
            sig_str = "★" if sig_idx.get((fn_name, row.algorithm)) else "—"
            lines.append(
                f"| {row.algorithm} "
                f"| {_fmt_mean_std(row.mean, row.std)} "
                f"| {row.best:.4e} "
                f"| {row.median:.4e} "
                f"| {row.worst:.4e} "
                f"| {int(row.nfev_mean)} "
                f"| {pval_str} "
                f"| {sig_str} |"
            )
        lines.append("")

    # Algorithm legend
    lines.append("**Algorithm descriptions:**")
    lines.append("")
    for algo, desc in algo_descriptions.items():
        lines.append(f"- **{algo}**: {desc}")
    lines.append("")
    lines.append(
        "★ p < 0.05 (Wilcoxon signed-rank test, two-sided) — "
        f"significantly different from *{ref}*."
    )

    return "\n".join(lines)


_LATEX_MIDRULE = r"\midrule"


def _latex_wilcoxon_subtable(wilcoxon: list[WilcoxonRow], _esc) -> list[str]:
    """Build the Wilcoxon significance sub-table lines for the LaTeX report."""
    lines: list[str] = []
    lines.append("")
    lines.append(r"\medskip")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Function & Challenger & $W$ & $p$-value & $r$ (effect) \\")
    lines.append(_LATEX_MIDRULE)
    for w in wilcoxon:
        sig = r"$^\star$" if w.significant else ""
        lines.append(
            f"  {w.function} & {_esc(w.algorithm)}{sig} "
            f"& {w.stat:.1f} "
            f"& {w.pvalue:.4f} "
            f"& {w.effect_r:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\\ \footnotesize{$W$: Wilcoxon test statistic; "
        r"$r$: rank-biserial correlation (effect size); "
        r"$^\star$: $p < 0.05$.}"
    )
    return lines


def to_latex(
    summary: list[SummaryRow],
    wilcoxon: list[WilcoxonRow],
    meta: dict,
) -> str:
    """Return a LaTeX string with a booktabs comparison table.

    Designed for SBPO / BRACIS paper submissions (A4, two-column or single).
    Requires: \\usepackage{booktabs} in the preamble.
    """
    sig_idx = {(r.function, r.algorithm): r.significant for r in wilcoxon}

    dims = meta.get("dims", "?")
    n_runs = meta.get("n_runs", "?")
    givp_ver = meta.get("givp_version", "?")

    functions = list(dict.fromkeys(r.function for r in summary))
    algorithms = list(dict.fromkeys(r.algorithm for r in summary))
    ref = algorithms[0] if algorithms else "GIVP-full"

    def _esc(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    lines: list[str] = []
    lines.append("% Generated by generate_report.py")
    lines.append(f"% givp {givp_ver} | n={dims} | {n_runs} independent runs per cell")
    lines.append("")
    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Comparison of optimization algorithms on standard benchmark functions "
        f"($n={dims}$ variables, {n_runs} independent runs per cell). "
        r"Mean $\pm$ std over all runs. "
        r"$^\star$ denotes $p < 0.05$ (Wilcoxon signed-rank, two-sided) "
        f"vs.\\ \\texttt{{{_esc(ref)}}}.}}"
    )
    lines.append(r"\label{tab:benchmark_comparison}")
    lines.append(r"\begin{tabular}{ll" + "r" * 4 + r"}")
    lines.append(r"\toprule")
    lines.append(r"Function & Algorithm & Mean & Std & Best & Median \\")
    lines.append(_LATEX_MIDRULE)

    for fn_name in functions:
        fn_rows = [r for r in summary if r.function == fn_name]
        for i, row in enumerate(fn_rows):
            fn_cell = (
                r"\multirow{" + str(len(fn_rows)) + r"}{*}{" + fn_name + "}"
                if i == 0
                else ""
            )
            sig = r"$^\star$" if sig_idx.get((fn_name, row.algorithm)) else ""
            lines.append(
                f"  {fn_cell} & {_esc(row.algorithm)}{sig} "
                f"& ${row.mean:.4e}$ "
                f"& ${row.std:.4e}$ "
                f"& ${row.best:.4e}$ "
                f"& ${row.median:.4e}$ \\\\"
            )
        lines.append(_LATEX_MIDRULE)

    # Replace last \midrule with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    if wilcoxon:
        lines.extend(_latex_wilcoxon_subtable(wilcoxon, _esc))

    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_boxplot(
    records: list[RunRecord],
    meta: dict,
    output_path: Path,
) -> None:
    """Save a boxplot figure: objective value distribution per function × algorithm."""
    if not _MPL_OK:
        _log.warning("[warning] matplotlib not installed — boxplot skipped.")
        return

    functions = meta.get("functions", sorted({r.function for r in records}))
    algorithms = meta.get("algorithms", sorted({r.algorithm for r in records}))
    dims = meta.get("dims", "?")
    n_runs = meta.get("n_runs", "?")

    palette = [
        "#2196F3",
        "#FF9800",
        "#4CAF50",
        "#F44336",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#607D8B",
    ]
    algo_colors = {a: palette[i % len(palette)] for i, a in enumerate(algorithms)}

    fig, axes = plt.subplots(
        1,
        len(functions),
        figsize=(max(4 * len(functions), 8), 5),
        sharey=False,
    )
    if len(functions) == 1:
        axes = [axes]

    for ax, fn_name in zip(axes, functions, strict=True):
        data = [
            [r.fun for r in records if r.function == fn_name and r.algorithm == a]
            for a in algorithms
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 4, "alpha": 0.5},
        )
        ax.set_xticks(range(1, len(algorithms) + 1))
        ax.set_xticklabels(algorithms)
        for patch, a in zip(bp["boxes"], algorithms, strict=True):
            patch.set_facecolor(algo_colors[a])
            patch.set_alpha(0.75)

        ax.set_title(fn_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Objective value", fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Objective value distribution — n={dims}, {n_runs} runs per cell",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info("  Boxplot saved \u2192 %s", output_path)


def plot_convergence(
    records: list[RunRecord],
    meta: dict,
    output_path: Path,
    trace_seed: int = 0,
) -> None:
    """Save convergence traces (best-so-far per iteration) for one seed.

    Only runs with a non-None `trace` list are plotted.  If the JSON was
    generated without ``--traces``, this function prints a warning and exits.
    """
    if not _MPL_OK:
        _log.warning("[warning] matplotlib not installed — convergence plot skipped.")
        return

    has_traces = any(r.trace is not None for r in records)
    if not has_traces:
        _log.warning(
            "[warning] No convergence traces in the JSON.  "
            "Re-run run_literature_comparison.py with --traces to capture them."
        )
        return

    functions = meta.get("functions", sorted({r.function for r in records}))
    algorithms = meta.get("algorithms", sorted({r.algorithm for r in records}))

    palette = [
        "#2196F3",
        "#FF9800",
        "#4CAF50",
        "#F44336",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#607D8B",
    ]
    algo_colors = {a: palette[i % len(palette)] for i, a in enumerate(algorithms)}

    fig, axes = plt.subplots(
        1,
        len(functions),
        figsize=(max(4 * len(functions), 8), 4),
        sharey=False,
    )
    if len(functions) == 1:
        axes = [axes]

    for ax, fn_name in zip(axes, functions, strict=True):
        plotted_any = False
        for a in algorithms:
            matching = [
                r
                for r in records
                if r.function == fn_name
                and r.algorithm == a
                and r.seed == trace_seed
                and r.trace is not None
            ]
            if not matching:
                continue
            trace = matching[0].trace
            assert trace is not None  # narrowing
            ax.plot(
                range(1, len(trace) + 1),
                trace,
                label=a,
                color=algo_colors[a],
                linewidth=1.6,
            )
            plotted_any = True

        ax.set_title(fn_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Best objective (symlog)", fontsize=9)
        ax.set_yscale("symlog", linthresh=1e-6)
        if plotted_any:
            ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Convergence traces — seed={trace_seed}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info("  Convergence plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSON file produced by run_literature_comparison.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for output files (tables + plots).  "
            "Defaults to the same directory as --input."
        ),
    )
    p.add_argument(
        "--format",
        choices=["markdown", "latex", "both"],
        default="both",
        help="Table format(s) to generate (default: both).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip boxplot and convergence PNG generation.",
    )
    p.add_argument(
        "--reference",
        type=str,
        default=None,
        metavar="ALGO",
        help=(
            "Reference algorithm for Wilcoxon pairwise tests.  "
            "Defaults to the first algorithm in the JSON."
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Significance level for Wilcoxon tests (default: 0.05).",
    )
    p.add_argument(
        "--trace-seed",
        type=int,
        default=0,
        metavar="N",
        help="Seed to use for convergence trace plot (default: 0).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show DEBUG-level messages (per-record Wilcoxon details, file paths).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load results JSON, run statistical tests, write reports."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.input.exists():
        _log.error("[error] Input file not found: %s", args.input)
        return 1

    meta, records, summary = load_results(args.input)
    out_dir = args.output_dir or args.input.parent
    stem = args.input.stem

    _log.debug("Loaded %d records, %d summary rows", len(records), len(summary))

    algorithms = meta.get("algorithms", sorted({r.algorithm for r in records}))
    reference = args.reference or (algorithms[0] if algorithms else "GIVP-full")

    if reference not in {r.algorithm for r in records}:
        _log.error(
            "[error] Reference algorithm %r not found in records.\n"
            "        Available: %s",
            reference,
            sorted({r.algorithm for r in records}),
        )
        return 1

    _log.info("Report for: %s", args.input)
    _log.info("  algorithms  : %s", algorithms)
    _log.info("  functions   : %s", meta.get("functions", "?"))
    _log.info("  dims        : %s", meta.get("dims", "?"))
    _log.info("  n_runs      : %s", meta.get("n_runs", "?"))
    _log.info("  reference   : %s", reference)
    _log.info("  alpha       : %s", args.alpha)
    _log.info("")

    # Wilcoxon tests
    wrows = wilcoxon_table(records, reference=reference, alpha=args.alpha)
    _log.debug(
        "Wilcoxon: %d significant pairs out of %d",
        sum(w.significant for w in wrows),
        len(wrows),
    )

    # Console summary
    print_console_summary(summary, wrows)

    if not _SCIPY_OK:
        _log.warning(textwrap.dedent("""\
            [note] scipy not installed — statistical significance tests skipped.
                   Install with:  pip install scipy
            """))

    # Markdown
    if args.format in ("markdown", "both"):
        md_path = out_dir / f"{stem}_report.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        meta["reference_algorithm"] = reference
        md_path.write_text(to_markdown(summary, wrows, meta), encoding="utf-8")
        _log.info("Markdown table → %s", md_path)

    # LaTeX
    if args.format in ("latex", "both"):
        tex_path = out_dir / f"{stem}_report.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(to_latex(summary, wrows, meta), encoding="utf-8")
        _log.info("LaTeX table    → %s", tex_path)

    # Plots
    if not args.no_plots:
        if not _MPL_OK:
            _log.warning(
                "[note] matplotlib not installed — plots skipped.\n"
                "       Install with:  pip install matplotlib"
            )
        else:
            plot_boxplot(records, meta, out_dir / f"{stem}_boxplot.png")
            plot_convergence(
                records,
                meta,
                out_dir / f"{stem}_convergence.png",
                trace_seed=args.trace_seed,
            )

    _log.info("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
