# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Statistical analysis and report generation from a Julia GIVP benchmark experiment.

Reads the JSON produced by ``run_literature_comparison.jl`` and generates:

    - Console summary table
    - Wilcoxon signed-rank test table (non-parametric, α = 0.05)
    - Markdown tables   — paste directly into README or paper supplementary
    - LaTeX tables      — booktabs format, ready for SBPO / BRACIS / journals

Usage (from repo root)
----------------------
    julia --project=julia julia/benchmarks/generate_report.jl \\
        --input julia/benchmarks/results_julia.json

    # LaTeX only, GIVP-full as reference
    julia --project=julia julia/benchmarks/generate_report.jl \\
        --input results.json --format latex --reference GIVP-full

Statistical method
------------------
Two-sided Wilcoxon signed-rank test (Wilcoxon, 1945).  For n > 20 we apply a
normal approximation with continuity correction; for n ≤ 20 we use the exact
distribution via enumeration.  p-values below α = 0.05 indicate that the
reference algorithm achieves significantly different objective values.
Effect size is reported as rank-biserial correlation r = 1 - 2W / (n(n+1)/2).

Reference: Wilcoxon, F. (1945). Individual comparisons by ranking methods.
Biometrics Bulletin, 1(6), 80–83.
"""

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using JSON
using Printf
using Statistics

# ── Normal CDF (Abramowitz & Stegun rational approximation, max err 7.5e-8) ──

function _normal_cdf(x::Float64)::Float64
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y =
        t * (
            0.254829592 +
            t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
        )
    p = 1.0 - y * exp(-x * x)
    x < 0 ? 1.0 - p : p
end

# ── Wilcoxon signed-rank test ─────────────────────────────────────────────────

"""
    wilcoxon_test(a, b) -> (stat, pvalue, effect_r)

Two-sided Wilcoxon signed-rank test for matched pairs `a` vs `b`.
Uses normal approximation with continuity correction (safe for n > 20).
For n ≤ 20 falls back to exact distribution via enumeration.

Returns `(W, p, r)`:
- `W`  — test statistic (sum of positive ranks)
- `p`  — two-sided p-value
- `r`  — rank-biserial correlation (effect size)
"""
function wilcoxon_test(a::Vector{Float64}, b::Vector{Float64})
    @assert length(a) == length(b) "wilcoxon_test: a and b must be same length"

    d = a .- b
    # Remove zero differences (ties on the null)
    nz = filter(!iszero, d)
    n = length(nz)
    n < 2 && return (0.0, 1.0, 0.0)

    # Rank |d_i| with average ranks for ties
    abs_d = abs.(nz)
    order = sortperm(abs_d)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && abs_d[order[j + 1]] ≈ abs_d[order[i]]
            j += 1
        end
        avg_rank = (i + j) / 2.0
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end

    W_plus = sum(ranks[k] for k in 1:n if nz[k] > 0; init = 0.0)
    W_minus = sum(ranks[k] for k in 1:n if nz[k] < 0; init = 0.0)
    W = min(W_plus, W_minus)

    effect_r = 1.0 - 2.0 * W / (n * (n + 1) / 2.0)

    if n <= 20
        # Exact: enumerate all 2^n sign assignments
        n_assign = 1 << n
        count_le = 0
        for mask in 0:(n_assign - 1)
            w = 0.0
            for k in 1:n
                if (mask >> (k - 1)) & 1 == 1
                    w += ranks[k]
                end
            end
            w <= W && (count_le += 1)
        end
        pvalue = 2.0 * count_le / n_assign
        pvalue = min(pvalue, 1.0)
    else
        # Normal approximation with continuity correction
        mu = n * (n + 1) / 4.0
        sigma = sqrt(n * (n + 1) * (2n + 1) / 24.0)
        z = (W - mu + 0.5) / sigma
        pvalue = 2.0 * _normal_cdf(-abs(z))
    end

    return W, pvalue, effect_r
end

# ── Data structures ───────────────────────────────────────────────────────────

struct RunRecord
    algorithm::String
    func::String
    seed::Int
    fun::Float64
    nit::Int
    nfev::Int
    time_s::Float64
    convergence_trace::Vector{Float64}   # empty when --traces not used
end

struct SummaryRow
    func::String
    algorithm::String
    n_runs::Int
    mean_val::Float64
    std_val::Float64
    best::Float64
    median_val::Float64
    worst::Float64
    nfev_mean::Float64
end

struct WilcoxonRow
    func::String
    algorithm::String   # challenger
    reference::String
    stat::Float64
    pvalue::Float64
    effect_r::Float64
    significant::Bool
end

# ── Loader ────────────────────────────────────────────────────────────────────

"""Load JSON output of run_literature_comparison.jl.

Returns `(metadata, records, summary_from_json)`.
Accepts both flat-list records (Julia format) and dict-by-function (Python format).
"""
function load_results(path::String)
    raw = JSON.parsefile(path)
    meta = get(raw, "metadata", Dict{String, Any}())

    records = RunRecord[]
    raw_records = get(raw, "records", [])

    if raw_records isa Vector
        # Julia flat-list format: [{"algorithm":..., "function":..., "seed":..., ...}]
        for r in raw_records
            push!(
                records,
                RunRecord(
                    r["algorithm"],
                    r["function"],
                    r["seed"],
                    Float64(r["fun"]),
                    get(r, "nit", 0),
                    get(r, "nfev", 0),
                    Float64(get(r, "time_s", 0.0)),
                    Float64.(get(r, "convergence_trace", Float64[])),
                ),
            )
        end
    elseif raw_records isa Dict
        # Python dict-by-function format: {"FunctionName": [{"algorithm":..., ...}]}
        for (fn_name, fn_recs) in raw_records
            for r in fn_recs
                push!(
                    records,
                    RunRecord(
                        r["algorithm"],
                        fn_name,
                        r["seed"],
                        Float64(r["fun"]),
                        get(r, "nit", 0),
                        get(r, "nfev", 0),
                        Float64(get(r, "time_s", 0.0)),
                        Float64.(get(r, "convergence_trace", Float64[])),
                    ),
                )
            end
        end
    end

    return meta, records
end

# ── Summary statistics ────────────────────────────────────────────────────────

function build_summary(records::Vector{RunRecord})::Vector{SummaryRow}
    # Collect unique (func, algorithm) pairs
    pairs = unique((r.func, r.algorithm) for r in records)
    rows = SummaryRow[]
    for (fn, algo) in sort(collect(pairs))
        vals = [r.fun for r in records if r.func == fn && r.algorithm == algo]
        nfevs = [Float64(r.nfev) for r in records if r.func == fn && r.algorithm == algo]
        isempty(vals) && continue
        push!(
            rows,
            SummaryRow(
                fn,
                algo,
                length(vals),
                mean(vals),
                length(vals) > 1 ? std(vals) : 0.0,
                minimum(vals),
                median(vals),
                maximum(vals),
                isempty(nfevs) ? 0.0 : mean(nfevs),
            ),
        )
    end
    return rows
end

# ── Statistical tests ─────────────────────────────────────────────────────────

function compute_wilcoxon(
    records::Vector{RunRecord},
    reference::String;
    alpha::Float64 = 0.05,
)::Vector{WilcoxonRow}
    # Index by (func, algorithm, seed)
    idx = Dict{Tuple{String, String, Int}, Float64}()
    for r in records
        idx[(r.func, r.algorithm, r.seed)] = r.fun
    end

    functions = unique(r.func for r in records)
    algorithms = unique(r.algorithm for r in records)
    challengers = filter(a -> a != reference, algorithms)

    rows = WilcoxonRow[]
    for fn in sort(functions)
        ref_seeds = sort(unique(s for (f, a, s) in keys(idx) if f == fn && a == reference))
        isempty(ref_seeds) && continue

        for chal in sort(challengers)
            # Matched pairs: only seeds present in BOTH reference and challenger
            common_seeds = [s for s in ref_seeds if haskey(idx, (fn, chal, s))]
            length(common_seeds) < 2 && continue

            a_vals = [idx[(fn, reference, s)] for s in common_seeds]
            b_vals = [idx[(fn, chal, s)] for s in common_seeds]

            W, pval, r = wilcoxon_test(a_vals, b_vals)
            push!(rows, WilcoxonRow(fn, chal, reference, W, pval, r, pval < alpha))
        end
    end
    return rows
end

# ── Convergence analysis ──────────────────────────────────────────────────────

"""Statistics at one checkpoint iteration for one (function, algorithm) pair."""
struct ConvergencePoint
    iteration::Int
    median_val::Float64
    mean_val::Float64
    std_val::Float64
    best::Float64
    n_traces::Int
end

"""Full convergence curve for one (function, algorithm) pair."""
struct ConvergenceStats
    func::String
    algorithm::String
    checkpoints::Vector{ConvergencePoint}
end

"""
    compute_convergence_stats(records; checkpoints) -> Vector{ConvergenceStats}

Extract per-iteration convergence curves from records that have `convergence_trace`.
`checkpoints` is a sorted list of iteration numbers (1-based).
Traces shorter than a checkpoint are padded with their last value.
"""
function compute_convergence_stats(
    records::Vector{RunRecord};
    checkpoints::Vector{Int} = Int[1, 5, 10, 25, 50, 75, 100],
)::Vector{ConvergenceStats}
    # Collect traces by (func, algorithm)
    trace_idx = Dict{Tuple{String, String}, Vector{Vector{Float64}}}()
    for r in records
        isempty(r.convergence_trace) && continue
        key = (r.func, r.algorithm)
        push!(get!(trace_idx, key, Vector{Float64}[]), r.convergence_trace)
    end
    isempty(trace_idx) && return ConvergenceStats[]

    result = ConvergenceStats[]
    for ((fn, algo), traces) in sort(collect(trace_idx))
        isempty(traces) && continue
        max_len = maximum(length(t) for t in traces)
        effective_cpts = filter(c -> c <= max_len, checkpoints)
        isempty(effective_cpts) && continue

        pts = ConvergencePoint[]
        for cp in effective_cpts
            vals = Float64[]
            for t in traces
                # Pad with last value if trace is shorter than checkpoint
                v = cp <= length(t) ? t[cp] : t[end]
                push!(vals, v)
            end
            push!(
                pts,
                ConvergencePoint(
                    cp,
                    median(vals),
                    mean(vals),
                    length(vals) > 1 ? std(vals) : 0.0,
                    minimum(vals),
                    length(vals),
                ),
            )
        end
        push!(result, ConvergenceStats(fn, algo, pts))
    end
    return result
end

function to_markdown_convergence(conv::Vector{ConvergenceStats}, meta::Dict)::String
    isempty(conv) && return ""

    dims = get(meta, "dims", "?")
    n_runs = get(meta, "n_runs", "?")
    givp_ver = get(meta, "givp_version", "?")

    lines = String[]
    push!(
        lines,
        "<!-- convergence tables generated by generate_report.jl — givp $givp_ver -->",
    )
    push!(lines, "")
    push!(lines, "## Convergence Curves")
    push!(lines, "")
    push!(lines, "> Objective value (best-so-far) at each outer iteration.")
    push!(lines, "> Median and mean over all seeds with trace data.")
    push!(lines, "")

    functions = unique(c.func for c in conv)
    for fn in functions
        fn_curves = filter(c -> c.func == fn, conv)
        isempty(fn_curves) && continue

        push!(lines, "### $fn")
        push!(lines, "")

        # Collect common checkpoints across all algorithms
        all_cpts = sort(unique(p.iteration for c in fn_curves for p in c.checkpoints))

        algos = [c.algorithm for c in fn_curves]
        header_parts = ["**Iteration**"]
        for a in algos
            push!(header_parts, "**$a** (median)")
            push!(header_parts, "**$a** (mean ± std)")
        end
        push!(lines, "| " * join(header_parts, " | ") * " |")
        sep = [":---:"]
        append!(sep, fill(":---:", 2 * length(algos)))
        push!(lines, "| " * join(sep, " | ") * " |")

        for cp in all_cpts
            row_parts = [string(cp)]
            for c in fn_curves
                pt = findfirst(p -> p.iteration == cp, c.checkpoints)
                if pt !== nothing
                    p = c.checkpoints[pt]
                    push!(row_parts, @sprintf("%.4e", p.median_val))
                    push!(row_parts, @sprintf("%.4e ± %.4e", p.mean_val, p.std_val))
                else
                    push!(row_parts, "—")
                    push!(row_parts, "—")
                end
            end
            push!(lines, "| " * join(row_parts, " | ") * " |")
        end
        push!(lines, "")
    end

    return join(lines, "\n")
end

function to_latex_convergence(conv::Vector{ConvergenceStats}, meta::Dict)::String
    isempty(conv) && return ""

    dims = get(meta, "dims", "?")
    n_runs = get(meta, "n_runs", "?")
    givp_ver = get(meta, "givp_version", "?")

    lines = String[]
    push!(lines, "% Convergence table generated by generate_report.jl")
    push!(lines, "% givp $givp_ver | n=$dims | $n_runs runs")
    push!(lines, "")

    functions = unique(c.func for c in conv)
    for fn in functions
        fn_curves = filter(c -> c.func == fn, conv)
        isempty(fn_curves) && continue

        algos = [c.algorithm for c in fn_curves]
        n_algos = length(algos)
        all_cpts = sort(unique(p.iteration for c in fn_curves for p in c.checkpoints))

        push!(lines, "\\begin{table}[htb]")
        push!(lines, "\\centering")
        push!(
            lines,
            "\\caption{Convergence of objective value (best-so-far) on $fn " *
            "(\$n=$dims\$, $n_runs seeds). Median and mean\$\\pm\$std at each outer iteration.}",
        )
        push!(lines, "\\label{tab:convergence_$(_latex_esc(lowercase(fn)))}")
        push!(lines, "\\begin{tabular}{r" * ("rr"^n_algos) * "}")
        push!(lines, "\\toprule")

        # Header rows: algorithm names spanning 2 cols each
        algo_cols =
            join(["\\multicolumn{2}{c}{\\texttt{$(_latex_esc(a))}}" for a in algos], " & ")
        push!(lines, "Iter. & $algo_cols \\\\")

        sub_header = join(fill("Median & Mean\$\\pm\$Std", n_algos), " & ")
        push!(lines, "\\cmidrule(lr){2-$(1 + 2*n_algos)}")
        push!(lines, " & $sub_header \\\\")
        push!(lines, "\\midrule")

        for cp in all_cpts
            row = [string(cp)]
            for c in fn_curves
                pt = findfirst(p -> p.iteration == cp, c.checkpoints)
                if pt !== nothing
                    p = c.checkpoints[pt]
                    push!(row, @sprintf("\$%.2e\$", p.median_val))
                    push!(row, @sprintf("\$%.2e\\pm%.2e\$", p.mean_val, p.std_val))
                else
                    push!(row, "—")
                    push!(row, "—")
                end
            end
            push!(lines, join(row, " & ") * " \\\\")
        end
        push!(lines, "\\bottomrule")
        push!(lines, "\\end{tabular}")
        push!(lines, "\\end{table}")
        push!(lines, "")
    end

    return join(lines, "\n")
end

# ── Formatters ────────────────────────────────────────────────────────────────

_fmt_sci(x::Float64) = @sprintf("%.4e", x)
_fmt_mean_std(m, s) = @sprintf("%.4e ± %.4e", m, s)

function print_console_summary(summary::Vector{SummaryRow}, wilcoxon::Vector{WilcoxonRow})
    pval_idx = Dict((w.func, w.algorithm) => w.pvalue for w in wilcoxon)
    sig_idx = Dict((w.func, w.algorithm) => w.significant for w in wilcoxon)

    functions = unique(r.func for r in summary)
    col = 16

    println()
    for fn in functions
        println("  ─── $fn ───")
        @printf(
            "  %-16s %14s %14s %14s %14s %10s %5s\n",
            "Algorithm",
            "Mean",
            "Std",
            "Best",
            "Median",
            "p-value",
            "Sig"
        )
        println("  " * "─"^(col + 14 * 4 + 10 + 5 + 6))
        for row in filter(r -> r.func == fn, summary)
            pval_str =
                haskey(pval_idx, (fn, row.algorithm)) ?
                @sprintf("%.4f", pval_idx[(fn, row.algorithm)]) : "  ref"
            sig_str = get(sig_idx, (fn, row.algorithm), false) ? "★" : "—"
            @printf(
                "  %-16s %14s %14s %14s %14s %10s %5s\n",
                row.algorithm,
                _fmt_sci(row.mean_val),
                _fmt_sci(row.std_val),
                _fmt_sci(row.best),
                _fmt_sci(row.median_val),
                pval_str,
                sig_str
            )
        end
        println()
    end
end

function to_markdown(
    summary::Vector{SummaryRow},
    wilcoxon::Vector{WilcoxonRow},
    meta::Dict,
)::String
    pval_idx = Dict((w.func, w.algorithm) => w.pvalue for w in wilcoxon)
    sig_idx = Dict((w.func, w.algorithm) => w.significant for w in wilcoxon)

    dims = get(meta, "dims", "?")
    n_runs = get(meta, "n_runs", "?")
    givp_ver = get(meta, "givp_version", "?")
    reference = get(meta, "reference_algorithm", "GIVP-full")

    lines = String[]
    push!(
        lines,
        "<!-- generated by generate_report.jl — givp $givp_ver, n=$dims, $n_runs runs -->",
    )
    push!(lines, "")

    functions = unique(r.func for r in summary)
    for fn in functions
        push!(lines, "### $fn")
        push!(lines, "")
        push!(
            lines,
            "| Algorithm | Mean ± Std | Best | Median | Worst | NFev (mean) | p-value | Sig |",
        )
        push!(
            lines,
            "|-----------|------------|------|--------|-------|-------------|---------|-----|",
        )
        for row in filter(r -> r.func == fn, summary)
            pval_str =
                haskey(pval_idx, (fn, row.algorithm)) ?
                @sprintf("%.4f", pval_idx[(fn, row.algorithm)]) : "*(ref)*"
            sig_str = get(sig_idx, (fn, row.algorithm), false) ? "★" : "—"
            push!(
                lines,
                "| $(row.algorithm) " *
                "| $(_fmt_mean_std(row.mean_val, row.std_val)) " *
                "| $(_fmt_sci(row.best)) " *
                "| $(_fmt_sci(row.median_val)) " *
                "| $(_fmt_sci(row.worst)) " *
                "| $(Int(round(row.nfev_mean))) " *
                "| $pval_str " *
                "| $sig_str |",
            )
        end
        push!(lines, "")
    end

    push!(
        lines,
        "★ p < 0.05 (Wilcoxon signed-rank test, two-sided) — significantly different from *$reference*.",
    )
    return join(lines, "\n")
end

function _latex_esc(s::String)::String
    s = replace(s, "_" => "\\_")
    s = replace(s, "&" => "\\&")
    s = replace(s, "%" => "\\%")
    return s
end

function to_latex(
    summary::Vector{SummaryRow},
    wilcoxon::Vector{WilcoxonRow},
    meta::Dict,
)::String
    sig_idx = Dict((w.func, w.algorithm) => w.significant for w in wilcoxon)

    dims = get(meta, "dims", "?")
    n_runs = get(meta, "n_runs", "?")
    givp_ver = get(meta, "givp_version", "?")
    algorithms = unique(r.algorithm for r in summary)
    reference = isempty(algorithms) ? "GIVP-full" : algorithms[1]

    lines = String[]
    push!(lines, "% Generated by generate_report.jl")
    push!(lines, "% givp $givp_ver | n=$dims | $n_runs independent runs per cell")
    push!(lines, "")
    push!(lines, "\\begin{table}[htb]")
    push!(lines, "\\centering")
    push!(
        lines,
        "\\caption{Comparison of optimization algorithms on standard benchmark functions " *
        "(\$n=$dims\$ variables, $n_runs independent runs per cell). " *
        "Mean \$\\pm\$ std over all runs. " *
        "\$^\\star\$ denotes \$p < 0.05\$ (Wilcoxon signed-rank, two-sided) " *
        "vs.\\ \\texttt{$(_latex_esc(reference))}.}}",
    )
    push!(lines, "\\label{tab:benchmark_comparison_julia}")
    push!(lines, "\\begin{tabular}{ll" * "r"^4 * "}")
    push!(lines, "\\toprule")
    push!(lines, "Function & Algorithm & Mean & Std & Best & Median \\\\")
    push!(lines, "\\midrule")

    functions = unique(r.func for r in summary)
    for fn in functions
        fn_rows = filter(r -> r.func == fn, summary)
        for (i, row) in enumerate(fn_rows)
            fn_cell = i == 1 ? "\\multirow{$(length(fn_rows))}{*}{$fn}" : ""
            sig = get(sig_idx, (fn, row.algorithm), false) ? "\$^\\star\$" : ""
            push!(
                lines,
                "  $fn_cell & $(_latex_esc(row.algorithm))$sig " *
                "& \$$(_fmt_sci(row.mean_val))\$ " *
                "& \$$(_fmt_sci(row.std_val))\$ " *
                "& \$$(_fmt_sci(row.best))\$ " *
                "& \$$(_fmt_sci(row.median_val))\$ \\\\",
            )
        end
        push!(lines, "\\midrule")
    end

    # Replace last \midrule with \bottomrule
    if !isempty(lines) && lines[end] == "\\midrule"
        lines[end] = "\\bottomrule"
    end
    push!(lines, "\\end{tabular}")

    # Wilcoxon sub-table
    if !isempty(wilcoxon)
        push!(lines, "")
        push!(lines, "\\medskip")
        push!(lines, "\\begin{tabular}{llrrr}")
        push!(lines, "\\toprule")
        push!(lines, "Function & Challenger & \$W\$ & \$p\$-value & \$r\$ (effect) \\\\")
        push!(lines, "\\midrule")
        for w in wilcoxon
            sig = w.significant ? "\$^\\star\$" : ""
            push!(
                lines,
                "  $(w.func) & $(_latex_esc(w.algorithm))$sig " *
                "& $(@sprintf("%.1f", w.stat)) " *
                "& $(@sprintf("%.4f", w.pvalue)) " *
                "& $(@sprintf("%.3f", w.effect_r)) \\\\",
            )
        end
        push!(lines, "\\bottomrule")
        push!(lines, "\\end{tabular}")
        push!(
            lines,
            "\\\\ \\footnotesize{\$W\$: Wilcoxon test statistic; " *
            "\$r\$: rank-biserial correlation (effect size); " *
            "\$^\\star\$: \$p < 0.05\$.}",
        )
    end

    push!(lines, "\\end{table}")
    return join(lines, "\n")
end

# ── CLI ───────────────────────────────────────────────────────────────────────

function parse_cli_args()
    args = ARGS
    params = Dict{String, Any}(
        "input" => "",
        "output-dir" => "",
        "format" => "both",
        "reference" => "",
        "alpha" => 0.05,
        "convergence" => false,
        "checkpoints" => Int[1, 5, 10, 25, 50, 75, 100],
        "verbose" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            println("""
Usage: julia --project=julia julia/benchmarks/generate_report.jl [OPTIONS]

Options:
  --input PATH          JSON file produced by run_literature_comparison.jl (required)
  --output-dir DIR      Directory for output files (default: same as --input)
  --format STR          markdown | latex | both  (default: both)
  --reference ALGO      Reference algorithm for Wilcoxon tests (default: first in JSON)
  --alpha FLOAT         Significance level (default: 0.05)
  --convergence         Output convergence curve tables (requires --traces in experiment)
  --checkpoints INTS    Space-separated iteration checkpoints for convergence tables
                          (default: 1 5 10 25 50 75 100)
  --verbose             Verbose output
  --help                Show this message
""")
            exit(0)
        elseif arg == "--input" && i < length(args)
            params["input"] = args[i + 1]
            i += 2
        elseif arg == "--output-dir" && i < length(args)
            params["output-dir"] = args[i + 1]
            i += 2
        elseif arg == "--format" && i < length(args)
            params["format"] = args[i + 1]
            i += 2
        elseif arg == "--reference" && i < length(args)
            params["reference"] = args[i + 1]
            i += 2
        elseif arg == "--alpha" && i < length(args)
            params["alpha"] = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--convergence"
            params["convergence"] = true
            i += 1
        elseif arg == "--checkpoints" && i < length(args)
            cpts = Int[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(cpts, parse(Int, args[i]))
                i += 1
            end
            isempty(cpts) || (params["checkpoints"] = sort(cpts))
        elseif arg == "--verbose"
            params["verbose"] = true
            i += 1
        else
            @warn "Unknown argument: $arg (ignored)"
            i += 1
        end
    end
    return params
end

function main()
    params = parse_cli_args()

    if isempty(params["input"])
        @error "--input is required"
        exit(1)
    end
    if !isfile(params["input"])
        @error "Input file not found: $(params["input"])"
        exit(1)
    end

    meta, records = load_results(params["input"])
    isempty(records) && (@error "No records found in $(params["input"])"; exit(1))

    input_path = params["input"]
    out_dir = isempty(params["output-dir"]) ? dirname(input_path) : params["output-dir"]
    isempty(out_dir) && (out_dir = ".")
    stem = splitext(basename(input_path))[1]

    params["verbose"] && @info "Loaded $(length(records)) records"

    algorithms = unique(r.algorithm for r in records)
    reference =
        isempty(params["reference"]) ?
        (isempty(algorithms) ? "GIVP-full" : first(algorithms)) : params["reference"]

    if reference ∉ Set(r.algorithm for r in records)
        @error "Reference algorithm \"$reference\" not found. Available: $(join(algorithms, ", "))"
        exit(1)
    end

    @info "Report: $(params["input"])"
    @info "  algorithms : $(join(algorithms, ", "))"
    @info "  functions  : $(join(unique(r.func for r in records), ", "))"
    @info "  dims       : $(get(meta, "dims", "?"))"
    @info "  n_runs     : $(get(meta, "n_runs", "?"))"
    @info "  reference  : $reference"
    @info "  alpha      : $(params["alpha"])"
    @info ""

    summary = build_summary(records)
    meta["reference_algorithm"] = reference

    wrows = compute_wilcoxon(records, reference; alpha = params["alpha"])
    n_sig = count(w -> w.significant, wrows)
    params["verbose"] && @info "Wilcoxon: $n_sig significant out of $(length(wrows)) pairs"

    print_console_summary(summary, wrows)

    # Auto-enable convergence if traces are present in data
    has_traces = any(!isempty(r.convergence_trace) for r in records)
    do_conv = params["convergence"] || has_traces
    conv_stats =
        do_conv ? compute_convergence_stats(records; checkpoints = params["checkpoints"]) :
        ConvergenceStats[]

    if has_traces && !params["convergence"]
        @info "Convergence traces detected — including convergence tables automatically."
        @info "  (suppress with: remove --traces from run_literature_comparison.jl call)"
    end
    if params["convergence"] && !has_traces
        @warn "--convergence requested but no traces found in data. " *
              "Re-run the experiment with --traces to collect convergence data."
    end

    fmt = params["format"]

    if fmt in ("markdown", "both")
        md_path = joinpath(out_dir, "$(stem)_report.md")
        open(md_path, "w") do io
            write(io, to_markdown(summary, wrows, meta))
            if !isempty(conv_stats)
                write(io, "\n\n")
                write(io, to_markdown_convergence(conv_stats, meta))
            end
        end
        @info "Markdown → $md_path"
    end

    if fmt in ("latex", "both")
        tex_path = joinpath(out_dir, "$(stem)_report.tex")
        open(tex_path, "w") do io
            write(io, to_latex(summary, wrows, meta))
            if !isempty(conv_stats)
                write(io, "\n\n")
                write(io, to_latex_convergence(conv_stats, meta))
            end
        end
        @info "LaTeX    → $tex_path"
    end
end

main()
