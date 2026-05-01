# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Reproducible multi-run benchmark experiment for GIVPOptimizer vs baselines.

Port of python/benchmarks/run_literature_comparison.py.
Output JSON is compatible with python/benchmarks/generate_report.py.

Usage (from repo root):

    julia --project=julia julia/benchmarks/run_literature_comparison.jl
    julia --project=julia julia/benchmarks/run_literature_comparison.jl \\
        --n-runs 30 --dims 10 --output results_julia.json --verbose
    julia --project=julia julia/benchmarks/run_literature_comparison.jl \\
        --n-runs 2 --dims 5 --algorithms GIVP-full GRASP-only --resume

References
----------
- Feo, T.A. & Resende, M.G.C. (1995). Greedy randomized adaptive search
  procedures. Journal of Global Optimization, 6, 109-133.
- De Jong, K.A. (1975). PhD thesis, University of Michigan.
- Rosenbrock, H.H. (1960). The Computer Journal, 3(3), 175-184.
- Rastrigin, L.A. (1974). Systems of Extremal Control. Nauka, Moscow.
- Ackley, D.H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer.
- Griewank, A.O. (1981). Journal of Optimization Theory and Applications, 34(1).
- Schwefel, H.P. (1981). Numerical Optimization of Computer Models. Wiley.
"""

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GIVPOptimizer
using JSON
using Printf
using Statistics
using Dates
using Random

# ── Optional external baselines (BlackBoxOptim.jl) ──────────────────────────
const _BBO_AVAILABLE = try
    @eval using BlackBoxOptim
    true
catch
    false
end

_BBO_AVAILABLE || @info(
    "BlackBoxOptim.jl not installed — BBO-* algorithms unavailable. " *
    "Install with: julia -e 'using Pkg; Pkg.add(\"BlackBoxOptim\")'"
)

# ── Benchmark functions ───────────────────────────────────────────────────────

function sphere(x::Vector{Float64})::Float64
    return sum(xi^2 for xi in x)
end

function rosenbrock(x::Vector{Float64})::Float64
    n = length(x)
    n < 2 && return 0.0
    return sum(100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2 for i in 1:(n - 1))
end

function rastrigin(x::Vector{Float64})::Float64
    n = length(x)
    return 10.0 * n + sum(xi^2 - 10.0 * cos(2π * xi) for xi in x)
end

function ackley(x::Vector{Float64})::Float64
    n = length(x)
    a, b, c = 20.0, 0.2, 2π
    sq = sqrt(sum(xi^2 for xi in x) / n)
    cs = sum(cos(c * xi) for xi in x) / n
    return -a * exp(-b * sq) - exp(cs) + a + exp(1.0)
end

function griewank(x::Vector{Float64})::Float64
    n = length(x)
    sum_sq = sum(xi^2 for xi in x) / 4000.0
    prod_cos = prod(cos(x[i] / sqrt(Float64(i))) for i in 1:n)
    return 1.0 + sum_sq - prod_cos
end

function schwefel(x::Vector{Float64})::Float64
    n = length(x)
    return 418.9829 * n - sum(xi * sin(sqrt(abs(xi))) for xi in x)
end

# ── Problem registry ──────────────────────────────────────────────────────────

const PROBLEM_REGISTRY = Dict(
    "Sphere" => (
        func = sphere,
        bounds_factory = n -> Tuple{Float64, Float64}[(-5.12, 5.12) for _ in 1:n],
        optimum = 0.0,
        reference = "De Jong (1975)",
    ),
    "Rosenbrock" => (
        func = rosenbrock,
        bounds_factory = n -> Tuple{Float64, Float64}[(-5.0, 10.0) for _ in 1:n],
        optimum = 0.0,
        reference = "Rosenbrock (1960)",
    ),
    "Rastrigin" => (
        func = rastrigin,
        bounds_factory = n -> Tuple{Float64, Float64}[(-5.12, 5.12) for _ in 1:n],
        optimum = 0.0,
        reference = "Rastrigin (1974)",
    ),
    "Ackley" => (
        func = ackley,
        bounds_factory = n -> Tuple{Float64, Float64}[(-32.768, 32.768) for _ in 1:n],
        optimum = 0.0,
        reference = "Ackley (1987)",
    ),
    "Griewank" => (
        func = griewank,
        bounds_factory = n -> Tuple{Float64, Float64}[(-600.0, 600.0) for _ in 1:n],
        optimum = 0.0,
        reference = "Griewank (1981)",
    ),
    "Schwefel" => (
        func = schwefel,
        bounds_factory = n -> Tuple{Float64, Float64}[(-500.0, 500.0) for _ in 1:n],
        optimum = 0.0,
        reference = "Schwefel (1981)",
    ),
)

const FUNCTION_ORDER =
    ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank", "Schwefel"]

const ALGO_DESCRIPTIONS = Dict(
    "GIVP-full" => "GRASP-ILS-VND-PR — full hybrid pipeline (this work)",
    "GRASP-only" => "GRASP-only baseline (Feo & Resende 1995)",
    "BBO-DE" => "BlackBoxOptim.jl — Differential Evolution de_rand_1_bin (Price et al. 1997)",
    "BBO-XNES" => "BlackBoxOptim.jl — Exp. Natural Evo. Strategies (Glasmachers et al. 2010)",
)

const _GIVP_ALGORITHMS = Set(["GIVP-full", "GRASP-only"])
const _BBO_METHOD_MAP = Dict("BBO-DE" => :de_rand_1_bin, "BBO-XNES" => :xnes)

# ── Config builders ───────────────────────────────────────────────────────────

function config_givp_full(max_iter::Int, time_limit::Float64)::GIVPConfig
    return GIVPConfig(
        max_iterations = max_iter,
        alpha = 0.12,
        adaptive_alpha = true,
        alpha_min = 0.08,
        alpha_max = 0.18,
        vnd_iterations = 200,
        ils_iterations = 10,
        perturbation_strength = 4,
        use_elite_pool = true,
        elite_size = 7,
        path_relink_frequency = 8,
        use_cache = true,
        cache_size = 10000,
        early_stop_threshold = 80,
        use_convergence_monitor = true,
        time_limit = time_limit,
    )
end

function config_grasp_only(max_iter::Int, time_limit::Float64)::GIVPConfig
    return GIVPConfig(
        max_iterations = max_iter,
        alpha = 0.12,
        adaptive_alpha = false,
        vnd_iterations = 1,
        ils_iterations = 1,
        perturbation_strength = 0,
        use_elite_pool = false,
        use_convergence_monitor = false,
        use_cache = true,
        cache_size = 10000,
        early_stop_threshold = max_iter,
        time_limit = time_limit,
    )
end

# ── External baseline (BlackBoxOptim) ───────────────────────────────────────

"""
    run_single_bbo(method, func, bounds, seed, max_iter, time_limit; verbose)

Run a single BlackBoxOptim.jl trial.  `method` must be a valid BBO method symbol,
e.g. `:de_rand_1_bin` or `:xnes`.
NFev budget is set to `max(10_000, max_iter * 200)` to be comparable with GIVP.
"""
function run_single_bbo(
    method::Symbol,
    func::Function,
    bounds::Vector{Tuple{Float64, Float64}},
    seed::Int,
    max_iter::Int,
    time_limit::Float64;
    verbose::Bool = false,
)::Dict{String, Any}
    _BBO_AVAILABLE || error(
        "BlackBoxOptim.jl is not installed.\n" *
        "Install with: julia -e 'using Pkg; Pkg.add(\"BlackBoxOptim\")'",
    )

    nfev = Ref(0)
    # Wrap to count evaluations; BBO passes AbstractVector, so collect to Float64
    wrapped = x -> begin
        nfev[] += 1
        func(collect(Float64, x))
    end

    # Match GIVP's typical evaluation budget
    max_steps = max(10_000, max_iter * 200)

    # Seed Julia's global RNG — BBO uses rand() internally
    Random.seed!(seed)

    t0 = time()
    bbo_result = try
        kwargs = Dict{Symbol, Any}(
            :SearchRange => bounds,
            :MaxSteps => max_steps,
            :Method => method,
            :TraceMode => :silent,
        )
        time_limit > 0 && (kwargs[:MaxTime] = time_limit)
        bboptimize(wrapped; kwargs...)
    catch e
        elapsed = time() - t0
        @warn "BBO $(method) run failed (seed=$seed): $e"
        return Dict{String, Any}(
            "seed" => seed,
            "fun" => Inf,
            "nit" => 0,
            "nfev" => nfev[],
            "time_s" => elapsed,
        )
    end
    elapsed = time() - t0

    best_val = best_fitness(bbo_result)
    verbose && @info @sprintf(
        "  %-10s seed=%2d → fun=%12.6f  nfev=%6d  t=%.2fs",
        string(method),
        seed,
        best_val,
        nfev[],
        elapsed
    )

    return Dict{String, Any}(
        "seed" => seed,
        "fun" => best_val,
        "nit" => 0,    # BBO has no outer iteration count
        "nfev" => nfev[],
        "time_s" => elapsed,
    )
end

# ── Single run ────────────────────────────────────────────────────────────────

function run_single(
    algo::String,
    func::Function,
    bounds::Vector{Tuple{Float64, Float64}},
    seed::Int,
    max_iter::Int,
    time_limit::Float64;
    verbose::Bool = false,
    traces::Bool = false,
)::Dict{String, Any}
    # ── External BBO baseline ───────────────────────────────────────────────
    if algo ∈ keys(_BBO_METHOD_MAP)
        method = _BBO_METHOD_MAP[algo]
        rec = run_single_bbo(method, func, bounds, seed, max_iter, time_limit; verbose)
        rec["algorithm"] = algo
        return rec
    end

    # ── GIVP algorithms ─────────────────────────────────────────────────────
    cfg =
        algo == "GIVP-full" ? config_givp_full(max_iter, time_limit) :
        config_grasp_only(max_iter, time_limit)

    trace_data = Float64[]
    cb = traces ? ((iter, cost, _sol) -> push!(trace_data, cost)) : nothing

    t0 = time()
    result = givp(func, bounds; config = cfg, seed = seed, iteration_callback = cb)
    elapsed = time() - t0

    verbose && @info @sprintf(
        "  %-10s seed=%2d → fun=%12.6f  nit=%4d  nfev=%6d  t=%.2fs",
        algo,
        seed,
        result.fun,
        result.nit,
        result.nfev,
        elapsed
    )

    rec = Dict{String, Any}(
        "algorithm" => algo,
        "seed" => seed,
        "fun" => result.fun,
        "nit" => result.nit,
        "nfev" => result.nfev,
        "time_s" => elapsed,
    )
    if traces && !isempty(trace_data)
        rec["convergence_trace"] = trace_data
    end
    return rec
end

# ── Summary statistics ────────────────────────────────────────────────────────

function build_summary(records::Vector{Dict{String, Any}})::Vector{Dict{String, Any}}
    groups = Dict{Tuple{String, String}, Vector{Float64}}()
    for r in records
        key = (r["function"], r["algorithm"])
        push!(get!(groups, key, Float64[]), r["fun"])
    end

    summary = Dict{String, Any}[]
    for ((fname, algo), vals) in sort(collect(groups))
        push!(
            summary,
            Dict{String, Any}(
                "function" => fname,
                "algorithm" => algo,
                "n_runs" => length(vals),
                "mean" => mean(vals),
                "std" => length(vals) > 1 ? std(vals) : 0.0,
                "best" => minimum(vals),
                "median" => median(vals),
                "worst" => maximum(vals),
                "nfev_mean" => mean(
                    Float64[
                        r["nfev"] for
                        r in records if r["function"] == fname && r["algorithm"] == algo
                    ],
                ),
            ),
        )
    end
    return summary
end

# ── IO ────────────────────────────────────────────────────────────────────────

function save_results(
    records::Vector{Dict{String, Any}},
    output::String,
    algorithms::Vector{String},
    functions::Vector{String},
    n_runs::Int,
    dims::Int,
)
    data = Dict{String, Any}(
        "metadata" => Dict{String, Any}(
            "julia_version" => string(VERSION),
            "givp_version" => string(pkgversion(GIVPOptimizer)),
            "dims" => dims,
            "n_runs" => n_runs,
            "algorithms" => algorithms,
            "functions" => functions,
            "generated_at" => string(now()),
        ),
        "summary" => build_summary(records),
        "records" => records,
    )
    open(output, "w") do io
        JSON.print(io, data, 2)
    end
end

# ── Experiment ────────────────────────────────────────────────────────────────

function run_experiment(;
    algorithms::Vector{String} = ["GIVP-full", "GRASP-only"],
    functions::Vector{String} = FUNCTION_ORDER,
    n_runs::Int = 30,
    dims::Int = 10,
    max_iter::Int = 100,
    time_limit::Float64 = 0.0,
    output::String = "results_julia.json",
    resume::Bool = false,
    verbose::Bool = false,
    traces::Bool = false,
)
    # Load existing results if resuming
    completed = Set{Tuple{String, String, Int}}()
    existing_records = Dict{String, Any}[]
    if resume && isfile(output)
        data = JSON.parsefile(output)
        existing_records = get(Vector{Dict{String, Any}}, data, "records")
        for r in existing_records
            push!(completed, (r["algorithm"], r["function"], r["seed"]))
        end
        @info "Resuming: $(length(completed)) runs already completed, loaded from $output"
    end

    records = copy(existing_records)
    total = length(algorithms) * length(functions) * n_runs
    done = length(completed)

    @info @sprintf(
        "Experiment: %d algorithms × %d functions × %d runs = %d total | dims=%d",
        length(algorithms),
        length(functions),
        n_runs,
        total,
        dims,
    )

    for algo in algorithms, fname in functions
        prob = PROBLEM_REGISTRY[fname]
        bounds = prob.bounds_factory(dims)
        verbose && @info @sprintf("Running %-10s on %-12s (%d-D)...", algo, fname, dims)

        for seed in 0:(n_runs - 1)
            (algo, fname, seed) in completed && continue

            record = run_single(
                algo,
                prob.func,
                bounds,
                seed,
                max_iter,
                time_limit;
                verbose = verbose,
                traces = traces,
            )
            record["function"] = fname
            push!(records, record)
            done += 1

            # Checkpoint after every run so crashes don't lose progress
            save_results(records, output, algorithms, functions, n_runs, dims)
        end

        if !verbose
            algo_fvals = [
                r["fun"] for
                r in records if r["algorithm"] == algo && r["function"] == fname
            ]
            isempty(algo_fvals) || @info @sprintf(
                "  %-10s  %-12s  mean=%.4f  best=%.4f  [%d runs]",
                algo,
                fname,
                mean(algo_fvals),
                minimum(algo_fvals),
                length(algo_fvals),
            )
        end
    end

    @info @sprintf("Experiment complete: %d records → %s", length(records), output)
    return records
end

# ── CLI ───────────────────────────────────────────────────────────────────────

function parse_cli_args()
    args = ARGS
    params = Dict{String, Any}(
        "n-runs" => 30,
        "dims" => 10,
        "max-iter" => 100,
        "time-limit" => 0.0,
        "output" => "results_julia.json",
        "algorithms" => ["GIVP-full", "GRASP-only"],
        "functions" => copy(FUNCTION_ORDER),
        "resume" => false,
        "verbose" => false,
        "traces" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            println("""
Usage: julia --project=julia julia/benchmarks/run_literature_comparison.jl [OPTIONS]

Options:
  --n-runs INT        Independent runs per (algorithm, function) pair (default: 30)
  --dims INT          Problem dimensionality (default: 10)
  --max-iter INT      Maximum GRASP iterations per run (default: 100)
  --time-limit FLOAT  Wall-clock seconds per run, 0 = unlimited (default: 0.0)
  --algorithms LIST   Space-separated list; available:
                        GIVP-full   — full GRASP-ILS-VND-PR pipeline (this work)
                        GRASP-only  — GRASP construction baseline
                        BBO-DE      — BlackBoxOptim.jl Differential Evolution
                        BBO-XNES    — BlackBoxOptim.jl Natural Evo. Strategies
                      (default: GIVP-full GRASP-only)
  --functions LIST    Space-separated subset of: $(join(FUNCTION_ORDER, " "))
  --output PATH       Output JSON file (default: results_julia.json)
  --resume            Resume from existing output, skipping completed runs
  --traces            Record per-iteration convergence trace (GIVP only)
  --verbose           Show per-run progress
  --help              Show this message
""")
            exit(0)
        elseif arg == "--n-runs" && i < length(args)
            params["n-runs"] = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--dims" && i < length(args)
            params["dims"] = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--max-iter" && i < length(args)
            params["max-iter"] = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--time-limit" && i < length(args)
            params["time-limit"] = parse(Float64, args[i + 1])
            i += 2
        elseif arg == "--output" && i < length(args)
            params["output"] = args[i + 1]
            i += 2
        elseif arg == "--algorithms" && i < length(args)
            algos = String[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(algos, args[i])
                i += 1
            end
            params["algorithms"] = algos
        elseif arg == "--functions" && i < length(args)
            funcs = String[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(funcs, args[i])
                i += 1
            end
            params["functions"] = funcs
        elseif arg == "--resume"
            params["resume"] = true
            i += 1
        elseif arg == "--traces"
            params["traces"] = true
            i += 1
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

    # Validate algorithms
    valid_algos = Set(keys(ALGO_DESCRIPTIONS))
    for algo in params["algorithms"]
        if algo ∉ valid_algos
            @error "Unknown algorithm: \"$algo\". Valid options: $(join(sort(collect(valid_algos)), ", "))"
            exit(1)
        end
    end

    # Validate functions
    valid_funcs = Set(FUNCTION_ORDER)
    for fname in params["functions"]
        if fname ∉ valid_funcs
            @error "Unknown function: \"$fname\". Valid options: $(join(FUNCTION_ORDER, ", "))"
            exit(1)
        end
    end

    run_experiment(
        algorithms = params["algorithms"],
        functions = params["functions"],
        n_runs = params["n-runs"],
        dims = params["dims"],
        max_iter = params["max-iter"],
        time_limit = params["time-limit"],
        output = params["output"],
        resume = params["resume"],
        verbose = params["verbose"],
        traces = params["traces"],
    )
end

main()
