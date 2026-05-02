# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Crash-finder fuzzer for GIVPOptimizer.jl.

Exercises the API with thousands of random and adversarial inputs to detect:
  - Unhandled exceptions (anything that is not a GivpError subtype)
  - NaN/Inf leaking into the solution vector (`r.x`)
  - Out-of-bounds solutions (r.x not in declared bounds)
  - Incorrect termination: r.success == true but r.fun == Inf (or vice versa)
  - Stack overflows / memory errors from degenerate configs
  - Unexpected behaviour on edge-case functions (constant, NaN-returning, throwing)

Exit codes:
  0 — all trials passed
  1 — at least one unexpected failure found (details on stderr)
  2 — argument error

Usage (from repo root):
    julia --project=julia julia/fuzz/fuzz_givp.jl
    julia --project=julia julia/fuzz/fuzz_givp.jl --n-trials 2000 --seed 1337 --verbose
    julia --project=julia julia/fuzz/fuzz_givp.jl --n-trials 200 --timeout 30
"""

# Bootstrap: only activate/instantiate when run standalone (not via --project=julia)
let active = Base.active_project()
    if isnothing(active) || !occursin(joinpath("julia", "Project.toml"), active)
        import Pkg
        Pkg.activate(joinpath(@__DIR__, ".."))
        Pkg.instantiate()
    end
end

using GIVPOptimizer
using Printf
using Random

# ── Fuzz configuration ────────────────────────────────────────────────────────

struct FuzzConfig
    n_trials::Int
    seed::Int
    timeout_s::Float64   # wall-clock limit for the entire fuzz session
    verbose::Bool
end

# ── Random generators ─────────────────────────────────────────────────────────

function rand_bounds(rng::AbstractRNG, ndim::Int)::Vector{Tuple{Float64, Float64}}
    [(lo, lo + abs(randn(rng)) * 10 + 1e-6) for lo in randn(rng, ndim) .* 50]
end

function rand_config(rng::AbstractRNG)::GIVPConfig
    max_it = rand(rng, 1:20)
    vnd_it = rand(rng, 1:50)
    ils_it = rand(rng, 1:10)
    GIVPConfig(
        max_iterations = max_it,
        vnd_iterations = vnd_it,
        ils_iterations = ils_it,
        perturbation_strength = rand(rng, 0:5),
        use_elite_pool = rand(rng, Bool),
        elite_size = rand(rng, 1:10),
        path_relink_frequency = rand(rng, 1:20),
        adaptive_alpha = rand(rng, Bool),
        alpha = rand(rng) * 0.4,
        alpha_min = rand(rng) * 0.1,
        alpha_max = 0.1 + rand(rng) * 0.4,
        num_candidates_per_step = rand(rng, 1:30),
        use_cache = rand(rng, Bool),
        cache_size = rand(rng, 100:5000),
        early_stop_threshold = rand(rng, 10:100),
        use_convergence_monitor = rand(rng, Bool),
        time_limit = 0.0,
    )
end

# ── Objective function zoo ────────────────────────────────────────────────────

sphere(x) = sum(xi^2 for xi in x)
neg_sphere(x) = -sphere(x)
constant_zero(x) = 0.0
constant_inf(x) = Inf
nan_func(x) = NaN
throwing_func(x) = (x[1] > 0 ? error("deliberate error") : 0.0)
mixed_func(x) = iseven(length(x)) ? sphere(x) : Inf
noisy_func(x) = sphere(x) + randn()   # non-deterministic (seed-breaking)

const FUNC_ZOO = [
    ("sphere", sphere, false),   # (name, func, expect_infeasible)
    ("neg_sphere", neg_sphere, false),
    ("constant_zero", constant_zero, false),
    ("constant_inf", constant_inf, true),    # all evaluations return Inf
    ("nan_func", nan_func, true),    # NaN must be treated as Inf
    ("throwing_func", throwing_func, false),   # exceptions must be caught
    ("mixed_func", mixed_func, false),
    ("noisy_func", noisy_func, false),
]

# ── Oracle ────────────────────────────────────────────────────────────────────

"""
    check_result(r, bounds, expect_infeasible) -> (passed::Bool, reason::String)

Verify invariants that must always hold for any OptimizeResult.
"""
function check_result(
    r::OptimizeResult,
    bounds::Vector{Tuple{Float64, Float64}},
    expect_infeasible::Bool,
)::Tuple{Bool, String}
    # I1: success ↔ isfinite(fun)
    if r.success != isfinite(r.fun)
        return false,
        "I1 violated: success=$(r.success) but isfinite(fun)=$(isfinite(r.fun))"
    end

    # I2: solution length must match dimensionality
    if length(r.x) != length(bounds)
        return false, "I2 violated: length(x)=$(length(r.x)) ≠ ndim=$(length(bounds))"
    end

    # I3: solution must be within bounds (when finite)
    if isfinite(r.fun)
        for (i, (lo, hi)) in enumerate(bounds)
            if r.x[i] < lo - 1e-9 || r.x[i] > hi + 1e-9
                return false, "I3 violated: x[$i]=$(r.x[i]) ∉ [$lo, $hi]"
            end
        end
    end

    # I4: nfev must be positive
    if r.nfev <= 0
        return false, "I4 violated: nfev=$(r.nfev) must be > 0"
    end

    # I5: no NaN in solution
    if any(isnan, r.x)
        return false, "I5 violated: NaN found in r.x"
    end

    # I6: message must be non-empty
    if isempty(r.message)
        return false, "I6 violated: message is empty"
    end

    return true, ""
end

# ── Fuzz trial ────────────────────────────────────────────────────────────────

struct FuzzFailure
    trial::Int
    category::String
    func_name::String
    ndim::Int
    seed::Int
    error::String
end

function run_trial(trial::Int, rng::AbstractRNG, verbose::Bool)::Union{Nothing, FuzzFailure}
    # Pick a random dimension (1 to 10)
    ndim = rand(rng, 1:10)
    bounds = rand_bounds(rng, ndim)
    cfg = rand_config(rng)
    cfg.integer_split = rand(rng, Bool) ? ndim : rand(rng, 0:ndim)
    seed = rand(rng, 0:9999)

    fn_idx = rand(rng, 1:length(FUNC_ZOO))
    fn_name, fn, expect_inf = FUNC_ZOO[fn_idx]

    verbose && @info @sprintf(
        "Trial %4d: %-15s ndim=%2d seed=%4d max_it=%3d",
        trial,
        fn_name,
        ndim,
        seed,
        cfg.max_iterations
    )

    # ── Category 1: valid call must not throw unexpected exception ────────────
    result = try
        givp(fn, bounds; config = cfg, seed = seed)
    catch e
        if e isa GivpError
            # Graceful, expected error — only allowed for truly degenerate inputs
            # (InvalidBoundsError, InvalidConfigError). Since we generate valid
            # bounds and config, this should not happen. Flag it.
            return FuzzFailure(
                trial,
                "unexpected_givp_error",
                fn_name,
                ndim,
                seed,
                sprint(showerror, e),
            )
        else
            return FuzzFailure(
                trial,
                "unexpected_exception",
                fn_name,
                ndim,
                seed,
                sprint(showerror, e),
            )
        end
    end

    # ── Category 2: oracle invariants ────────────────────────────────────────
    passed, reason = check_result(result, bounds, expect_inf)
    if !passed
        return FuzzFailure(trial, "invariant_violation", fn_name, ndim, seed, reason)
    end

    return nothing   # trial passed
end

# ── Invalid-input fuzz (error taxonomy) ──────────────────────────────────────

"""
Run trials with *deliberately invalid* inputs.
The only acceptable outcome is a subtype of GivpError; anything else is a bug.
"""
function run_invalid_input_trials(rng::AbstractRNG, verbose::Bool)::Vector{FuzzFailure}
    failures = FuzzFailure[]

    # Empty bounds
    for seed in 0:4
        try
            givp(sphere, Tuple{Float64, Float64}[]; config = GIVPConfig(), seed = seed)
            push!(
                failures,
                FuzzFailure(
                    -1,
                    "empty_bounds_no_error",
                    "sphere",
                    0,
                    seed,
                    "empty bounds should raise InvalidBoundsError",
                ),
            )
        catch e
            e isa InvalidBoundsError || push!(
                failures,
                FuzzFailure(
                    -1,
                    "wrong_exception_type",
                    "sphere",
                    0,
                    seed,
                    sprint(showerror, e),
                ),
            )
        end
    end

    # Inverted bounds
    for seed in 0:4
        try
            givp(sphere, [(5.0, -5.0)]; config = GIVPConfig(), seed = seed)
            push!(
                failures,
                FuzzFailure(
                    -1,
                    "inverted_bounds_no_error",
                    "sphere",
                    1,
                    seed,
                    "inverted bounds should raise InvalidBoundsError",
                ),
            )
        catch e
            e isa InvalidBoundsError || push!(
                failures,
                FuzzFailure(
                    -1,
                    "wrong_exception_type",
                    "sphere",
                    1,
                    seed,
                    sprint(showerror, e),
                ),
            )
        end
    end

    # Zero-width bounds
    for seed in 0:4
        try
            givp(sphere, [(0.0, 0.0)]; config = GIVPConfig(), seed = seed)
            push!(
                failures,
                FuzzFailure(
                    -1,
                    "zero_bounds_no_error",
                    "sphere",
                    1,
                    seed,
                    "zero-width bounds should raise InvalidBoundsError",
                ),
            )
        catch e
            e isa InvalidBoundsError || push!(
                failures,
                FuzzFailure(
                    -1,
                    "wrong_exception_type",
                    "sphere",
                    1,
                    seed,
                    sprint(showerror, e),
                ),
            )
        end
    end

    # Invalid config (max_iterations = 0)
    for seed in 0:2
        try
            givp(
                sphere,
                [(-1.0, 1.0)];
                config = GIVPConfig(max_iterations = 0),
                seed = seed,
            )
            push!(
                failures,
                FuzzFailure(
                    -1,
                    "invalid_config_no_error",
                    "sphere",
                    1,
                    seed,
                    "max_iterations=0 should raise InvalidConfigError",
                ),
            )
        catch e
            e isa InvalidConfigError || push!(
                failures,
                FuzzFailure(
                    -1,
                    "wrong_exception_type",
                    "sphere",
                    1,
                    seed,
                    sprint(showerror, e),
                ),
            )
        end
    end

    verbose && @info "Invalid-input trials: $(length(failures)) failures"
    return failures
end

# ── Main ──────────────────────────────────────────────────────────────────────

function parse_args()::FuzzConfig
    args = ARGS
    n_trials = 500
    seed = 1337
    timeout_s = 120.0
    verbose = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--help", "-h")
            println("""
Usage: julia --project=julia julia/fuzz/fuzz_givp.jl [OPTIONS]

Options:
  --n-trials INT    Number of random fuzz trials (default: 500)
  --seed INT        Master RNG seed (default: 1337)
  --timeout FLOAT   Wall-clock seconds for the full session (default: 120.0)
  --verbose         Print each trial (default: false)
  --help            Show this message
""")
            exit(0)
        elseif a == "--n-trials" && i < length(args)
            n_trials = parse(Int, args[i + 1])
            i += 2
        elseif a == "--seed" && i < length(args)
            seed = parse(Int, args[i + 1])
            i += 2
        elseif a == "--timeout" && i < length(args)
            timeout_s = parse(Float64, args[i + 1])
            i += 2
        elseif a == "--verbose"
            verbose = true
            i += 1
        else
            @warn "Unknown argument: $a (ignored)"
            i += 1
        end
    end
    return FuzzConfig(n_trials, seed, timeout_s, verbose)
end

function main()
    cfg = parse_args()

    rng = MersenneTwister(cfg.seed)
    failures = FuzzFailure[]
    t_start = time()

    @info @sprintf(
        "GIVPOptimizer Fuzzer — %d trials | seed=%d | timeout=%.0fs",
        cfg.n_trials,
        cfg.seed,
        cfg.timeout_s
    )
    @info "  Package version: $(GIVPOptimizer.__version__)"
    @info "  Julia version  : $(VERSION)"

    # ── Phase 1: invalid-input taxonomy ──────────────────────────────────────
    @info "Phase 1: invalid-input trials..."
    inv_failures = run_invalid_input_trials(rng, cfg.verbose)
    append!(failures, inv_failures)
    isempty(inv_failures) ? @info("  ✓ All invalid-input checks passed") :
    @warn("  ✗ $(length(inv_failures)) invalid-input failures")

    # ── Phase 2: random valid trials ─────────────────────────────────────────
    @info "Phase 2: $(cfg.n_trials) random valid trials..."
    passed = 0
    skipped = 0  # due to timeout

    for trial in 1:(cfg.n_trials)
        if time() - t_start > cfg.timeout_s
            skipped = cfg.n_trials - trial + 1
            @warn "Timeout reached after $(trial-1) trials ($(skipped) skipped)"
            break
        end

        fail = run_trial(trial, rng, cfg.verbose)
        if fail !== nothing
            push!(failures, fail)
            @warn @sprintf(
                "  ✗ Trial %4d FAILED [%s]: %s",
                trial,
                fail.func_name,
                fail.error
            )
        else
            passed += 1
        end
    end

    total_run = cfg.n_trials - skipped
    elapsed = time() - t_start
    @info @sprintf("Phase 2 complete: %d/%d passed in %.1fs", passed, total_run, elapsed)

    # ── Summary ───────────────────────────────────────────────────────────────
    if isempty(failures)
        @info "✓ All fuzz trials passed!"
        exit(0)
    else
        println(stderr, "\n$(length(failures)) failure(s) found:\n")
        for (i, f) in enumerate(failures)
            println(
                stderr,
                "  [$i] trial=$(f.trial) category=$(f.category) " *
                "func=$(f.func_name) ndim=$(f.ndim) seed=$(f.seed)",
            )
            println(stderr, "       $(f.error)")
        end
        exit(1)
    end
end

main()
