# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Convergence quality gate — run as a standalone script in CI.
# Verifies that GIVPOptimizer actually converges on standard benchmark
# functions, not merely that it executes without errors.
#
# Exit codes:
#   0  all assertions passed
#   1  one or more assertions failed (algorithmic regression detected)
#
# Run locally:
#   julia --project=julia julia/test/test_quality_gate.jl

using GIVPOptimizer
using Statistics

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

sphere(x) = sum(x .^ 2)

function rosenbrock(x::Vector{Float64})::Float64
    return sum(100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2 for i in 1:(length(x) - 1))
end

# ---------------------------------------------------------------------------
# Shared config — intentionally lightweight for CI speed
# ---------------------------------------------------------------------------

function make_config()
    return GIVPConfig(;
        max_iterations = 50,
        vnd_iterations = 80,
        ils_iterations = 5,
        early_stop_threshold = 30,
    )
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

failures = String[]

function assert_below(label::String, value::Float64, threshold::Float64)
    if value < threshold
        println("  PASS  $label: $value < $threshold")
    else
        msg = "  FAIL  $label: $value >= $threshold  (threshold: $threshold)"
        println(msg)
        push!(failures, msg)
    end
end

# ---------------------------------------------------------------------------
# Gate 1 — Sphere 5D: median over 5 seeds must be < 1.0
# ---------------------------------------------------------------------------

println("\n=== Gate 1: Sphere 5D — median over 5 seeds ===")
let
    bounds = [(-5.12, 5.12) for _ in 1:5]
    cfg = make_config()
    fvals = Float64[]
    for s in 0:4
        r = givp(sphere, bounds; direction = minimize, config = cfg, seed = s)
        push!(fvals, r.fun)
        println(
            "  seed=$s  fun=$(round(r.fun; digits=6))  nit=$(r.nit)  success=$(r.success)",
        )
    end
    med = median(fvals)
    println("  median fun = $(round(med; digits=6))")
    assert_below("Sphere-5D median", med, 1.0)
end

# ---------------------------------------------------------------------------
# Gate 2 — Rosenbrock 4D: median over 5 seeds must be < 100.0
# ---------------------------------------------------------------------------

println("\n=== Gate 2: Rosenbrock 4D — median over 5 seeds ===")
let
    bounds = [(-2.048, 2.048) for _ in 1:4]
    cfg = make_config()
    fvals = Float64[]
    for s in 0:4
        r = givp(rosenbrock, bounds; direction = minimize, config = cfg, seed = s)
        push!(fvals, r.fun)
        println(
            "  seed=$s  fun=$(round(r.fun; digits=4))  nit=$(r.nit)  success=$(r.success)",
        )
    end
    med = median(fvals)
    println("  median fun = $(round(med; digits=4))")
    assert_below("Rosenbrock-4D median", med, 100.0)
end

# ---------------------------------------------------------------------------
# Gate 3 — Strict canary: seed=0, Sphere 5D must achieve fun < 0.5
# ---------------------------------------------------------------------------

println("\n=== Gate 3: Sphere 5D strict canary (seed=0) ===")
let
    bounds = [(-5.12, 5.12) for _ in 1:5]
    cfg = make_config()
    r = givp(sphere, bounds; direction = minimize, config = cfg, seed = 0)
    println("  seed=0  fun=$(round(r.fun; digits=6))  nit=$(r.nit)  success=$(r.success)")
    assert_below("Sphere-5D canary (seed=0)", r.fun, 0.5)
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

println("\n=== Quality Gate Summary ===")
if isempty(failures)
    println("All convergence assertions passed ✓")
    exit(0)
else
    println("$(length(failures)) assertion(s) FAILED — algorithmic regression detected:")
    for f in failures
        println(f)
    end
    exit(1)
end
