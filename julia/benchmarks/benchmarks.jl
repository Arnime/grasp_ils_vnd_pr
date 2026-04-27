# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""
Performance benchmarks for the GIVP optimizer.

Run with:

    julia --project=.. benchmarks.jl

These benchmarks use BenchmarkTools.jl to track regressions.
"""

using BenchmarkTools
using Printf

# Activate the parent project so GIVP is available
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GIVP

# ── Test functions ──────────────────────────────────────────────────────

"""Sphere: sum of squares, global minimum 0 at origin."""
function sphere(x::Vector{Float64})::Float64
    return sum(x .^ 2)
end

"""Rosenbrock banana function: global minimum 0 at (1,…,1)."""
function rosenbrock(x::Vector{Float64})::Float64
    return sum(100.0 .* (x[2:end] .- x[1:end-1] .^ 2) .^ 2 .+ (1.0 .- x[1:end-1]) .^ 2)
end

"""Rastrigin function: highly multimodal, global minimum 0 at origin."""
function rastrigin(x::Vector{Float64})::Float64
    n = length(x)
    return 10.0 * n + sum(x .^ 2 .- 10.0 .* cos.(2.0 * π .* x))
end

"""Ackley function: global minimum 0 at origin."""
function ackley(x::Vector{Float64})::Float64
    n = length(x)
    sum_sq = sum(x .^ 2)
    sum_cos = sum(cos.(2.0 * π .* x))
    return -20.0 * exp(-0.2 * sqrt(sum_sq / n)) - exp(sum_cos / n) + 20.0 + ℯ
end

# ── Benchmark configuration ────────────────────────────────────────────

const FUNCS = Dict(
    "sphere"     => (sphere,     -5.0,  5.0),
    "rosenbrock" => (rosenbrock, -2.0,  2.0),
    "rastrigin"  => (rastrigin,  -5.12, 5.12),
    "ackley"     => (ackley,     -5.0,  5.0),
)

const DIMS = [5, 10]
const CFG = GIVPConfig(; max_iterations=5, vnd_iterations=10)

# ── Run benchmarks ─────────────────────────────────────────────────────

const suite = BenchmarkGroup()

for (name, (func, lo, hi)) in FUNCS
    for dim in DIMS
        bounds = [(lo, hi) for _ in 1:dim]
        tag = "$(name)_dim$(dim)"
        suite[tag] = @benchmarkable begin
            givp($func, $bounds; config=$CFG)
        end samples=5 evals=1 seconds=120
    end
end

println("=" ^ 70)
println("GIVP Benchmarks — $(length(suite)) cases")
println("=" ^ 70)

# Warm-up: run each once to trigger compilation
for (name, (func, lo, hi)) in FUNCS
    bounds = [(lo, hi) for _ in 1:DIMS[1]]
    givp(func, bounds; config=CFG)
end
println("Warm-up complete.\n")

results = run(suite; verbose=true)

println("\n", "=" ^ 70)
println("Results")
println("=" ^ 70)

for tag in sort(collect(keys(results)))
    trial = results[tag]
    med = median(trial)
    @printf("%-25s  median=%10.2f ms  memory=%8.2f KiB  allocs=%d\n",
            tag, med.time / 1e6, med.memory / 1024, med.allocs)
end

println("\n", "=" ^ 70)
println("Summary (median times)")
println("=" ^ 70)
display(median(results))
println()

# ── Optionally save results for regression tracking ────────────────────

const RESULTS_FILE = joinpath(@__DIR__, "results.json")
if isfile(RESULTS_FILE)
    println("\nComparing with previous results from $(RESULTS_FILE):")
    previous = BenchmarkTools.load(RESULTS_FILE)[1]
    judgment = judge(median(results), median(previous); time_tolerance=0.10)
    display(judgment)
    println()
end

BenchmarkTools.save(RESULTS_FILE, results)
println("Results saved to $(RESULTS_FILE)")
