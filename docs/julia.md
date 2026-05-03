# Julia

`givp` is also available as a native Julia package (`GIVPOptimizer.jl`),
providing the same GRASP-ILS-VND with Path Relinking algorithm with an idiomatic
Julia API.

## Installation

From a local clone of the repository:

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia 1.9+.

## Quick start

```julia
using GIVPOptimizer

sphere(x) = sum(x .^ 2)

result = givp(sphere, [(-5.0, 5.0) for _ in 1:10])
println(result.x)       # best vector found
println(result.fun)     # best objective value
println(result.nfev)    # number of evaluations
```

## Maximize

```julia
result = givp(my_score, bounds; direction=maximize)
```

## Configuration

All the same hyper-parameters available in Python are exposed in Julia via
`GIVPConfig`:

```julia
cfg = GIVPConfig(;
    max_iterations=50,
    vnd_iterations=100,
    ils_iterations=10,
    elite_size=7,
    adaptive_alpha=true,
    time_limit=30.0,
)
result = givp(sphere, bounds; config=cfg, seed=42, verbose=true)
```

## Warm start

```julia
result = givp(
    rosenbrock,
    [(-2.0, 2.0) for _ in 1:5];
    initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
)
```

## Mixed continuous/integer problems

```julia
n_cont, n_int = 12, 8
bounds = vcat(
    [(-5.0, 5.0) for _ in 1:n_cont],
    [(0.0, 4.0) for _ in 1:n_int],
)
cfg = GIVPConfig(; integer_split=n_cont)
result = givp(my_objective, bounds; config=cfg)
```

## Result struct

`givp` returns an `OptimizeResult` with the following fields:

| Field       | Type              | Meaning                                          |
|-------------|-------------------|--------------------------------------------------|
| `x`         | `Vector{Float64}` | Best solution vector.                            |
| `fun`       | `Float64`         | Objective value at `x` (user's original sign).   |
| `nit`       | `Int`             | GRASP outer iterations executed.                 |
| `nfev`      | `Int`             | Number of objective evaluations.                 |
| `success`   | `Bool`            | `true` when at least one feasible solution found.|
| `message`   | `String`          | Human-readable termination reason.               |
| `direction` | `Direction`       | `minimize` or `maximize`.                        |
| `meta`      | `Dict`            | Algorithm-specific extras.                       |

The result also supports `to_dict()` for JSON serialization.

## Running tests

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

With coverage:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; coverage=true)'
```

## Experimental seed sweep API

For reproducible multi-seed studies, Julia also provides the same
`seed_sweep`/`sweep_summary` workflow as the Python port:

```julia
using GIVPOptimizer

sphere(x) = sum(x .^ 2)
bounds = [(-5.12, 5.12) for _ in 1:10]

rows = seed_sweep(sphere, bounds; seeds=0:29)
summary = sweep_summary(rows)

println(summary["fun"]["mean"])
println(summary["fun"]["std"])
```

Each row contains `seed`, `fun`, `nit`, `nfev`, `time_s`, `success`, and
`message`.

## CLI

A command-line interface equivalent to `givp run` is available at
`julia/cli.jl`:

```bash
# Inline lambda
julia julia/cli.jl run \
    --func '(x) -> sum(x .^ 2)' \
    --bounds '[[-5.0,-5.0,-5.0],[5.0,5.0,5.0]]' \
    --seed 42

# Load function from a source file
julia julia/cli.jl run \
    --func-file examples/sphere.jl --func-name sphere \
    --bounds '[[-5.12,-5.12],[5.12,5.12]]' --verbose

# JSON mode (mirrors Python CLI)
julia julia/cli.jl run --json '{"func_file":"sphere.jl","func_name":"sphere","bounds":[[-5],[5]]}'

julia julia/cli.jl version
```

Output is JSON to stdout — compatible with the Python `givp run` format.

## Literature comparison experiment

A reproducible multi-run experiment comparing GIVP against baselines is
provided in `julia/benchmarks/`:

```bash
# Run experiment: 30 seeds × 6 functions × GIVP-full + GRASP-only
julia --project=julia julia/benchmarks/run_literature_comparison.jl \
    --n-runs 30 --dims 10 --output results.json --verbose

# Include BlackBoxOptim.jl baselines (DE and XNES)
julia --project=julia julia/benchmarks/run_literature_comparison.jl \
    --algorithms GIVP-full GRASP-only BBO-DE BBO-XNES

# Generate Markdown + LaTeX tables with Wilcoxon tests
julia --project=julia julia/benchmarks/generate_report.jl \
    --input results.json --format both

# Include per-iteration convergence curves (requires --traces in run step)
julia --project=julia julia/benchmarks/run_literature_comparison.jl --traces
julia --project=julia julia/benchmarks/generate_report.jl --input results.json --convergence
```

An interactive version is available as a Jupyter notebook at
`Notebooks/Julia/benchmark_literature_comparison_julia.ipynb`.

## Running benchmarks

The benchmarks use `BenchmarkTools.jl` and cover four classic test functions
(sphere, rosenbrock, rastrigin, ackley) at dimensions 5 and 10:

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

Results are saved to `benchmarks/results.json` for regression tracking.
Subsequent runs automatically compare against the previous results.

## Fuzzing

A crash-finder fuzzer exercises the API with random and adversarial inputs:

```bash
julia --project=julia julia/fuzz/fuzz_givp.jl --n-trials 500 --verbose
```

Phase 1 checks that invalid inputs always raise the correct `GivpError` subtype.
Phase 2 runs random valid trials verifying six invariants per result
(bounds containment, `nfev > 0`, `success ↔ isfinite(fun)`, etc.).
Exit code 0 = all passed, 1 = failures found.

## Coverage

The CI enforces a minimum of **95 %** line coverage on `julia/src/`.
To check locally:

```bash
julia --project=julia -e '
  using Pkg; Pkg.add("CoverageTools")
  using CoverageTools
  cov = process_folder("julia/src")
  hit   = count(c -> c !== nothing && c > 0, [c for s in cov for c in s.coverage])
  total = count(c -> c !== nothing,          [c for s in cov for c in s.coverage])
  println("Coverage: $(round(hit/total*100; digits=1))%")
'
```

## API parity with Python

The Julia port aims for full feature parity with the Python implementation:

| Feature                    | Python | Julia |
|----------------------------|--------|-------|
| GRASP construction         | ✓      | ✓     |
| VND local search           | ✓      | ✓     |
| ILS perturbation           | ✓      | ✓     |
| Path Relinking             | ✓      | ✓     |
| Elite pool                 | ✓      | ✓     |
| Convergence monitor        | ✓      | ✓     |
| LRU evaluation cache       | ✓      | ✓     |
| Adaptive α                 | ✓      | ✓     |
| Time budget                | ✓      | ✓     |
| Mixed integer/continuous   | ✓      | ✓     |
| Warm start                 | ✓      | ✓     |
| Reproducible (`seed=`)     | ✓      | ✓     |
| CLI entry point            | ✓      | ✓     |
| `GIVPOptimizer` class      | ✓      | ✓     |
| Literature comparison      | ✓      | ✓     |
| Wilcoxon + LaTeX reports   | ✓      | ✓     |
| Fuzzing driver             | ✓      | ✓     |
