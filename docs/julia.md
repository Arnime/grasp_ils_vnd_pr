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

## Running benchmarks

The benchmarks use `BenchmarkTools.jl` and cover four classic test functions
(sphere, rosenbrock, rastrigin, ackley) at dimensions 5 and 10:

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

Results are saved to `benchmarks/results.json` for regression tracking.
Subsequent runs automatically compare against the previous results.

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
| CLI entry point            | ✓      | —     |
| `GIVPOptimizer` class      | ✓      | —     |
