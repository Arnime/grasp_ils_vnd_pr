# Julia Guide

## Why a Julia port?

Julia's just-in-time compilation and native array operations make it ideal for
computationally intensive metaheuristics. The Julia port of GIVPOptimizer:

- Achieves **1.5–3× speedup** over the Python port on CPU-bound benchmarks
- Supports **Julia 1.9 – 1.12** (tested in CI across all versions)
- Has an identical API to the Python version (same parameter names and semantics)
- Is available via the Julia General Registry: `Pkg.add("GIVPOptimizer")`

## Bounds formats

Three formats are accepted, matching the Python API:

```julia
# 1. Vector of (low, high) tuples (recommended)
bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

# 2. Tuple of (lower_vector, upper_vector)
bounds = ([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])

# 3. Vector of [lower_vector, upper_vector]
bounds = [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]
```

## Performance tips

### Avoid closures over large objects

The objective function is called millions of times. Keep it allocation-free:

```julia
# Good — no allocations inside objective
function my_func(x::Vector{Float64})::Float64
    s = 0.0
    for xi in x
        s += xi^2
    end
    return s
end

# Bad — allocates a new array on each call
my_func_slow(x) = sum(x .^ 2)   # dot-broadcast allocates
```

### Use `time_limit` for long experiments

```julia
cfg = GIVPConfig(; max_iterations = 100_000, time_limit = 30.0)
# Stops after 30s regardless of iterations remaining
```

### Parallel multi-start (external)

Julia's built-in `Threads.@threads` can run multiple independent seeds in parallel:

```julia
using Base.Threads

results = Vector{Any}(undef, 30)
@threads for seed in 0:29
    results[seed + 1] = givp(my_func, bounds; config = cfg, seed = seed)
end

best = argmin(r -> r.fun, results)
println("Best across 30 runs: ", best.fun)
```

## Comparison with Python port

| Feature | Julia | Python |
|---|---|---|
| Package | `GIVPOptimizer.jl` | `givp` |
| Entry point | `givp(func, bounds; ...)` | `givp(func, bounds, ...)` |
| Config struct | `GIVPConfig(; kwargs...)` | `GIVPConfig(**kwargs)` |
| Result | `OptimizeResult` | `OptimizeResult` |
| Direction | `minimize` / `maximize` (enum) | `"minimize"` / `"maximize"` (string) |
| Bounds | tuples / vectors | list of tuples |
| Type checking | static (JET.jl) | mypy strict |
| CI coverage | ≥ 90% | ≥ 95% |

## Running benchmarks

```bash
# From the repo root
julia --project=julia julia/benchmarks/benchmarks.jl

# Full literature comparison (30 runs × 6 functions):
julia --project=julia julia/benchmarks/run_literature_comparison.jl \
    --n-runs 30 --dims 10 --output results.json --verbose

# Generate Markdown/LaTeX report from results:
julia --project=julia julia/benchmarks/generate_report.jl \
    --input results.json --format both
```

## Running tests

```bash
# All tests (from repo root):
julia --project=julia -e 'using Pkg; Pkg.test()'

# Static analysis only (JET + Aqua):
julia --project=julia julia/test/test_static_analysis.jl

# Single test file:
julia --project=julia -e '
  using Test, GIVPOptimizer, Random, LinearAlgebra
  include("julia/test/test_api.jl")
'
```
