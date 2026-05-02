# Quick Start

## Installation

```julia
using Pkg
Pkg.add("GIVPOptimizer")   # Julia ≥ 1.9
```

## Minimizing a function

```julia
using GIVPOptimizer

# Define an objective (Vector{Float64} → scalar)
rastrigin(x) = 10.0 * length(x) + sum(xi^2 - 10.0 * cos(2π * xi) for xi in x)

# Bounds: list of (lower, upper) tuples, one per variable
bounds = [(-5.12, 5.12) for _ in 1:10]

result = givp(rastrigin, bounds; seed = 42)

println(result.fun)     # best objective value found
println(result.x)       # best solution vector
println(result.nit)     # number of GRASP iterations
println(result.nfev)    # total function evaluations
println(result.message) # termination reason
```

## Maximizing a function

Pass `direction = maximize` (imported from `GIVPOptimizer`):

```julia
using GIVPOptimizer

neg_sphere(x) = sum(x .^ 2)  # maximize = find largest sphere value
bounds = [(-5.0, 5.0) for _ in 1:5]

result = givp(neg_sphere, bounds; direction = maximize, seed = 0)
println(result.fun)   # positive value (maximum)
```

## Controlling the algorithm

Use `GIVPConfig` to tune every hyperparameter:

```julia
cfg = GIVPConfig(;
    max_iterations      = 200,    # total GRASP iterations
    alpha               = 0.12,   # RCL greediness  (0=greedy, 1=random)
    adaptive_alpha      = true,   # linearly vary alpha over iterations
    alpha_min           = 0.08,
    alpha_max           = 0.20,
    vnd_iterations      = 300,    # VND inner iterations per GRASP call
    ils_iterations      = 10,     # ILS restarts per GRASP iteration
    perturbation_strength = 4,    # number of dimensions perturbed in ILS
    use_elite_pool      = true,
    elite_size          = 7,
    path_relink_frequency = 8,    # run PR every N iterations
    use_cache           = true,
    cache_size          = 10_000,
    early_stop_threshold = 80,    # stop if no improvement for N iterations
    use_convergence_monitor = true,
    time_limit          = 60.0,   # hard wall-clock cap in seconds
)

result = givp(rastrigin, bounds; config = cfg, seed = 42)
```

## Iteration callback

Monitor progress without verbose output:

```julia
history = Float64[]

result = givp(
    rastrigin, bounds;
    seed = 42,
    iteration_callback = (iter, cost, sol) -> push!(history, cost),
)

# history[i] = best cost at iteration i
```

## Warm start

Provide a known feasible solution to seed the search:

```julia
x0 = zeros(10)   # start at origin

result = givp(rastrigin, bounds; initial_guess = x0, seed = 42)
```

## Mixed-integer problems

Set `integer_split` in `GIVPConfig` to treat variables `integer_split+1 : n` as
integers:

```julia
cfg = GIVPConfig(; integer_split = 3, max_iterations = 50)
# Variables 1-3 are continuous, 4-n are integer

bounds_mixed = vcat(
    [(-5.0, 5.0) for _ in 1:3],   # continuous
    [(0.0, 10.0) for _ in 1:5],   # integer
)

result = givp(my_function, bounds_mixed; config = cfg, seed = 1)
```

## Checking the result

`OptimizeResult` supports tuple unpacking:

```julia
x, fun = result          # unpack to (solution_vector, best_value)

result.success           # Bool — true if a finite solution was found
result.nit               # Int  — iterations completed
result.nfev              # Int  — function evaluations
result.message           # String — "max iterations reached" |
                         # "time limit reached" | ...
result.meta              # Dict  — extra metadata
to_dict(result)          # Dict — JSON-serialisable representation
```
