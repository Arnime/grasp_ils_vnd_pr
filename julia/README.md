# GIVPOptimizer.jl

[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![CI Julia](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Julia port of **GRASP-ILS-VND with Path Relinking** — a direction-agnostic
metaheuristic optimizer for **continuous, integer, and mixed** black-box problems.

The algorithm bundles:

- **GRASP** — Greedy Randomized Adaptive Search Procedure
- **ILS** — Iterated Local Search
- **VND** — Variable Neighborhood Descent (with an adaptive variant)
- **Path Relinking** between elite solutions
- LRU evaluation cache, convergence monitor, and a wall-clock time budget

The public API mirrors `scipy.optimize`: pass an objective callable, bounds,
and optional configuration, and get back an `OptimizeResult` with `x`, `fun`,
`nit`, `nfev`, `success`, `message`, `direction`, and `meta`.

---

## Installation

From the Julia General Registry (recommended):

```julia
using Pkg
Pkg.add("GIVPOptimizer")
```

From a local clone:

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr.git
cd grasp_ils_vnd_pr/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires **Julia 1.9+**.

---

## Quick start

```julia
using GIVPOptimizer

# Minimize the sphere function in 10 dimensions
function sphere(x::Vector{Float64})::Float64
    return sum(x .^ 2)
end

bounds = [(-5.0, 5.0) for _ in 1:10]
result = givp(sphere, bounds)

println(result.x)      # best vector found
println(result.fun)    # best objective value
println(result.nfev)   # number of function evaluations
println(result.success)
```

### Maximization

```julia
result = givp(my_score, bounds; direction=maximize)
```

### Custom configuration

```julia
cfg = GIVPConfig(
    max_iterations = 50,
    vnd_iterations = 100,
    time_limit     = 30.0,
    adaptive_alpha = true,
)
result = givp(sphere, bounds; config=cfg, seed=42, verbose=true)
```

### Mixed integer–continuous variables

`integer_split` sets the index where continuous variables end and integer ones
begin. Variables `1:integer_split` are treated as continuous; the rest are
rounded to integers.

```julia
# 3 continuous variables, 2 integer variables
cfg = GIVPConfig(integer_split=3)
result = givp(f, bounds; config=cfg)
```

### Iteration callback

```julia
function my_callback(iteration::Int, best_value::Float64)
    println("iter $iteration → $best_value")
end

result = givp(sphere, bounds; iteration_callback=my_callback)
```

---

## API reference

### `givp(func, bounds; kwargs...) -> OptimizeResult`

| Argument | Type | Default | Description |
|---|---|---|---|
| `func` | `Function` | required | Objective `Vector{Float64} → scalar` |
| `bounds` | `Vector{Tuple}` or `(lower, upper)` | required | Search bounds |
| `direction` | `Direction` | `minimize` | `minimize` or `maximize` |
| `config` | `GIVPConfig` | `nothing` | Algorithm hyper-parameters |
| `initial_guess` | `Vector{Float64}` | `nothing` | Warm-start vector |
| `iteration_callback` | `Function` | `nothing` | Called as `f(iter, best)` each iteration |
| `seed` | `Int` | `nothing` | RNG seed for reproducibility |
| `verbose` | `Bool` | `false` | Print progress to stdout |

### `GIVPConfig` fields

| Field | Default | Description |
|---|---|---|
| `max_iterations` | `100` | Maximum number of GRASP outer iterations |
| `alpha` | `0.12` | GRASP greediness parameter ∈ [0, 1] |
| `adaptive_alpha` | `true` | Adaptively vary `alpha` between `alpha_min` and `alpha_max` |
| `alpha_min` | `0.08` | Lower bound for adaptive alpha |
| `alpha_max` | `0.18` | Upper bound for adaptive alpha |
| `vnd_iterations` | `200` | VND inner iterations |
| `ils_iterations` | `10` | ILS perturbation iterations |
| `perturbation_strength` | `4` | Number of dimensions perturbed by ILS |
| `use_elite_pool` | `true` | Maintain an elite solution pool |
| `elite_size` | `7` | Maximum number of elite solutions |
| `path_relink_frequency` | `8` | Run path relinking every N iterations |
| `num_candidates_per_step` | `20` | Candidate list size per construction step |
| `use_cache` | `true` | Enable LRU evaluation cache |
| `cache_size` | `10000` | Maximum cached evaluations |
| `early_stop_threshold` | `80` | Stop after N iterations without improvement |
| `use_convergence_monitor` | `true` | Enable convergence monitoring |
| `n_workers` | `1` | Thread-parallel candidate evaluation |
| `time_limit` | `0.0` | Wall-clock budget in seconds (0 = unlimited) |
| `integer_split` | `nothing` | Index split for mixed integer–continuous problems |

### `OptimizeResult` fields

| Field | Type | Description |
|---|---|---|
| `x` | `Vector{Float64}` | Best solution found |
| `fun` | `Float64` | Objective value at `x` |
| `nit` | `Int` | Number of outer iterations executed |
| `nfev` | `Int` | Total objective evaluations |
| `success` | `Bool` | Whether a finite solution was found |
| `message` | `String` | Human-readable termination reason |
| `direction` | `Direction` | `minimize` or `maximize` |
| `meta` | `Dict{String,Any}` | Extra diagnostic information |

---

## Running tests

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI enforces **≥ 95 %** line coverage on `julia/src/`.

## Running benchmarks

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

---

## License

MIT — see [LICENSE](LICENSE).
