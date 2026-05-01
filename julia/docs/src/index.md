# GIVPOptimizer.jl

**GRASP-ILS-VND with Path Relinking** — Julia implementation.

A high-performance metaheuristic for continuous and mixed-integer black-box optimization.
Combines four complementary strategies:

| Component | Role                                                                         |
|-----------|------------------------------------------------------------------------------|
| **GRASP** | Greedy Randomized Adaptive Search (construction phase)                       |
| **ILS**   | Iterated Local Search (escape local optima via perturbation)                 |
| **VND**   | Variable Neighborhood Descent (structured local search with 3 neighborhoods) |
| **PR**    | Path Relinking (intensification via elite solution pool)                     |

## Quick links

- [Quick Start](@ref) — install and run in 5 minutes
- [Algorithm](@ref) — how GRASP-ILS-VND-PR works
- [API Reference](@ref) — full function/struct documentation
- [Julia Guide](@ref) — Julia-specific tips and comparisons

## Installation

```julia
# From Julia General Registry (≥ v1.0):
using Pkg; Pkg.add("GIVPOptimizer")

# Or the latest development version:
using Pkg; Pkg.develop(url="https://github.com/Arnime/grasp_ils_vnd_pr", subdir="julia")
```

## Minimal example

```julia
using GIVPOptimizer

sphere(x) = sum(x .^ 2)
bounds = [(-5.12, 5.12) for _ in 1:10]  # 10-D

result = givp(sphere, bounds; seed = 42)
println("Best value: ", result.fun)   # ≈ 0
println("Best point: ", result.x)
```

## Citing

If you use GIVPOptimizer in research, please cite:

```bibtex
@software{pires2026givp,
  author  = {Pires Junior, Arnaldo Mendes},
  title   = {{GIVPOptimizer.jl}: {GRASP-ILS-VND} with Path Relinking},
  year    = {2026},
  url     = {https://github.com/Arnime/grasp_ils_vnd_pr},
  version = {1.0.0}
}
```
