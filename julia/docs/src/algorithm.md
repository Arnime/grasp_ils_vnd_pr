# Algorithm

## Overview

GIVPOptimizer combines four complementary metaheuristic strategies into one pipeline:

```text
GRASP construction → VND local search → ILS perturbation → Path Relinking
       ↑___________________feedback (elite pool)__________________↑
```

Each GRASP **iteration** performs:

1. **Construction** — build a candidate solution via greedy randomized adaptive
   search
2. **VND** — improve it with variable neighborhood descent (3 nested neighborhoods)
3. **ILS** — perturb and re-search multiple times to escape local optima
4. **Path Relinking** — every `path_relink_frequency` iterations, interpolate
   between the current solution and an elite solution to find improving paths

## GRASP Construction

The RCL (Restricted Candidate List) controls exploration vs. exploitation:

- `alpha = 0` → purely greedy (deterministic)
- `alpha = 1` → uniformly random
- `0 < alpha < 1` → balanced (recommended: `0.08`–`0.20`)

**Adaptive alpha** linearly interpolates `alpha_min` → `alpha_max` across iterations,
starting focused (greedy) and becoming more exploratory over time.

**Candidate generation**: `num_candidates_per_step` random points are sampled per
variable step; only those within `alpha × (f_best − f_worst)` of `f_best` enter
the RCL.

## VND Local Search

Three neighborhoods are tried in sequence (VND order):

| # | Neighborhood | Description |
|---|---|---|
| 1 | **Flip** | Perturb one variable at a time (first-improvement) |
| 2 | **Swap** | Exchange values of two variables |
| 3 | **Multiflip** | Perturb multiple variables simultaneously |

On improvement, the search restarts from neighborhood 1 (standard VND).
Integer and continuous variables are treated differently:

- **Continuous**: Gaussian perturbation within bounds
- **Integer**: ±1 step within bounds

`vnd_iterations` controls the maximum total moves per VND call.

## ILS Perturbation

After VND converges, `ils_iterations` perturbation-restarting cycles are performed:

1. Perturb `perturbation_strength` dimensions of the current best
2. Apply VND to the perturbed solution
3. Accept if improvement found (greedy acceptance criterion)

## Elite Pool & Path Relinking

An elite pool of up to `elite_size` diverse solutions is maintained.
Diversity is enforced by `min_distance` (normalised Euclidean distance).

Every `path_relink_frequency` iterations:

- A random elite solution is chosen as **target**
- A **relinking path** from current → target is explored
- The best solution along the path replaces the current if it improves

Three strategies are available: `:forward`, `:best`, and `:bidirectional`.

## Convergence Monitor & Early Stop

The `ConvergenceMonitor` tracks iterations without improvement.

- When `no_improve_count ≥ early_stop_threshold`, the search terminates early
- When `no_improve_count ≥ threshold ÷ 2` **and** pool diversity is low,
  `should_intensify=true` is signaled → path relinking is triggered immediately

## Complexity

| Component | Time per iteration |
|---|---|
| GRASP construction | O(n × num_candidates_per_step) |
| VND | O(vnd_iterations × n) |
| ILS | O(ils_iterations × VND cost) |
| Path Relinking | O(n²) per call |
| Cache lookup | O(1) amortised (LRU) |

## References

1. Feo, T.A. & Resende, M.G.C. (1995). Greedy randomized adaptive search procedures.
   *Journal of Global Optimization*, 6(2), 109–133.
2. Lourenço, H.R., Martin, O.C. & Stützle, T. (2003). Iterated local search.
   In *Handbook of Metaheuristics*, Kluwer.
3. Hansen, P. & Mladenović, N. (2001). Variable neighborhood search.
   *European Journal of Operational Research*, 130(3), 449–467.
4. Glover, F. (1996). Tabu search and adaptive memory programming.
   In *Advances in Metaheuristic Optimization*, Kluwer.
