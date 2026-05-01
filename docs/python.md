# Python

The reference implementation of GIVP lives in `python/src/givp/` and is published
to PyPI as the `givp` package.

## Installation

```bash
pip install givp
```

For process-based parallel evaluation of closures/lambdas (bypasses the GIL):

```bash
pip install "givp[parallel]"   # adds cloudpickle
```

Requires Python 3.10+.

## Quick start

```python
from givp import givp
import numpy as np

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

bounds = [(-5.12, 5.12)] * 10
result = givp(sphere, bounds)
print(result.x)     # best vector found
print(result.fun)   # best objective value
print(result.nfev)  # number of evaluations
```

## Maximize

```python
result = givp(my_score, bounds, direction="maximize")
```

## Configuration

All algorithm hyper-parameters are exposed via `GIVPConfig`:

```python
from givp import givp, GIVPConfig

cfg = GIVPConfig(
    max_iterations=50,
    vnd_iterations=100,
    ils_iterations=10,
    elite_size=7,
    adaptive_alpha=True,
    time_limit=30.0,
)
result = givp(sphere, bounds, config=cfg, seed=42, verbose=True)
```

## Configuration parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_iterations` | 100 | GRASP outer iterations |
| `alpha` | 0.12 | Initial RCL randomisation parameter (0 = greedy, 1 = random) |
| `vnd_iterations` | 200 | Maximum VND local-search iterations |
| `ils_iterations` | 10 | ILS perturbation attempts per GRASP iteration |
| `perturbation_strength` | 4 | Number of variables perturbed in ILS |
| `use_elite_pool` | `True` | Maintain elite pool for path relinking |
| `elite_size` | 7 | Maximum elite pool size |
| `path_relink_frequency` | 8 | GRASP iterations between path-relinking calls |
| `adaptive_alpha` | `True` | Vary alpha between `alpha_min` and `alpha_max` |
| `alpha_min` | 0.08 | Lower bound of adaptive alpha |
| `alpha_max` | 0.18 | Upper bound of adaptive alpha |
| `num_candidates_per_step` | 20 | Candidates generated per construction step |
| `use_cache` | `True` | Memoize evaluations with an LRU cache |
| `cache_size` | 10000 | LRU cache capacity |
| `early_stop_threshold` | 80 | Iterations without improvement before stopping |
| `use_convergence_monitor` | `True` | Enable diversification heuristics |
| `n_workers` | 1 | Parallel workers for candidate evaluation |
| `time_limit` | 0.0 | Wall-clock budget in seconds (0 = unlimited) |
| `direction` | `"minimize"` | `"minimize"` or `"maximize"` |
| `integer_split` | `None` | Index where integer variables begin (`None` = `n//2`) |
| `group_size` | `None` | Variable group size for group/block neighbourhoods |

## Warm start

```python
result = givp(
    rosenbrock,
    [(-2.0, 2.0)] * 5,
    initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
)
```

## Mixed continuous/integer problems

```python
n_cont, n_int = 12, 8
bounds = [(-5.0, 5.0)] * n_cont + [(0.0, 4.0)] * n_int

from givp import GIVPConfig
cfg = GIVPConfig(integer_split=n_cont)
result = givp(my_objective, bounds, config=cfg)
```

## Parallelism

Set `n_workers > 1` to evaluate GRASP candidates in parallel.  Three strategies
are tried in order:

1. **ProcessPoolExecutor** — used when the objective is picklable.
   Provides true multi-core speedup, bypasses the GIL.
2. **cloudpickle ProcessPoolExecutor** — used when the objective is a closure.
   Requires `pip install "givp[parallel]"`.
3. **ThreadPoolExecutor** — fallback when neither process strategy works, or when
   `use_cache=True` (cache cannot be shared across processes).

> **Warning**: `use_cache=True` with `n_workers > 1` always degrades to
> ThreadPoolExecutor and emits a `WARNING` log.
> For true multi-core speedup use `GIVPConfig(use_cache=False, n_workers=4)`.

## Result object

`givp` returns an `OptimizeResult` with:

| Field | Type | Meaning |
|---|---|---|
| `x` | `np.ndarray` | Best solution vector |
| `fun` | `float` | Objective value at `x` (user's original sign) |
| `nit` | `int` | GRASP outer iterations executed |
| `nfev` | `int` | Number of objective evaluations |
| `success` | `bool` | `True` when at least one feasible solution found |
| `message` | `str` | Human-readable termination reason |
| `direction` | `str` | `"minimize"` or `"maximize"` |
| `meta` | `dict` | Algorithm-specific extras (`termination_reason`, etc.) |

## Iteration callback

```python
def my_callback(iteration: int, cost: float, solution: np.ndarray) -> None:
    print(f"iter {iteration}: best={cost:.6f}")

result = givp(sphere, bounds, iteration_callback=my_callback)
```

## Reproducible multi-seed experiments

Use `seed_sweep` to run the optimizer over many independent seeds and collect
metrics suitable for academic reporting:

```python
from givp import seed_sweep, sweep_summary
from givp.benchmarks import sphere

bounds = [(-5.12, 5.12)] * 10
rows = seed_sweep(sphere, bounds, seeds=30)      # list[dict] or pd.DataFrame
stats = sweep_summary(rows)
# {'fun': {'mean': ..., 'std': ..., 'min': ..., 'max': ...}, 'nit': ..., ...}
print(f"fun: {stats['fun']['mean']:.4e} ± {stats['fun']['std']:.4e}")
```

Requires `pandas` for DataFrame output; works without it returning `list[dict]`.

## OO interface

```python
from givp import GIVPOptimizer

opt = GIVPOptimizer(sphere, [(-5.0, 5.0)] * 5)
r1 = opt.run(seed=1)
r2 = opt.run(seed=2)
print(opt.best_fun)    # global best across all runs
print(len(opt.history))  # 2
```

## Component-level API

Each algorithmic phase can be called independently:

```python
import numpy as np
from givp import GIVPConfig
from givp.core import (
    construct_grasp, local_search_vnd, ils_search,
    path_relinking, bidirectional_path_relinking,
)

lower = np.array([-5.12] * 5)
upper = np.array([5.12] * 5)
sphere_fn = lambda x: float(np.sum(x**2))

# 1. Construction
sol = construct_grasp(5, lower, upper, sphere_fn, None, alpha=0.1)

# 2. Local search (VND)
sol = local_search_vnd(sphere_fn, sol, 5, max_iter=50, lower_arr=lower, upper_arr=upper)

# 3. ILS
from givp.core.ils import ils_search
sol, cost = ils_search(
    sol, sphere_fn(sol), 5, sphere_fn, GIVPConfig(), lower_arr=lower, upper_arr=upper
    )

# 4. Path relinking between two solutions
source, target = sol.copy(), sol + np.random.uniform(-0.1, 0.1, 5)
best_pr, _ = bidirectional_path_relinking(sphere_fn, source, target)
```

## Running tests

```bash
cd python
pytest
```

With coverage (gate: 95 %):

```bash
pytest --cov=givp --cov-report=term-missing --cov-fail-under=95
```

## CLI

```bash
givp run \
    --func "lambda x: float(sum(xi**2 for xi in x))" \
    --bounds '[[-5.12,-5.12,-5.12],[5.12,5.12,5.12]]' \
    --seed 42 --verbose

givp version
```

## Fuzzing

```bash
python python/fuzz/fuzz_givp.py --n-trials 200 --verbose
```

## API parity with other ports

All parameters described above have identical names and semantics in the Julia,
Rust, and C++ ports.  See the respective documentation pages for port-specific
details.
