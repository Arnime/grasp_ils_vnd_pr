# givp — GRASP-ILS-VND with Path Relinking

[![PyPI version](https://img.shields.io/pypi/v/givp.svg)](https://pypi.org/project/givp/)
[![Python versions](https://img.shields.io/pypi/pyversions/givp.svg)](https://pypi.org/project/givp/)
[![CI](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/branch/main/graph/badge.svg)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/Arnime/grasp_ils_vnd_pr/badge)](https://securityscorecards.dev/viewer/?uri=github.com/Arnime/grasp_ils_vnd_pr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A direction-agnostic, NumPy-native metaheuristic optimizer for **continuous,
integer or mixed** black-box problems. The library bundles:

- **GRASP** — Greedy Randomized Adaptive Search Procedure
- **ILS** — Iterated Local Search
- **VND** — Variable Neighborhood Descent (with an adaptive variant)
- **Path Relinking** between elite solutions
- LRU evaluation cache, convergence monitor, optional thread-parallel candidate
  evaluation, and a wall-clock time budget

The public API mirrors `scipy.optimize`: pass an objective callable, bounds and
optional configuration, get back an `OptimizeResult` with `x`, `fun`, `nit`,
`nfev`, `success`, `message`, `direction`, `meta`.

---

## Table of contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Choosing the optimization sense](#choosing-the-optimization-sense)
4. [Bounds, integer variables and mixed problems](#bounds-integer-variables-and-mixed-problems)
5. [Object-oriented API and multi-start](#object-oriented-api-and-multi-start)
6. [Configuration cookbook](#configuration-cookbook)
7. [Inspecting progress (callback and verbose)](#inspecting-progress-callback-and-verbose)
8. [Public API reference](#public-api-reference)
9. [Glossary of hyper-parameters](#glossary-of-hyper-parameters)
10. [Adapting to a domain-specific model](#adapting-to-a-domain-specific-model)
11. [Comparison with other optimizers](#comparison-with-other-optimizers)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Install

From PyPI (once published):

```bash
pip install givp
```

From source (editable):

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr.git
cd grasp_ils_vnd_pr
pip install -e .[dev]
```

Requires Python 3.10+ and NumPy.

---

## Quick start

```python
import numpy as np
from givp import grasp_ils_vnd_pr

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

result = grasp_ils_vnd_pr(sphere, bounds=[(-5.0, 5.0)] * 10)
print(result.x)        # best vector found
print(result.fun)      # best objective value
print(result.nfev)     # number of evaluations performed
```

Default behavior:

- **Minimization** (`minimize=True` / `direction="minimize"`).
- All variables treated as continuous.
- Default hyper-parameters (`GraspIlsVndConfig()`).

---

## Choosing the optimization sense

The library is **agnostic** to whether you want the lowest or the highest
value of `func`. Two equivalent ways to declare it:

### Boolean flag (recommended)

```python
from givp import grasp_ils_vnd_pr

def gain(x):
    return float((x ** 2).sum())  # higher is better

result = grasp_ils_vnd_pr(gain, [(-5, 5)] * 10, minimize=False)
assert result.direction == "maximize"
```

### String flag (SciPy/Optuna compatible)

```python
result = grasp_ils_vnd_pr(gain, [(-5, 5)] * 10, direction="maximize")
```

Both flags are accepted on `grasp_ils_vnd_pr`, on `GraspOptimizer` and on
`GraspIlsVndConfig`. Setting **both** simultaneously is allowed only when they
agree; conflicting values raise `ValueError`.

> **Internal note.** The core algorithm always minimizes. When you ask for
> maximization the public API wraps your objective with a sign flip and
> restores the sign on `result.fun`. This means `result.fun` is always
> reported in your original sign — no need to negate it back yourself.

---

## Bounds, integer variables and mixed problems

`bounds` is accepted in two equivalent forms:

```python
# SciPy style: list of (low, high) per variable
bounds = [(-5.0, 5.0), (0.0, 10.0), (-1.0, 1.0)]

# (lower, upper) tuple of two equally-sized sequences
bounds = ([-5.0, 0.0, -1.0], [5.0, 10.0, 1.0])
```

By default every variable is continuous. To declare a **mixed** problem (some
continuous variables followed by some integer variables in the decision
vector), use `integer_split` on the configuration:

```python
from givp import GraspIlsVndConfig, grasp_ils_vnd_pr

n_cont, n_int = 12, 8
bounds = [(-5.0, 5.0)] * n_cont + [(0.0, 4.0)] * n_int

cfg = GraspIlsVndConfig(integer_split=n_cont)  # indices >= n_cont are integer

result = grasp_ils_vnd_pr(my_objective, bounds, config=cfg)
```

Special cases:

| `integer_split` | Meaning                                |
|-----------------|----------------------------------------|
| `None` (public API default: `num_vars`) | All-continuous problem.    |
| `0`             | All-integer problem.                   |
| `n_vars`        | All-continuous problem (explicit).     |
| `k` (0 < k < n) | First `k` continuous, rest integer.    |

---

## Object-oriented API and multi-start

When you want to keep configuration around, run the optimizer multiple times
and track the best result automatically, use `GraspOptimizer`:

```python
from givp import GraspIlsVndConfig, GraspOptimizer

opt = GraspOptimizer(
    func=sphere,
    bounds=[(-5.0, 5.0)] * 10,
    minimize=True,
    config=GraspIlsVndConfig(max_iterations=50, time_limit=30.0),
    verbose=True,
)
for _ in range(5):
    opt.run()
print("best across 5 restarts:", opt.best_fun)
print("history length:", len(opt.history))
```

`opt.best_x` and `opt.best_fun` always reflect the best result observed across
all `run()` calls, in the **user's original sign**.

---

## Configuration cookbook

```python
from givp import GraspIlsVndConfig

# 1) Fast triage (small budget, no warm-up)
cfg_fast = GraspIlsVndConfig(
    max_iterations=20,
    vnd_iterations=50,
    ils_iterations=5,
    use_elite_pool=False,
    use_convergence_monitor=False,
    use_cache=True,
)

# 2) Production-quality run with wall-clock budget
cfg_quality = GraspIlsVndConfig(
    max_iterations=200,
    vnd_iterations=300,
    ils_iterations=15,
    elite_size=10,
    path_relink_frequency=5,
    adaptive_alpha=True,
    alpha_min=0.05,
    alpha_max=0.20,
    time_limit=600.0,         # stop after 10 minutes
    n_workers=4,              # parallelize candidate evaluation
)

# 3) Expensive objective: maximize cache reuse, keep evaluations few
cfg_expensive = GraspIlsVndConfig(
    num_candidates_per_step=8,
    cache_size=50_000,
    use_cache=True,
    early_stop_threshold=40,  # stop earlier on stagnation
)

# 4) Maximization with hourly-shaped layout (3 plants × 24 hours)
cfg_hydro = GraspIlsVndConfig(
    minimize=False,
    integer_split=72,         # first 72 vars continuous, rest integer
    max_iterations=120,
    time_limit=300.0,
)
```

---

## Inspecting progress (callback and verbose)

Both `grasp_ils_vnd_pr` and `GraspOptimizer` accept:

- `verbose=True` — prints per-iteration cost and cache statistics.
- `iteration_callback=fn` — calls `fn(iteration_index, best_cost, best_solution)`
  once per outer GRASP iteration. The callback receives the cost in the
  **internal minimization sign** (i.e., already sign-flipped if you asked for
  maximization). Useful to plot convergence or persist intermediate results.

```python
costs = []

def log_iter(i, cost, sol):
    costs.append(cost)

result = grasp_ils_vnd_pr(
    sphere,
    [(-5, 5)] * 10,
    iteration_callback=log_iter,
    verbose=True,
)
```

---

## Public API reference

### `grasp_ils_vnd_pr(...) -> OptimizeResult`

```python
grasp_ils_vnd_pr(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]] | tuple[Sequence[float], Sequence[float]],
    *,
    num_vars: int | None = None,
    minimize: bool | None = None,
    direction: str | None = None,         # 'minimize' or 'maximize'
    config: GraspIlsVndConfig | None = None,
    initial_guess: Sequence[float] | None = None,
    iteration_callback: Callable[[int, float, np.ndarray], None] | None = None,
    verbose: bool = False,
) -> OptimizeResult
```

### `class GraspOptimizer`

Same constructor signature, exposes `.run() -> OptimizeResult` and tracks
`.best_x`, `.best_fun`, `.history`.

### `class GraspIlsVndConfig` (dataclass)

All hyper-parameters listed in the [glossary](#glossary-of-hyper-parameters).

### `class OptimizeResult`

| Field       | Type        | Meaning                                                    |
|-------------|-------------|------------------------------------------------------------|
| `x`         | `np.ndarray`| Best solution vector.                                      |
| `fun`       | `float`     | Objective value at `x`, in the **user's original sign**.   |
| `nit`       | `int`       | GRASP outer iterations executed.                           |
| `nfev`      | `int`       | Number of objective evaluations.                           |
| `success`   | `bool`      | True when at least one feasible solution was produced.     |
| `message`   | `str`       | Human-readable termination reason.                         |
| `direction` | `str`       | `'minimize'` or `'maximize'`.                              |
| `meta`      | `dict`      | Algorithm-specific extras (cache stats, etc.).             |

For backward compatibility the result is iterable: `x, fun = result` works.

---

## Glossary of hyper-parameters

| Field                       | Default | Meaning                                                            |
|-----------------------------|---------|--------------------------------------------------------------------|
| `max_iterations`            | 100     | GRASP outer iterations.                                            |
| `alpha`                     | 0.12    | Initial RCL randomization (0 = greedy, 1 = uniform).               |
| `vnd_iterations`            | 200     | Maximum VND inner iterations.                                      |
| `ils_iterations`            | 10      | Iterated Local Search loops per outer iteration.                   |
| `perturbation_strength`     | 4       | Magnitude of ILS perturbation (number of variables jolted).        |
| `use_elite_pool`            | True    | Maintain a diverse pool of elite solutions for path relinking.     |
| `elite_size`                | 7       | Maximum number of elite solutions kept.                            |
| `path_relink_frequency`     | 8       | Every N GRASP iterations, run path relinking on elite pairs.       |
| `adaptive_alpha`            | True    | If True, alpha varies in `[alpha_min, alpha_max]` over iterations. |
| `alpha_min` / `alpha_max`   | 0.08 / 0.18 | Bounds for adaptive alpha.                                     |
| `num_candidates_per_step`   | 20      | Candidates evaluated per construction step.                        |
| `use_cache`                 | True    | Memoize evaluations via LRU cache.                                 |
| `cache_size`                | 10000   | LRU cache capacity.                                                |
| `early_stop_threshold`      | 80      | Iterations without improvement before terminating.                 |
| `use_convergence_monitor`   | True    | Enable diversification/restart heuristics.                         |
| `n_workers`                 | 1       | Threads used to evaluate candidates concurrently.                  |
| `time_limit`                | 0.0     | Wall-clock budget in seconds (`0` = unlimited).                    |
| `minimize`                  | `None`  | Boolean direction flag. `True` = minimize, `False` = maximize.     |
| `direction`                 | `'minimize'` | String direction flag (alternative form).                     |
| `integer_split`             | `None`  | Index where integer variables begin in the decision vector.        |

---

## Adapting to a domain-specific model

The library knows nothing about your problem. Wrap your domain code so it
exposes a `func(x: np.ndarray) -> float` and a list of bounds. Penalty terms,
repair operators and constraint handling all live in your project.

Minimal pattern:

```python
def make_objective(model):
    def f(x):
        try:
            return float(model.evaluate(x))
        except (ValueError, RuntimeError):
            return float("inf")  # treat infeasibility as worst possible cost
    return f

result = grasp_ils_vnd_pr(make_objective(my_model), bounds=my_bounds)
```

For an end-to-end example with a mixed continuous/integer hydropower model,
see the SOG2 adapter in the upstream project repository
(`grasp_ils_vnd_pr.py`).

---

## Comparison with other optimizers

| Library                                  | Sense convention                  | Discrete vars?  | Built-in cache | Built-in time budget |
|------------------------------------------|-----------------------------------|-----------------|----------------|----------------------|
| `scipy.optimize.minimize`                | Always minimize                   | No              | No             | No                   |
| `scipy.optimize.differential_evolution`  | Always minimize                   | Continuous only | No             | Via callback         |
| `scipy.optimize.dual_annealing`          | Always minimize                   | No              | No             | `maxiter` only       |
| `optuna`                                 | Explicit (`direction`)            | Yes             | Per-trial only | Yes (`timeout`)      |
| `pygad`                                  | Always maximize                   | Yes             | No             | No                   |
| **`givp`**                               | Explicit (`minimize`/`direction`) | Yes (mixed)     | LRU cache      | Yes (`time_limit`)   |

---

## Troubleshooting

**`ValueError: each element of upper must be strictly greater than lower`**
A bounds entry has `low >= high`. Even fixed values must use a strictly
positive interval (`(v - 1e-9, v + 1e-9)`) or be removed from the search.

**`ValueError: bounds length (...) does not match num_vars (...)`**
You passed `num_vars` explicitly but the bounds disagree. Drop `num_vars` to
let the library infer it from `bounds`, or fix the mismatch.

**`ValueError: 'minimize' and 'direction' disagree: ...`**
You passed both flags with conflicting values. Use one or the other (or pass
both with matching values).

**Optimization converges to `inf`.**
Your objective is raising or returning `nan`. The wrapper coerces non-finite
values to `+inf` so they are always comparable, but if *every* candidate is
infeasible the algorithm has nothing to improve. Lower `perturbation_strength`,
revisit your bounds, or relax the feasibility logic in `func`.

**Run is too slow.**
Try `use_cache=True`, increase `cache_size`, raise `n_workers`, lower
`num_candidates_per_step`, or set a `time_limit`. For very expensive
objectives, also reduce `vnd_iterations` and `ils_iterations`.

**Final solution looks too "rough" / integer values look noisy.**
Make sure `integer_split` is set correctly. With the default (`None` /
`num_vars`) all variables are treated as continuous and the integer-aware
neighborhoods are skipped.

---

## License

MIT
# givp — GRASP-ILS-VND with Path Relinking

A direction-agnostic, NumPy-native metaheuristic optimizer for **continuous,
integer or mixed** black-box problems. Combines:

- **GRASP** — Greedy Randomized Adaptive Search Procedure
- **ILS** — Iterated Local Search
- **VND** — Variable Neighborhood Descent (with adaptive variant)
- **Path Relinking** — between elite solutions
- LRU evaluation cache, convergence monitor, optional thread-parallel candidate
  evaluation, time budget

## Install

From PyPI (once published):

```bash
pip install givp
```

From source (editable):

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr.git
cd grasp_ils_vnd_pr
pip install -e .[dev]
```

## Quick start — minimization (SciPy-style)

```python
import numpy as np
from givp import grasp_ils_vnd_pr

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

bounds = [(-5.0, 5.0)] * 10
result = grasp_ils_vnd_pr(sphere, bounds, direction="minimize")
print(result.x, result.fun)
```

## Quick start — maximization

The library is **agnostic** to the optimization sense. Pass
`direction="maximize"` and `result.fun` is returned in your original sign.

```python
from givp import grasp_ils_vnd_pr

def gain(x):
    return -float((x ** 2).sum())  # higher is better

result = grasp_ils_vnd_pr(gain, [(-5, 5)] * 10, direction="maximize")
assert result.direction == "maximize"
```

## Object-oriented API (multi-start friendly)

```python
from givp import GraspIlsVndConfig, GraspOptimizer

opt = GraspOptimizer(
    func=sphere,
    bounds=[(-5.0, 5.0)] * 10,
    direction="minimize",
    config=GraspIlsVndConfig(max_iterations=50, time_limit=30.0),
    verbose=True,
)
for _ in range(5):
    opt.run()
print("best:", opt.best_fun)
```

## API

- `grasp_ils_vnd_pr(func, bounds, *, direction='minimize', config=None,
  initial_guess=None, iteration_callback=None, verbose=False) -> OptimizeResult`
- `GraspOptimizer(func, bounds, *, ...)` with `.run() -> OptimizeResult` and
  `.best_x`, `.best_fun`, `.history`.
- `GraspIlsVndConfig` — dataclass with all hyper-parameters.
- `OptimizeResult` — `x`, `fun`, `nit`, `nfev`, `success`, `message`,
  `direction`, `meta`. Iterable as `(x, fun)` for legacy unpacking.

`bounds` accepts either a list of `(low, high)` pairs **or** a
`(lower, upper)` 2-tuple of equally-sized sequences.

## Adapting to a domain-specific model

Wrap your domain code so it presents a `func(x: np.ndarray) -> float` and a set
of bounds. Anything else (problem-specific decoders, penalty terms, repair
operators) lives in your project. See the upstream SOG2 repository for an
example adapter.

## License

MIT
