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
git clone https://github.com/TBD/grasp_ils_vnd_pr.git
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
