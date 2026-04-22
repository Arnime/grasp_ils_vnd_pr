# Quickstart

## Minimize a function

```python
import numpy as np
from givp import grasp_ils_vnd_pr, GraspIlsVndConfig

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

cfg = GraspIlsVndConfig(max_iterations=20, vnd_iterations=30)
result = grasp_ils_vnd_pr(rosenbrock, [(-2.0, 2.0)] * 5, config=cfg)
print(result)
```

## Maximize

```python
result = grasp_ils_vnd_pr(my_score, bounds, direction="maximize")
```

## Object-oriented API

`GraspOptimizer` keeps the run history when you call `.run()` repeatedly:

```python
from givp import GraspOptimizer

opt = GraspOptimizer(rosenbrock, [(-2.0, 2.0)] * 5)
opt.run()
opt.run()
print(opt.best_x, opt.best_fun, len(opt.history))
```

## Warm start

```python
result = grasp_ils_vnd_pr(
    rosenbrock,
    [(-2.0, 2.0)] * 5,
    initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
)
```

## Stopping by wall clock

```python
cfg = GraspIlsVndConfig(time_limit=2.0)  # seconds
result = grasp_ils_vnd_pr(rosenbrock, [(-2.0, 2.0)] * 5, config=cfg)
```
