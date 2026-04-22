# givp

**GRASP-ILS-VND with Path Relinking** — a direction-agnostic metaheuristic
optimizer for continuous, integer, and mixed-variable problems.

## Highlights

- SciPy-style API: pass an objective and bounds, get an `OptimizeResult`.
- Minimize **or** maximize via the `direction` argument.
- Composes GRASP construction, ILS perturbation, VND local search, and
  bidirectional Path Relinking with an elite pool.
- Optional adaptive α, evaluation cache, convergence monitor, parallel
  candidate evaluation, and warm-start via `initial_guess`.
- Fully typed package (ships `py.typed`).

## Installation

```bash
pip install givp
```

## Minimal example

```python
import numpy as np
from givp import grasp_ils_vnd_pr

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

result = grasp_ils_vnd_pr(sphere, [(-5.0, 5.0)] * 4)
print(result.x, result.fun)
```

See the [Quickstart](quickstart.md) for richer examples and the
[API Reference](api/grasp_ils_vnd_pr.md) for the full surface.
