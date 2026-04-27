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
- **CLI entry point** (`givp run`) — run the optimizer from the shell or
  pipe structured JSON output to AI agents and LLM tool calls.
- **Typed result schema** — `OptimizeResult.to_dict()` returns a JSON-safe
  dict with a closed `TerminationReason` enum, eliminating prompt-injection
  risk when output is passed to language models.
- Fully typed package (ships `py.typed`).

## Installation

```bash
pip install givp
```

## Minimal example

```python
import numpy as np
from givp import givp

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

result = givp(sphere, [(-5.0, 5.0)] * 4)
print(result.x, result.fun)
```

See the [Quickstart](quickstart.md) for richer examples and the
[API Reference](api/grasp_ils_vnd_pr.md) for the full surface.
