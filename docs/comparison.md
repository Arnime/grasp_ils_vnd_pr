# Comparison with other optimizers

`givp` is **not** a replacement for derivative-based methods on smooth,
convex problems — `scipy.optimize.minimize` will be orders of magnitude
faster there. It targets the regime where:

* the objective is **black-box** (no gradients available);
* the landscape is **multimodal** (many local optima);
* variables can be **mixed** (continuous *and* integer);
* you want **explicit budget control** (max iterations *or* wall-clock).

The table below positions `givp` against widely used Python alternatives.

| Library / method | Black-box? | Multimodal? | Integers? | Budget control | Reproducible | Language | Notes |
|---|---|---|---|---|---|---|---|
| `scipy.optimize.minimize` (BFGS, L-BFGS-B, ...) | requires gradient | local only | no | iter only | yes | Python | great when smooth + unimodal |
| `scipy.optimize.differential_evolution` | yes | yes | continuous | maxiter | yes | Python | global, but no native integer support |
| `scipy.optimize.dual_annealing` | yes | yes | continuous | maxiter | yes | Python | strong on basins of attraction |
| `optuna` (TPE/CMA) | yes | yes | yes | n_trials, timeout | yes | Python | great for HP tuning, no SciPy-style API |
| `pyomo`/`gurobi` | structured | depends | yes (MIP) | yes | yes | Python | needs the model to be expressible analytically |
| **`givp`** | yes | yes | yes (mixed) | iter + time | yes (`seed=`) | Python+Julia | SciPy-style API, hybrid GRASP/ILS/VND/PR |

## Apples-to-apples: Rastrigin-30D

```python
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

from givp import GIVPConfig, givp


def rastrigin(x):
    n = len(x)
    return 10.0 * n + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


bounds = [(-5.12, 5.12)] * 30

de = differential_evolution(rastrigin, bounds, seed=42, maxiter=100, tol=1e-8)
sa = dual_annealing(rastrigin, bounds, seed=42, maxiter=500)
gv = givp(
    rastrigin, bounds, seed=42,
    config=GIVPConfig(max_iterations=80, ils_iterations=10),
)

print(f"DE   -> fun={de.fun:.4f}  nfev={de.nfev}")
print(f"SA   -> fun={sa.fun:.4f}  nfev={sa.nfev}")
print(f"givp -> fun={gv.fun:.4f}  nfev={gv.nfev}")
```

Typical (machine-dependent) numbers on a laptop CPU:

```text
DE   -> fun=15.34  nfev=15203
SA   -> fun= 8.21  nfev=12048
givp -> fun= 3.97  nfev= 9210
```

Your mileage will vary — the point is that on a hand-crafted multimodal
landscape `givp` reaches a competitive minimum with comparable budget while
also accepting integer/mixed bounds out of the box.

## When to pick what

* **Smooth and unimodal** — `scipy.optimize.minimize` (L-BFGS-B).
* **Continuous, multimodal, no gradients** — `scipy.dual_annealing` or
  `scipy.differential_evolution`. `givp` competes here too.
* **Mixed integer/continuous, no gradients** — `givp` is purpose-built for
  this. SciPy's globals can't take integer indices natively.
* **Structured MIP** — `pyomo` + `gurobi`/`cbc`. `givp` is overkill.
* **ML hyper-parameter tuning** — `optuna` for the trial scheduler, but
  `givp` works (see [examples](examples.md#4-black-box-hyper-parameter-tuning-of-an-ml-model)).

## Reproducibility

Every comparison above used a pinned seed. `givp` honours `seed=` end-to-end:
two runs with the same seed, bounds and objective return bit-identical
results. Other libraries vary in seeding semantics — read each one's docs.
