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

## Experimental protocol (30-seed sweep)

For publication-quality comparisons, run the full 30-seed sweep from the
`python/` directory:

```bash
# Step 1 — run experiment (≈ 30 min on a laptop)
python benchmarks/run_literature_comparison.py \
    --dims 10 --n-runs 30 \
    --algorithms GIVP-full GRASP-only DE SA \
    --traces --verbose \
    --output results/comparison_10d_30runs.json

# Step 2 — generate Markdown + LaTeX tables with Wilcoxon tests
python benchmarks/generate_report.py \
    --input results/comparison_10d_30runs.json \
    --format both \
    --output-dir paper/tables/

# Step 3 — include in paper
#   paper/tables/comparison_10d_30runs_report.tex  → \input{tables/...}
#   paper/tables/comparison_10d_30runs_boxplot.png → \includegraphics{...}
```

> **Seeds**: uses `[0, n_runs)` by default — byte-identical results for any
> given seed. Use `--seed-start N` to shift the window.
> **Resume**: add `--resume` to continue a partial run (checkpointed after
> each function).

See `python/benchmarks/README.md` for the full option reference, including
higher-dimensional sweeps (`--dims 30`) and tuned-config runs
(`--algorithms GIVP-tuned --tune-config best_config.json`).

## Experimental results — GIVP-full vs. GRASP-only (30 seeds, 10-D)

The table below summarises the Julia benchmark (`Notebooks/Julia/results_notebook_julia.json`),
30 independent seeds per algorithm and function, 10 dimensions.
All results are objective function values
(lower = better, global minimum = 0 for all functions).

| Function | GIVP-full mean ± std | GRASP-only mean ± std | W | p-value | Sig |
|---|---|---|---|---|---|
| Sphere | 2.0692e-04 ± 6.3134e-05 | 2.5144e+00 ± 5.6531e-01 | 0.0 | < 0.0001 | ★ |
| Rosenbrock | 4.3611e-01 ± 3.1966e-01 | 1.1314e+04 ± 5.7702e+03 | 0.0 | < 0.0001 | ★ |
| Rastrigin | 9.8794e-01 ± 6.0726e-01 | 3.9875e+01 ± 7.4324e+00 | 0.0 | < 0.0001 | ★ |
| Ackley | 1.3495e-01 ± 2.6316e-02 | 1.0992e+01 ± 7.9834e-01 | 0.0 | < 0.0001 | ★ |
| Griewank | 1.7085e-01 ± 3.7264e-02 | 9.5772e+00 ± 1.7650e+00 | 0.0 | < 0.0001 | ★ |
| Schwefel | 4.9916e+01 ± — | 1.9143e+03 ± 1.6299e+02 | 0.0 | < 0.0001 | ★ |

**Statistical test**: two-sided Wilcoxon signed-rank test (α = 0.05).  
**★** = statistically significant difference (p < 0.05) in favour of GIVP-full.
**Metadata**: Julia 1.12.6, GIVPOptimizer v1.0.0, generated 2026-04-29.
