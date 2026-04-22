# Real-world examples

Four end-to-end recipes you can paste into a script. Each one runs in under
10 seconds with default settings.

## 1. Continuous: Rastrigin (multimodal, 30D)

A classic benchmark that defeats naive gradient methods because it has
hundreds of local minima.

```python
import numpy as np

from givp import GraspIlsVndConfig, grasp_ils_vnd_pr


def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return 10.0 * n + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


bounds = [(-5.12, 5.12)] * 30
result = grasp_ils_vnd_pr(
    rastrigin,
    bounds,
    seed=42,
    config=GraspIlsVndConfig(max_iterations=80, ils_iterations=10),
)
print(f"best={result.fun:.4f}  nfev={result.nfev}")
```

## 2. Integer: 0/1 Knapsack

Maximise total value subject to a weight budget. The objective uses a heavy
penalty so infeasible solutions are pushed away naturally.

```python
import numpy as np

from givp import GraspIlsVndConfig, grasp_ils_vnd_pr

values = np.array([60, 100, 120, 80, 30, 70, 45, 90])
weights = np.array([10, 20, 30, 15, 5, 18, 8, 25])
capacity = 60


def knapsack(x: np.ndarray) -> float:
    chosen = np.rint(x).astype(int)
    total_w = float((chosen * weights).sum())
    total_v = float((chosen * values).sum())
    if total_w > capacity:
        return -1e6  # heavy penalty (we are maximizing value)
    return total_v


bounds = [(0.0, 1.0)] * len(values)
result = grasp_ils_vnd_pr(
    knapsack,
    bounds,
    minimize=False,
    seed=7,
    config=GraspIlsVndConfig(integer_split=0, max_iterations=50),
)
print("picked:", np.rint(result.x).astype(int))
print(f"value={result.fun:.0f}")
```

## 3. Mixed continuous + integer: portfolio with discrete lots

Optimize 5 continuous weights *plus* 3 integer "lot counts". `integer_split=5`
tells `givp` that indices ``[5, 6, 7]`` must stay integer.

```python
import numpy as np

from givp import GraspIlsVndConfig, grasp_ils_vnd_pr

rng = np.random.default_rng(0)
mu = rng.uniform(0.05, 0.20, size=5)
cov = np.diag(rng.uniform(0.01, 0.05, size=5))


def portfolio(x: np.ndarray) -> float:
    weights, lots = x[:5], x[5:]
    weights = np.abs(weights)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    expected_return = float(weights @ mu) + 0.001 * float(np.rint(lots).sum())
    risk = float(weights @ cov @ weights)
    # Sharpe-like: maximize return per unit of risk.
    return expected_return / np.sqrt(risk + 1e-12)


bounds = [(0.0, 1.0)] * 5 + [(0, 10)] * 3
result = grasp_ils_vnd_pr(
    portfolio,
    bounds,
    minimize=False,
    seed=11,
    config=GraspIlsVndConfig(integer_split=5, max_iterations=60),
)
print("weights :", np.round(result.x[:5], 3))
print("lots    :", np.rint(result.x[5:]).astype(int))
print(f"sharpe~ {result.fun:.4f}")
```

## 4. Black-box: hyper-parameter tuning of an ML model

Plug any `cross_val_score` into the objective. Each evaluation can be
expensive, so enable the cache and a wall-clock budget.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from givp import GraspIlsVndConfig, grasp_ils_vnd_pr

X, y = load_iris(return_X_y=True)


def objective(params: np.ndarray) -> float:
    learning_rate, max_depth, n_estimators = params
    model = GradientBoostingClassifier(
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        random_state=0,
    )
    return float(cross_val_score(model, X, y, cv=3).mean())


# learning_rate continuous, max_depth + n_estimators integer
bounds = [(0.01, 0.5), (2, 10), (10, 200)]
result = grasp_ils_vnd_pr(
    objective,
    bounds,
    minimize=False,
    seed=2024,
    config=GraspIlsVndConfig(
        integer_split=1,
        max_iterations=20,
        time_limit=60.0,
        use_cache=True,
    ),
)
print(f"best CV accuracy: {result.fun:.4f}")
print(f"hyperparams     : lr={result.x[0]:.3f}, depth={int(result.x[1])}, n={int(result.x[2])}")
```

## See also

* [Quickstart](quickstart.md) — minimal first run.
* [Algorithm overview](algorithm.md) — what each component does.
* [Profiling](profiling.md) — measuring and improving performance.
