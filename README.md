# givp — GRASP-ILS-VND with Path Relinking

**Python** &nbsp;
[![PyPI version](https://img.shields.io/pypi/v/givp?cacheSeconds=300)](https://pypi.org/project/givp/)
[![Python versions](https://img.shields.io/badge/python-3.10%E2%80%933.15-blue?logo=python&logoColor=white)](https://pypi.org/project/givp/)
[![CI Python](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-python.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-python.yml)
[![codecov (python)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=python)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/python)
[![Ruff](https://img.shields.io/badge/linter-ruff-red)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy-lang.org/)

**Julia** &nbsp;
[![JuliaHub](https://img.shields.io/badge/JuliaHub-GIVPOptimizer-9558B2?logo=julia&logoColor=white)](https://juliahub.com/ui/Packages/General/GIVPOptimizer)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![CI Julia](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml)
[![codecov (julia)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=julia)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/julia)

**Rust** &nbsp;
[![Crates.io](https://img.shields.io/crates/v/givp?cacheSeconds=300)](https://crates.io/crates/givp)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI Rust](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-rust.yml)
[![codecov (rust)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=rust)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/rust)
[![docs.rs](https://img.shields.io/docsrs/givp?cacheSeconds=300)](https://docs.rs/givp)

**C++** &nbsp;
[![header-only](https://img.shields.io/badge/header--only-yes-brightgreen)](cpp/include/givp/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CI C++](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-cpp.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-cpp.yml)
[![codecov (cpp)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=cpp)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/cpp)

**Project** &nbsp;
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/Arnime/grasp_ils_vnd_pr?cacheSeconds=300)](https://securityscorecards.dev/viewer/?uri=github.com/Arnime/grasp_ils_vnd_pr)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12627/badge)](https://www.bestpractices.dev/projects/12627)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A direction-agnostic metaheuristic optimizer for **continuous, integer or
mixed** black-box problems, available in four languages:

| Language | Distribution | Requires |
|----------|-------------|---------|
| **Python** (NumPy-native) | [PyPI `givp`](https://pypi.org/project/givp/) | Python 3.10+, NumPy |
| **Julia** | [JuliaHub `GIVPOptimizer`](https://juliahub.com/ui/Packages/General/GIVPOptimizer) | Julia 1.9+ |
| **Rust** | [crates.io `givp`](https://crates.io/crates/givp) | Rust 1.85+ |
| **C++17** | Header-only (FetchContent / copy) | C++17 compiler, CMake 3.21+ |

The library bundles:

- **GRASP** — Greedy Randomized Adaptive Search Procedure
- **ILS** — Iterated Local Search
- **VND** — Variable Neighborhood Descent (with an adaptive variant)
- **Path Relinking** between elite solutions
- LRU evaluation cache, convergence monitor, optional thread-parallel candidate
  evaluation, and a wall-clock time budget

---

## Table of contents

- [givp — GRASP-ILS-VND with Path Relinking](#givp--grasp-ils-vnd-with-path-relinking)
  - [Table of contents](#table-of-contents)
  - [Install](#install)
    - [Python installation](#python-installation)
    - [Julia installation](#julia-installation)
    - [Rust installation](#rust-installation)
    - [C++ installation](#c-installation)
  - [Python](#python)
    - [Quick start](#quick-start)
  - [Julia](#julia)
  - [Rust](#rust)
    - [Choosing the optimization sense](#choosing-the-optimization-sense)
      - [Boolean flag (recommended)](#boolean-flag-recommended)
      - [String flag (SciPy/Optuna compatible)](#string-flag-scipyoptuna-compatible)
    - [Bounds, integer variables and mixed problems](#bounds-integer-variables-and-mixed-problems)
    - [Object-oriented API and multi-start](#object-oriented-api-and-multi-start)
    - [Configuration cookbook](#configuration-cookbook)
    - [Inspecting progress (callback and verbose)](#inspecting-progress-callback-and-verbose)
    - [API reference](#api-reference)
      - [`givp(...) -> OptimizeResult`](#givp---optimizeresult)
      - [`class GIVPOptimizer`](#class-givpoptimizer)
      - [`class GIVPConfig` (dataclass)](#class-givpconfig-dataclass)
      - [`class OptimizeResult`](#class-optimizeresult)
    - [Glossary of hyper-parameters](#glossary-of-hyper-parameters)
    - [Adapting to a domain-specific model](#adapting-to-a-domain-specific-model)
  - [C++](#c)
    - [C++ quick start](#c-quick-start)
    - [C++ configuration](#c-configuration)
    - [Running tests and benchmarks](#running-tests-and-benchmarks)
  - [Comparison with other optimizers](#comparison-with-other-optimizers)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

---

## Install

### Python installation

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

### Julia installation

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr.git
cd grasp_ils_vnd_pr/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia 1.9+.

### Rust installation

Add to your `Cargo.toml`:

```toml
[dependencies]
givp = "0.5"
```

From source:

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr.git
cd grasp_ils_vnd_pr/rust
cargo build --release
cargo test
```

Requires Rust 1.85+ (edition 2021).

### C++ installation

The C++ port is **header-only**. The recommended way is CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    givp
    GIT_REPOSITORY https://github.com/Arnime/grasp_ils_vnd_pr.git
    GIT_TAG        v0.5.4
    SOURCE_SUBDIR  cpp
)
FetchContent_MakeAvailable(givp)

target_link_libraries(my_app PRIVATE givp::givp)
```

Or copy the `cpp/include/givp/` directory into your project and add it to your
include path. No external dependencies are required at runtime.

Requires a C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+) and CMake 3.21+
(for the `FetchContent` approach).

---

## Python

### Quick start

```python
import numpy as np
from givp import givp

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

result = givp(sphere, bounds=[(-5.0, 5.0)] * 10)
print(result.x)        # best vector found
print(result.fun)      # best objective value
print(result.nfev)     # number of evaluations performed
```

Default behavior:

- **Minimization** (`minimize=True` / `direction="minimize"`).
- All variables treated as continuous.
- Default hyper-parameters (`GIVPConfig()`).

---

### Choosing the optimization sense

The library is **agnostic** to whether you want the lowest or the highest
value of `func`. Two equivalent ways to declare it:

#### Boolean flag (recommended)

```python
from givp import givp

def gain(x):
    return float((x ** 2).sum())  # higher is better

result = givp(gain, [(-5, 5)] * 10, minimize=False)
assert result.direction == "maximize"
```

#### String flag (SciPy/Optuna compatible)

```python
result = givp(gain, [(-5, 5)] * 10, direction="maximize")
```

Both flags are accepted on `givp`, on `GIVPOptimizer` and on
`GIVPConfig`. Setting **both** simultaneously is allowed only when they
agree; conflicting values raise `ValueError`.

> **Internal note.** The core algorithm always minimizes. When you ask for
> maximization the public API wraps your objective with a sign flip and
> restores the sign on `result.fun`. This means `result.fun` is always
> reported in your original sign — no need to negate it back yourself.

---

### Bounds, integer variables and mixed problems

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
from givp import GIVPConfig, givp

n_cont, n_int = 12, 8
bounds = [(-5.0, 5.0)] * n_cont + [(0.0, 4.0)] * n_int

cfg = GIVPConfig(integer_split=n_cont)  # indices >= n_cont are integer

result = givp(my_objective, bounds, config=cfg)
```

Special cases:

| `integer_split` | Meaning                                |
|-----------------|----------------------------------------|
| `None` (public API default: `num_vars`) | All-continuous problem.    |
| `0`             | All-integer problem.                   |
| `n_vars`        | All-continuous problem (explicit).     |
| `k` (0 < k < n) | First `k` continuous, rest integer.    |

---

### Object-oriented API and multi-start

When you want to keep configuration around, run the optimizer multiple times
and track the best result automatically, use `GIVPOptimizer`:

```python
from givp import GIVPConfig, GIVPOptimizer

opt = GIVPOptimizer(
    func=sphere,
    bounds=[(-5.0, 5.0)] * 10,
    minimize=True,
    config=GIVPConfig(max_iterations=50, time_limit=30.0),
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

### Configuration cookbook

```python
from givp import GIVPConfig

# 1) Fast triage (small budget, no warm-up)
cfg_fast = GIVPConfig(
    max_iterations=20,
    vnd_iterations=50,
    ils_iterations=5,
    use_elite_pool=False,
    use_convergence_monitor=False,
    use_cache=True,
)

# 2) Production-quality run with wall-clock budget
cfg_quality = GIVPConfig(
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
cfg_expensive = GIVPConfig(
    num_candidates_per_step=8,
    cache_size=50_000,
    use_cache=True,
    early_stop_threshold=40,  # stop earlier on stagnation
)

# 4) Maximization with hourly-shaped layout (3 plants × 24 hours)
cfg_hydro = GIVPConfig(
    minimize=False,
    integer_split=72,         # first 72 vars continuous, rest integer
    max_iterations=120,
    time_limit=300.0,
)
```

---

### Inspecting progress (callback and verbose)

Both `givp` and `GIVPOptimizer` accept:

- `verbose=True` — prints per-iteration cost and cache statistics.
- `iteration_callback=fn` — calls `fn(iteration_index, best_cost, best_solution)`
  once per outer GRASP iteration. The callback receives the cost in the
  **internal minimization sign** (i.e., already sign-flipped if you asked for
  maximization). Useful to plot convergence or persist intermediate results.

```python
costs = []

def log_iter(i, cost, sol):
    costs.append(cost)

result = givp(
    sphere,
    [(-5, 5)] * 10,
    iteration_callback=log_iter,
    verbose=True,
)
```

---

### API reference

#### `givp(...) -> OptimizeResult`

```python
givp(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]] | tuple[Sequence[float], Sequence[float]],
    *,
    num_vars: int | None = None,
    minimize: bool | None = None,
    direction: str | None = None,         # 'minimize' or 'maximize'
    config: GIVPConfig | None = None,
    initial_guess: Sequence[float] | None = None,
    iteration_callback: Callable[[int, float, np.ndarray], None] | None = None,
    verbose: bool = False,
) -> OptimizeResult
```

#### `class GIVPOptimizer`

Same constructor signature, exposes `.run() -> OptimizeResult` and tracks
`.best_x`, `.best_fun`, `.history`.

#### `class GIVPConfig` (dataclass)

All hyper-parameters listed in the [glossary](#glossary-of-hyper-parameters).

#### `class OptimizeResult`

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

### Glossary of hyper-parameters

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

### Adapting to a domain-specific model

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

result = givp(make_objective(my_model), bounds=my_bounds)
```

For an end-to-end example with a mixed continuous/integer hydropower model,
see the SOG2 adapter in the upstream project repository
(`givp.py`).

---

## Julia

The Julia port exposes the same algorithm with an idiomatic Julia API:

```julia
using GIVPOptimizer

function sphere(x::Vector{Float64})::Float64
    return sum(x .^ 2)
end

result = givp(sphere, [(-5.0, 5.0) for _ in 1:10])
println(result.x)       # best vector found
println(result.fun)     # best objective value
println(result.nfev)    # number of evaluations
```

Maximization:

```julia
result = givp(my_score, bounds; direction=maximize)
```

Configuration:

```julia
cfg = GIVPConfig(; max_iterations=50, vnd_iterations=100, time_limit=30.0)
result = givp(sphere, bounds; config=cfg, seed=42, verbose=true)
```

Running tests:

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

Running benchmarks:

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

---

## Rust

The Rust port provides a zero-dependency-on-NumPy, native-performance implementation:

```rust
use givp::{givp, GivpConfig};

let sphere = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
let bounds: Vec<(f64, f64)> = vec![(-5.12, 5.12); 5];

let config = GivpConfig {
    max_iterations: 50,
    seed: Some(42),
    integer_split: Some(5), // all continuous
    ..Default::default()
};

let result = givp(sphere, &bounds, config).unwrap();
println!("Best: {:.6} at {:?}", result.fun, result.x);
```

Maximization:

```rust
use givp::{givp, GivpConfig, Direction};

let config = GivpConfig {
    direction: Direction::Maximize,
    ..Default::default()
};
```

Running tests:

```bash
cd rust
cargo test
```

Running benchmarks:

```bash
cd rust
cargo bench
```

---

## C++

The C++ port is a **header-only, zero-runtime-dependency** implementation.
It uses the same algorithm and identical default hyper-parameters as the
Python, Julia, and Rust ports.

### C++ quick start

```cpp
#include <givp/givp.hpp>
#include <iostream>
#include <vector>

int main() {
    auto sphere = [](const std::vector<double>& x) {
        double s = 0.0;
        for (auto v : x) s += v * v;
        return s;
    };

    std::vector<std::pair<double, double>> bounds(10, {-5.0, 5.0});
    givp::OptimizeResult r = givp::givp(sphere, bounds);

    std::cout << "best: " << r.fun  << "\n";  // best objective value
    std::cout << "nfev: " << r.nfev << "\n";  // evaluations used
    std::cout << "nit:  " << r.nit  << "\n";  // outer iterations
}
```

Maximization:

```cpp
givp::GivpConfig cfg;
cfg.direction = givp::Direction::Maximize;
auto result = givp::givp(my_func, bounds, cfg);
```

### C++ configuration

```cpp
givp::GivpConfig cfg;
cfg.max_iterations          = 50;
cfg.vnd_iterations          = 100;
cfg.time_limit              = 30.0;          // wall-clock seconds
cfg.seed                    = 42;
cfg.integer_split           = 8;             // first 8 vars continuous
cfg.verbose                 = true;

givp::OptimizeResult r = givp::givp(sphere, bounds, cfg);
```

`OptimizeResult` fields mirror all other ports:

| Field     | Type                    | Meaning                                          |
|-----------|-------------------------|--------------------------------------------------|
| `x`       | `std::vector<double>`   | Best solution vector.                            |
| `fun`     | `double`                | Objective value at `x`, in the original sign.    |
| `nit`     | `std::size_t`           | Outer GRASP iterations executed.                 |
| `nfev`    | `std::size_t`           | Total objective-function evaluations.            |
| `success` | `bool`                  | True when `fun` is finite.                       |
| `message` | `std::string`           | Human-readable termination reason.               |

### Running tests and benchmarks

```bash
# Configure and build
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_TESTS=ON
cmake --build build

# Run the 41 Catch2 test cases
ctest --test-dir build --output-on-failure

# Build and run nanobench benchmarks
cmake -S cpp -B build_bench -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_BENCHMARKS=ON
cmake --build build_bench
./build_bench/benchmarks/givp_benchmarks
```

---

## Comparison with other optimizers

| Library                                  | Sense convention                  | Discrete vars?  | Built-in cache | Built-in time budget | Language          |
|------------------------------------------|-----------------------------------|-----------------|----------------|----------------------|-------------------|
| `scipy.optimize.minimize`                | Always minimize                   | No              | No             | No                   | Python            |
| `scipy.optimize.differential_evolution`  | Always minimize                   | Continuous only | No             | Via callback         | Python            |
| `scipy.optimize.dual_annealing`          | Always minimize                   | No              | No             | `maxiter` only       | Python            |
| `optuna`                                 | Explicit (`direction`)            | Yes             | Per-trial only | Yes (`timeout`)      | Python            |
| `pygad`                                  | Always maximize                   | Yes             | No             | No                   | Python            |
| **`givp`** (Python / Julia / Rust)       | Explicit (`minimize`/`direction`) | Yes (mixed)     | LRU cache      | Yes (`time_limit`)   | Python+Julia+Rust |
| **`givp`** (C++)                         | Explicit (`Direction` enum)       | Yes (mixed)     | LRU cache      | Yes (`time_limit`)   | C++17             |

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
