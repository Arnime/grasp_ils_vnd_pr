# givp — GRASP-ILS-VND with Path Relinking

**Python** &nbsp;
[![PyPI: givp v1.0.0](https://img.shields.io/badge/PyPI-givp%20v1.0.0-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/givp/)
[![Python versions](https://img.shields.io/badge/Python-3.10%E2%80%933.15-blue?logo=python&logoColor=white)](https://pypi.org/project/givp/)
[![CI Python](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-python.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-python.yml)
[![Codecov (python)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=python)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/python)
[![Ruff](https://img.shields.io/badge/Linter-ruff-red?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/Type--Checked-mypy-blue?logo=python&logoColor=white)](https://mypy-lang.org/)

**Julia** &nbsp;
[![JuliaHub: GIVPOptimizer v1.0.0](https://img.shields.io/badge/JuliaHub-GIVPOptimizer%20v1.0.0-9558B2?logo=julia&logoColor=white)](https://juliahub.com/ui/Packages/General/GIVPOptimizer)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![CI Julia](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-julia.yml)
[![Codecov (julia)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=julia)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/julia)
[![JuliaFormatter](https://img.shields.io/badge/Formatter-JuliaFormatter-9558B2?logo=julia&logoColor=white)](https://github.com/domluna/JuliaFormatter.jl)
[![JET](https://img.shields.io/badge/Type--Checked-JET.jl-9558B2?logo=julia&logoColor=white)](https://github.com/aviatesk/JET.jl)
[![Aqua](https://img.shields.io/badge/Quality-Aqua.jl-9558B2?logo=julia&logoColor=white)](https://github.com/JuliaTesting/Aqua.jl)

**Rust** &nbsp;
[![crates.io: givp v1.0.0](https://img.shields.io/badge/Crates-givp%20v1.0.0-E05D44?logo=rust&logoColor=white)](https://crates.io/crates/givp)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CI Rust](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-rust.yml)
[![Codecov (rust)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=rust)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/rust)
[![docs.rs](https://img.shields.io/docsrs/givp?cacheSeconds=300&logo=docs.rs&logoColor=white)](https://docs.rs/givp)
[![Clippy](https://img.shields.io/badge/Linter-Clippy-E05D44?logo=rust&logoColor=white)](https://doc.rust-lang.org/clippy/)
[![rustfmt](https://img.shields.io/badge/Formatter-rustfmt-E05D44?logo=rust&logoColor=white)](https://github.com/rust-lang/rustfmt)

**C++** &nbsp;
[![header-only](https://img.shields.io/badge/header--only-yes-brightgreen?logo=cplusplus&logoColor=white)](cpp/include/givp/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CI C++](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-cpp.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-cpp.yml)
[![Codecov (cpp)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/graph/badge.svg?flag=cpp)](https://codecov.io/gh/Arnime/grasp_ils_vnd_pr/flags/cpp)
[![clang-format](https://img.shields.io/badge/Formatter-clang--format-blue?logo=llvm&logoColor=white)](https://clang.llvm.org/docs/ClangFormat.html)
[![clang-tidy](https://img.shields.io/badge/Linter-clang--tidy-blue?logo=llvm&logoColor=white)](https://clang.llvm.org/extra/clang-tidy/)

**Project** &nbsp;
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/Arnime/grasp_ils_vnd_pr?cacheSeconds=300&logo=openssf&logoColor=white)](https://securityscorecards.dev/viewer/?uri=github.com/Arnime/grasp_ils_vnd_pr)
[![CI SonarQube](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-sonarqube.yml/badge.svg)](https://github.com/Arnime/grasp_ils_vnd_pr/actions/workflows/ci-sonarqube.yml)
[![OpenSSF Gold](https://img.shields.io/badge/OpenSSF-best%20practices-gold?logo=openssf&logoColor=white)](https://www.bestpractices.dev/projects/12627)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github&logoColor=white)](CONTRIBUTING.md)

A direction-agnostic metaheuristic optimizer for **continuous, integer or
mixed** black-box problems, available in five languages:

| Language | Distribution | Requires |
|----------|-------------|---------|
| **Python** (NumPy-native) | [PyPI `givp`](https://pypi.org/project/givp/) | Python 3.10+, NumPy |
| **Julia** | [JuliaHub `GIVPOptimizer`](https://juliahub.com/ui/Packages/General/GIVPOptimizer) | Julia 1.9+ |
| **Rust** | [crates.io `givp`](https://crates.io/crates/givp) | Rust 1.85+ |
| **C++17** | Header-only (vcpkg / FetchContent / copy) | C++17 compiler, CMake 3.21+ |
| **R** | Local package (`r/`) | R 4.1+ |

The library bundles:

- **GRASP** — Greedy Randomized Adaptive Search Procedure
- **ILS** — Iterated Local Search
- **VND** — Variable Neighborhood Descent (with an adaptive variant)
- **Path Relinking** between elite solutions
- LRU evaluation cache, convergence monitor, optional thread-parallel candidate
  evaluation, and a wall-clock time budget

Code quality is enforced in CI with language-specific static analysis and
formatting checks — analogous to mypy + ruff in Python — across all ports:

| Language | Linter / static analysis | Formatter | Type checker |
|----------|--------------------------|-----------|-------------|
| **Python** | [ruff](https://github.com/astral-sh/ruff) | ruff | [mypy](https://mypy-lang.org/) (strict) |
| **Julia** | [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) | [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl) | [JET.jl](https://github.com/aviatesk/JET.jl) |
| **Rust** | [Clippy](https://doc.rust-lang.org/clippy/) (`-D warnings`, all targets) | [rustfmt](https://github.com/rust-lang/rustfmt) | rustc (type system) |
| **C++** | [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) (`--warnings-as-errors`) | [clang-format](https://clang.llvm.org/docs/ClangFormat.html) (LLVM style) | compiler (`-Wall -Wextra -Wpedantic -Werror`) |
| **R** | [lintr](https://lintr.r-lib.org/) (`linters_with_defaults()`) | — | — |

All checks run as mandatory CI gates. A monorepo SonarQube quality gate
workflow runs in addition (CI SonarQube).

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
  - [Julia](#julia)
    - [Julia quick start](#julia-quick-start)
    - [Julia bounds and integer variables](#julia-bounds-and-integer-variables)
    - [Julia configuration cookbook](#julia-configuration-cookbook)
    - [Julia result fields](#julia-result-fields)
    - [Julia progress monitoring](#julia-progress-monitoring)
    - [Running Julia tests and benchmarks](#running-julia-tests-and-benchmarks)
  - [Rust](#rust)
    - [Rust quick start](#rust-quick-start)
    - [Rust bounds and integer variables](#rust-bounds-and-integer-variables)
    - [Rust configuration cookbook](#rust-configuration-cookbook)
    - [Rust result fields](#rust-result-fields)
    - [Running Rust tests and benchmarks](#running-rust-tests-and-benchmarks)
  - [C++](#c)
    - [C++ quick start](#c-quick-start)
    - [C++ configuration](#c-configuration)
    - [Running tests and benchmarks](#running-tests-and-benchmarks)
  - [Comparison with other optimizers](#comparison-with-other-optimizers)
  - [Empirical results](#empirical-results)
    - [Sphere — $f(\\mathbf{x}) = \\sum x\_i^2$, global minimum 0 at **0**](#sphere--fmathbfx--sum-x_i2-global-minimum-0-at-0)
    - [Rosenbrock — global minimum 0 at **1**](#rosenbrock--global-minimum-0-at-1)
    - [Rastrigin — multimodal, global minimum 0 at **0**](#rastrigin--multimodal-global-minimum-0-at-0)
    - [Ackley — multimodal, global minimum 0 at **0**](#ackley--multimodal-global-minimum-0-at-0)
      - [**References**](#references)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

---

## Install

### Python installation

```bash
pip install givp
```

> **Recommended:** install with the optional `cache` extra to use
> [xxhash](https://github.com/ifduyue/python-xxhash) for a ~10× faster
> evaluation cache:
>
> ```bash
> pip install givp[cache]
> ```
>
> Without `xxhash`, the cache falls back to Python's built-in `hashlib`
> (SHA-256), which works correctly but is slower.

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
givp = "1.0.0"
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

The C++ port is **header-only**.

Install with vcpkg:

```bash
vcpkg install givp
```

Then use in CMake:

```cmake
find_package(givp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE givp::givp)
```

Alternatively, use CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    givp
    GIT_REPOSITORY https://github.com/Arnime/grasp_ils_vnd_pr.git
    GIT_TAG        v1.0.0
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
    seed: int | None = None,
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
| `n_workers`                 | 1       | Threads used to evaluate candidates concurrently in the GRASP construction phase. Set to `os.cpu_count()` or a fixed value (2–4) for CPU-bound objectives. Due to the Python GIL, gains are most visible when `func` releases the GIL (e.g. NumPy-heavy code). |
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

The Julia port exposes the same algorithm with an idiomatic Julia API.
Install via the [Julia installation](#julia-installation) instructions above.

### Julia quick start

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

### Julia bounds and integer variables

`bounds` is a vector of `(lower, upper)` tuples, one per variable.
Use `integer_split` to declare mixed continuous/integer problems:

```julia
n_cont, n_int = 8, 4
bounds = vcat(fill((-5.0, 5.0), n_cont), fill((0.0, 3.0), n_int))
cfg = GIVPConfig(; integer_split=n_cont)          # indices >= n_cont are integer
result = givp(my_obj, bounds; config=cfg)
```

### Julia configuration cookbook

```julia
# 1) Fast triage
cfg_fast = GIVPConfig(;
    max_iterations=20, vnd_iterations=50, ils_iterations=5,
    use_elite_pool=false, use_convergence_monitor=false,
)

# 2) Production-quality with wall-clock budget
cfg_quality = GIVPConfig(;
    max_iterations=200, vnd_iterations=300, ils_iterations=15,
    elite_size=10, path_relink_frequency=5,
    adaptive_alpha=true, alpha_min=0.05, alpha_max=0.20,
    time_limit=600.0,
)

# 3) Maximization
cfg_max = GIVPConfig(; direction=maximize, max_iterations=100)

result = givp(sphere, bounds; config=cfg_quality, seed=42, verbose=true)
```

### Julia result fields

| Field       | Type               | Meaning                                          |
|-------------|--------------------|--------------------------------------------------|
| `x`         | `Vector{Float64}`  | Best solution vector.                            |
| `fun`       | `Float64`          | Objective value at `x`, in the original sign.    |
| `nit`       | `Int`              | Outer GRASP iterations executed.                 |
| `nfev`      | `Int`              | Total objective-function evaluations.            |
| `success`   | `Bool`             | True when `fun` is finite.                       |
| `message`   | `String`           | Human-readable termination reason.               |
| `direction` | `Direction`        | `minimize` or `maximize` (enum).                 |
| `meta`      | `Dict{String,Any}` | Algorithm-specific extras (cache stats, etc.).   |

Tuple unpacking works: `x, fun = result`.

### Julia progress monitoring

```julia
costs = Float64[]

function log_iter(i, cost, sol)
    push!(costs, cost)
end

result = givp(sphere, bounds; iteration_callback=log_iter, verbose=true)
```

### Running Julia tests and benchmarks

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. benchmarks/benchmarks.jl
```

---

## Rust

The Rust port is a zero-dependency (no NumPy), native-performance implementation
suitable for embedding in systems code or for maximum throughput.
Install via the [Rust installation](#rust-installation) instructions above.

### Rust quick start

```rust
use givp::{givp, GivpConfig};

let sphere = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
let bounds: Vec<(f64, f64)> = vec![(-5.12, 5.12); 5];

let result = givp(sphere, &bounds, GivpConfig::default()).unwrap();
println!("best: {:.6}", result.fun);
println!("x:    {:?}", result.x);
println!("nfev: {}", result.nfev);
```

### Rust bounds and integer variables

Bounds are a `&[(f64, f64)]` slice. Use `integer_split` for mixed problems:

```rust
use givp::{givp, GivpConfig};

let n_cont = 8usize;
let mut bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); n_cont];
bounds.extend(vec![(0.0, 3.0); 4]);  // 4 integer variables

let config = GivpConfig {
    integer_split: Some(n_cont),  // indices >= n_cont are rounded to int
    ..Default::default()
};
let result = givp(my_obj, &bounds, config).unwrap();
```

### Rust configuration cookbook

```rust
use givp::{givp, GivpConfig, Direction};

// Maximization
let cfg_max = GivpConfig {
    direction: Direction::Maximize,
    seed: Some(42),
    ..Default::default()
};

// Production-quality run with wall-clock budget
let cfg_quality = GivpConfig {
    max_iterations: 200,
    vnd_iterations: 300,
    ils_iterations: 15,
    elite_size: 7,
    path_relink_frequency: 5,
    adaptive_alpha: true,
    alpha_min: 0.05,
    alpha_max: 0.20,
    time_limit: 600.0,
    seed: Some(0),
    ..Default::default()
};

let result = givp(sphere, &bounds, cfg_quality).unwrap();
```

### Rust result fields

| Field       | Type         | Meaning                                         |
|-------------|--------------|-------------------------------------------------|
| `x`         | `Vec<f64>`   | Best solution vector.                           |
| `fun`       | `f64`        | Objective value at `x`, in the original sign.   |
| `nit`       | `usize`      | Outer GRASP iterations executed.                |
| `nfev`      | `usize`      | Total objective-function evaluations.           |
| `success`   | `bool`       | True when `fun` is finite.                      |
| `message`   | `String`     | Human-readable termination reason.              |
| `termination` | `TerminationReason` | Typed termination enum.               |

The function returns `Result<OptimizeResult, GivpError>` — use `.unwrap()` for
quick scripts or pattern-match for production code.

### Running Rust tests and benchmarks

```bash
cd rust
cargo test
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

## Empirical results

Results of **30 independent runs** (seeds 0–29) on four standard continuous benchmarks
(`n = 10` dimensions) comparing the full GIVP pipeline against a GRASP-only baseline
(construction phase only, no ILS/VND/path relinking — equivalent to plain GRASP
as described by Feo & Resende, 1995).

Each run uses an explicit seed for full reproducibility.  To reproduce, run the
[`Notebooks/benchmark_literature_comparison.ipynb`](Notebooks/benchmark_literature_comparison.ipynb)
notebook.

> **Note**: Values below are representative results from the notebook execution.
> Run the notebook on your machine to obtain hardware-specific timings.

### Sphere — $f(\mathbf{x}) = \sum x_i^2$, global minimum 0 at **0**

| Algorithm   | Mean ± Std          | Best         | Median       | NFev (mean) |
|-------------|---------------------|--------------|--------------|-------------|
| GIVP-full   | 0.0002 ± 0.0001     | 6.2935e-05   | 1.9862e-04   | 605 626     |
| GRASP-only  | 2.1568 ± 0.4827     | 1.0564e+00   | 2.1947e+00   | 4 828       |

### Rosenbrock — global minimum 0 at **1**

| Algorithm   | Mean ± Std          | Best         | Median       | NFev (mean) |
|-------------|---------------------|--------------|--------------|-------------|
| GIVP-full   | 0.513 ± 0.325       | 1.0413e-02   | 5.1000e-01   | 571 765     |
| GRASP-only  | 5441.154 ± 2232.312 | 1.6380e+03   | 5.3735e+03   | 4 834       |

### Rastrigin — multimodal, global minimum 0 at **0**

| Algorithm   | Mean ± Std          | Best         | Median       | NFev (mean) |
|-------------|---------------------|--------------|--------------|-------------|
| GIVP-full   | 0.8492 ± 0.6247     | 9.1427e-03   | 1.0144e+00   | 524 825     |
| GRASP-only  | 28.0927 ± 4.9224    | 1.8621e+01   | 2.9218e+01   | 4 898       |

### Ackley — multimodal, global minimum 0 at **0**

| Algorithm   | Mean ± Std          | Best         | Median       | NFev (mean) |
|-------------|---------------------|--------------|--------------|-------------|
| GIVP-full   | 0.1525 ± 0.0277     | 1.0691e-01   | 1.5015e-01   | 477 840     |
| GRASP-only  | 8.9242 ± 1.0037     | 7.2171e+00   | 9.2427e+00   | 4 852       |

#### **References**

- Feo, T.A. & Resende, M.G.C. (1995). *Greedy randomized adaptive search procedures.*
  Journal of Global Optimization, 6, 109–133.
- Festa, P. & Resende, M.G.C. (2011). *GRASP: An annotated bibliography.*
  Essays and Surveys in Metaheuristics. Springer.

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
Try `use_cache=True` (with `givp[cache]` for maximum speed), increase
`cache_size`, raise `n_workers` (2–4 is a practical sweet spot under the
Python GIL; set `n_workers=os.cpu_count()` for NumPy-heavy objectives),
lower `num_candidates_per_step`, or set a `time_limit`. For very expensive
objectives, also reduce `vnd_iterations` and `ils_iterations`.

**Final solution looks too "rough" / integer values look noisy.**
Make sure `integer_split` is set correctly. With the default (`None` /
`num_vars`) all variables are treated as continuous and the integer-aware
neighborhoods are skipped.

---

## License

MIT
