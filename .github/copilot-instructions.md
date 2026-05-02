# GitHub Copilot Instructions — GIVP

This repository is a multi-language implementation of the **GRASP-ILS-VND with Path Relinking** (GIVP) metaheuristic optimizer for continuous and mixed-integer black-box optimization. All language ports must remain functionally equivalent: same algorithm, same public API shape, same module structure.

---

## Project overview

| Language | Location      | Distribution      | Min version |
|----------|---------------|-------------------|-------------|
| Python   | `python/`     | PyPI (`givp`)     | 3.10        |
| Julia    | `julia/`      | JuliaHub          | 1.9         |
| Rust     | `rust/`       | crates.io         | 1.85        |
| C++      | `cpp/`        | header-only       | C++17       |
| R        | `r/`          | CRAN / r-universe | 4.1         |

---

## Universal rules (all languages)

- Every source file **must** begin with the SPDX header:
  ```
  # SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
  # SPDX-License-Identifier: MIT
  ```
  Adjust the comment style to the language (`//` for C++/Rust, `#` for Python/Julia/R).
- All commits require a **DCO sign-off** (`Signed-off-by:` line).
- Never introduce runtime dependencies beyond what is already declared per language.
- Algorithm parameters and their semantics must stay in sync across all ports. If a parameter changes in one language, update the others.
- Do not add speculative features. Implement only what is required.

---

## Module structure (mirror across all languages)

Every port exposes the same logical modules:

| Module        | Responsibility                                      |
|---------------|-----------------------------------------------------|
| `config`      | Algorithm hyper-parameters / configuration object   |
| `result`      | Result container + `TerminationReason` enum         |
| `exceptions`  | Custom exception hierarchy rooted at `GivpError`    |
| `grasp`       | GRASP construction phase (RCL, greedy)              |
| `vnd`         | VND local search — 5 neighbourhood moves            |
| `ils`         | ILS perturbation + re-search loop                   |
| `pr`          | Path relinking (forward / backward / bidirectional) |
| `impl` / core | Main orchestrator                                   |
| `cache`       | LRU evaluation cache                                |
| `elite`       | Elite pool for path relinking                       |
| `convergence` | Convergence monitor                                 |
| `helpers`     | RNG, bounds utilities, shared math                  |
| `api`         | Public-facing wrapper (direction-agnostic)          |
| `benchmarks`  | Standard test functions (sphere, rosenbrock, …)     |

---

## Python

- Python **≥ 3.10**; use `from __future__ import annotations` in every module.
- **mypy strict** mode. All public symbols must be fully annotated.
- **ruff** is the linter and formatter (`line-length = 88`, `target-version = "py310"`).
  Enabled rule sets: `E, F, I, UP, B, SIM, S, RUF, PERF, PT, RET`.
- Tests live in `python/tests/` and use **pytest** with `--cov-fail-under=95`.
  Use **Hypothesis** for property-based tests (`max_examples=15`).
- Conftest fixtures: `sphere`, `knapsack`, `qap`.
- Source layout: `python/src/givp/` (hatchling, `src` layout).

### Public API shape (Python)
```python
givp(func, bounds, *, num_vars=None, minimize=None, direction=None,
     config=None, initial_guess=None, iteration_callback=None,
     seed=None, verbose=False) -> OptimizeResult
```

---

## Julia

- Target Julia **≥ 1.9**. Package name: `GIVPOptimizer`.
- Follow standard Julia style: `CamelCase` for types, `snake_case` for functions, 4-space indentation.
- Format with **JuliaFormatter** (`style = "default"`).
- Tests are in `julia/test/runtests.jl`; minimum coverage gate: **85 %**.
- Fuzzing driver: `julia/fuzz/fuzz_givp.jl`.
- CLI entry point: `julia/cli.jl`.
- Use `@kwdef` structs for configuration to allow keyword construction.
- Exceptions must be subtypes of `GivpError <: Exception`.

---

## Rust

- Minimum Rust **1.85** (stable channel).
- Use `cargo fmt` and `cargo clippy -- -D warnings` before committing.
- No `unsafe` blocks unless unavoidable; if added, document the invariant.
- Public API re-exported from `lib.rs`; internal modules are `pub(crate)`.
- Errors use a custom `GivpError` enum; `Result<T>` is `std::result::Result<T, GivpError>`.
- Tests live in `rust/tests/test_givp.rs`.
- `n_workers` parallelism must use **rayon** (do not introduce other thread runtimes).

---

## C++

- Standard: **C++17**. Build system: **CMake ≥ 3.21**.
- Header-only library under `cpp/include/givp/`. No `.cpp` files in the public API.
- Use `#pragma once` guards. All public symbols are in the `givp::` namespace.
- Document with Doxygen-style `///` comments.
- Tests use **Catch2**; benchmarks use **nanobench** (both fetched via `FetchContent`).
- No exceptions thrown across ABI boundaries in the header-only layer.

---

## R

- Minimum R **4.1** (uses native pipe `|>` and `\(x)` lambda syntax).
- Package name: **givp** (matches Python PyPI name).
- Directory layout:
  ```
  r/
    DESCRIPTION
    NAMESPACE
    R/
      api.R          # givp() and GIVPOptimizer R6 class
      config.R       # givp_config() constructor
      result.R       # result helpers, termination reason constants
      grasp.R
      vnd.R
      ils.R
      pr.R
      impl.R
      cache.R
      elite.R
      convergence.R
      helpers.R
      benchmarks.R
      exceptions.R
    tests/
      testthat/
        test-api.R
        test-config.R
        test-core.R
        test-benchmarks.R
    man/              # Roxygen2-generated .Rd files
    vignettes/
    inst/
  ```
- Use **roxygen2** for documentation; every exported symbol needs `@export` and `@param`/`@return` tags.
- Use **R6** classes for `GIVPOptimizer` and `GIVPConfig`; plain functions for the functional `givp()` API.
- Tests use **testthat** (≥ 3rd edition). Minimum coverage: **90 %** (enforced with covr).
- Code style follows the **tidyverse style guide**: `snake_case` everywhere, spaces around operators, 2-space indentation.
- Lint with **lintr** (`linters = linters_with_defaults()`).
- Errors: define a condition hierarchy with `rlang::abort()` using `.subclass = "givp_error"`.
- The functional entry point must match the Python signature as closely as R allows:
  ```r
  givp(func, bounds, num_vars = NULL, minimize = NULL, direction = NULL,
       config = NULL, initial_guess = NULL, iteration_callback = NULL,
       seed = NULL, verbose = FALSE)
  ```
- CI workflow: `ci-r.yml` using `r-lib/actions/setup-r` with R 4.1, 4.3, and release.

---

## Testing guidelines (all languages)

- Write tests for **happy-path**, **edge-cases** (empty bounds, single variable, large dim), and **error conditions**.
- Each new algorithm feature must be covered by at least one unit test.
- Do not mock the objective function unless testing error propagation.
- Fuzz drivers exist for Python (`python/fuzz/`) and Julia (`julia/fuzz/`); add `r/fuzz/fuzz_givp.R` when the R port is ready.

---

## Documentation

- Docs use **MkDocs Material** (`mkdocs.yml` at repo root).
- Add a `docs/r.md` page for the R implementation (mirror of `docs/julia.md`).
- All algorithm parameters must be described identically across language doc pages.
- Code examples in docs must be runnable (tested in CI where possible).

---

## Commit and PR conventions

- Commit message format: `<type>(<scope>): <subject>` (Conventional Commits).
  Scopes: `python`, `julia`, `rust`, `cpp`, `r`, `docs`, `ci`, `deps`.
- Each commit must include `Signed-off-by: Name <email>` (DCO).
- PRs that change algorithm behaviour must update **all** language ports, or clearly document which are deferred with a tracking issue.
