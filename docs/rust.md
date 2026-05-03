<!-- SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior -->
<!-- SPDX-License-Identifier: MIT -->

# Rust

`givp` is also available as a native Rust crate, providing the same
GRASP-ILS-VND with Path Relinking algorithm with an idiomatic, zero-overhead
Rust API.

## Installation

From a local clone of the repository:

```bash
cd rust
cargo build
```

Requires Rust ≥ 1.85 (stable channel).

Add to your own project's `Cargo.toml`:

```toml
[dependencies]
givp = { path = "../rust" }
```

Optional JSON serialisation (via `serde`):

```toml
[dependencies]
givp = { path = "../rust", features = ["serde"] }
```

## Quick start

```rust
use givp::{givp, GivpConfig};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn main() {
    let bounds = vec![(-5.12, 5.12); 10];
    let cfg = GivpConfig {
        seed: Some(42),
        integer_split: Some(10), // all continuous
        ..Default::default()
    };
    let result = givp(sphere, &bounds, cfg).unwrap();
    println!("{:?}", result.x);   // best vector found
    println!("{}", result.fun);   // best objective value
    println!("{}", result.nfev);  // number of evaluations
}
```

## Maximize

```rust
use givp::{givp, Direction, GivpConfig};

let cfg = GivpConfig {
    direction: Direction::Maximize,
    seed: Some(42),
    integer_split: Some(5),
    ..Default::default()
};
let result = givp(my_score, &bounds, cfg).unwrap();
```

## Configuration

All hyper-parameters are exposed via the `GivpConfig` struct:

```rust
use givp::GivpConfig;

let cfg = GivpConfig {
    max_iterations: 50,
    vnd_iterations: 100,
    ils_iterations: 10,
    elite_size: 7,
    adaptive_alpha: true,
    time_limit: 30.0,
    seed: Some(42),
    integer_split: Some(10), // number of continuous variables
    ..Default::default()
};
```

## Warm start

```rust
let cfg = GivpConfig {
    initial_guess: Some(vec![1.0; 5]),
    integer_split: Some(5),
    ..Default::default()
};
let result = givp(rosenbrock, &bounds, cfg).unwrap();
```

## Mixed continuous/integer problems

```rust
let n_cont = 12usize;
let n_int  = 8usize;
let bounds: Vec<(f64, f64)> = (0..n_cont)
    .map(|_| (-5.0_f64, 5.0_f64))
    .chain((0..n_int).map(|_| (0.0_f64, 4.0_f64)))
    .collect();

let cfg = GivpConfig {
    integer_split: Some(n_cont),
    ..Default::default()
};
let result = givp(my_objective, &bounds, cfg).unwrap();
```

## Parallel candidate evaluation (rayon)

```rust
let cfg = GivpConfig {
    n_workers: 4,          // rayon thread pool size
    use_cache: false,      // cache is not thread-safe; must be disabled
    integer_split: Some(10),
    ..Default::default()
};
let result = givp(sphere, &bounds, cfg).unwrap();
```

## Result struct

`givp` returns `Result<OptimizeResult, GivpError>` with the following fields:

| Field         | Type                       | Meaning                                           |
|---------------|----------------------------|---------------------------------------------------|
| `x`           | `Vec<f64>`                 | Best solution vector.                             |
| `fun`         | `f64`                      | Objective value at `x` (user's original sign).    |
| `nit`         | `usize`                    | GRASP outer iterations executed.                  |
| `nfev`        | `usize`                    | Number of objective evaluations.                  |
| `success`     | `bool`                     | `true` when at least one feasible solution found. |
| `message`     | `String`                   | Human-readable termination reason.                |
| `direction`   | `Direction`                | `Minimize` or `Maximize`.                         |
| `termination` | `TerminationReason`        | Typed termination reason enum.                    |
| `meta`        | `HashMap<String, String>`  | Algorithm-specific extras.                        |

## Error handling

```rust
use givp::GivpError;

match givp(sphere, &bounds, cfg) {
    Ok(r) => println!("best = {}", r.fun),
    Err(GivpError::InvalidBounds(msg))      => eprintln!("bounds: {msg}"),
    Err(GivpError::InvalidConfig(msg))      => eprintln!("config: {msg}"),
    Err(GivpError::InvalidInitialGuess(msg))=> eprintln!("guess: {msg}"),
    Err(e) => eprintln!("error: {e}"),
}
```

## Running tests

```bash
cd rust
cargo test
```

With property-based tests (proptest, 64 cases per property):

```bash
cargo test          # proptest runs automatically as part of the suite
```

## Running benchmarks

Benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) and
cover sphere 5D, Rosenbrock 5D, and Rastrigin 10D:

```bash
cd rust
cargo bench
```

HTML reports are written to `rust/target/criterion/`.

## Literature comparison experiment

A reproducible multi-run experiment is provided as a Cargo example:

```bash
# Default: 30 seeds × 10-D × 6 functions → rust/benchmarks/literature_comparison.json
cargo run --example run_literature_comparison

# Custom parameters
cargo run --example run_literature_comparison -- \
    --n-runs 30 --dims 10 --output results.json --verbose
```

Output JSON is compatible with the Python `generate_report.py` tool.

## CLI

The Rust port also provides a small CLI binary:

```bash
cd rust
cargo run --bin givp -- --function sphere --dims 10 --seed 42
```

Supported options include `--function`, `--dims`, `--seed`, and `--direction`.

## Coverage

The CI enforces a minimum of **90 %** line coverage on `rust/src/`.

To check locally:

```bash
cd rust
cargo install cargo-llvm-cov  # once
cargo llvm-cov --all-features --lcov --output-path lcov.info
# Parse coverage percentage from lcov.info
grep '^LH:\|^LF:' lcov.info | awk -F: '
  /^LH:/ { hit  += $2 }
  /^LF:/ { total+= $2 }
  END    { printf "Coverage: %.1f%% (%d / %d lines)\n",
           hit/total*100, hit, total }
'
```

## API parity with Python

| Feature                     | Python | Julia | Rust |
|-----------------------------|--------|-------|------|
| GRASP construction          | ✓      | ✓     | ✓    |
| VND local search            | ✓      | ✓     | ✓    |
| ILS perturbation            | ✓      | ✓     | ✓    |
| Path Relinking              | ✓      | ✓     | ✓    |
| Elite pool                  | ✓      | ✓     | ✓    |
| Convergence monitor         | ✓      | ✓     | ✓    |
| LRU evaluation cache        | ✓      | ✓     | ✓    |
| Adaptive α                  | ✓      | ✓     | ✓    |
| Time budget                 | ✓      | ✓     | ✓    |
| Mixed integer/continuous    | ✓      | ✓     | ✓    |
| Warm start                  | ✓      | ✓     | ✓    |
| Reproducible (`seed=`)      | ✓      | ✓     | ✓    |
| Parallel candidates (rayon) | ✓      | ✓     | ✓    |
| Literature comparison       | ✓      | ✓     | ✓    |
| JSON serialisation          | ✓      | ✓     | ✓ (feature) |
| Property-based tests        | ✓      | ✓     | ✓    |
| CLI entry point             | ✓      | ✓     | ✓    |
| Fuzzing driver              | ✓      | ✓     | —    |
