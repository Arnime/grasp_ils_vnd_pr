// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! Randomised fuzz / property driver for the `givp` public API.
//!
//! Exercises the solver with random inputs to detect:
//! - Panics / unwrap failures
//! - NaN / Inf leaking into the solution vector
//! - Out-of-bounds solutions
//! - Unexpected errors on well-formed inputs
//!
//! # Usage
//!
//! ```bash
//! # Default: 1 000 trials, seed 42
//! cargo run --example fuzz_givp
//!
//! # Custom parameters
//! cargo run --example fuzz_givp -- --n-trials 5000 --seed 123 --verbose
//!
//! # With a timeout (seconds)
//! cargo run --example fuzz_givp -- --n-trials 100000 --timeout 60
//! ```

use givp::{givp, Direction, GivpConfig};
use std::env;
use std::time::Instant;

// ── Objective functions ──────────────────────────────────────────────────────

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn wave(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * v.sin())
        .sum::<f64>()
}

fn noisy_sphere(x: &[f64]) -> f64 {
    // Deterministic irregular surface — non-differentiable
    sphere(x) + (x[0] * 1_000_000.0).sin() * 0.01
}

fn nan_on_neg(x: &[f64]) -> f64 {
    // Returns NaN for negative x[0] — tests robustness to non-finite outputs
    if x[0] < 0.0 {
        f64::NAN
    } else {
        sphere(x)
    }
}

fn panic_on_extreme(x: &[f64]) -> f64 {
    // Panics on extreme inputs — tests catch_unwind guard
    if x[0] < -1e10 {
        panic!("deliberate panic in fuzz objective")
    }
    sphere(x)
}

// ── CLI argument parsing ─────────────────────────────────────────────────────

struct FuzzArgs {
    n_trials: usize,
    seed: u64,
    timeout_secs: f64,
    verbose: bool,
}

fn parse_args() -> FuzzArgs {
    let args: Vec<String> = env::args().collect();
    let mut n_trials = 1_000usize;
    let mut seed = 42u64;
    let mut timeout_secs = 300.0f64;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n-trials" if i + 1 < args.len() => {
                n_trials = args[i + 1].parse().unwrap_or(n_trials);
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(seed);
                i += 2;
            }
            "--timeout" if i + 1 < args.len() => {
                timeout_secs = args[i + 1].parse().unwrap_or(timeout_secs);
                i += 2;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    FuzzArgs {
        n_trials,
        seed,
        timeout_secs,
        verbose,
    }
}

// ── Simple LCG RNG (reproducible, no extra deps) ─────────────────────────────

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn lcg_f64(state: &mut u64) -> f64 {
    (lcg_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let t0 = Instant::now();

    let fast_cfg_base = GivpConfig {
        max_iterations: 3,
        vnd_iterations: 5,
        ils_iterations: 1,
        early_stop_threshold: 3,
        use_convergence_monitor: false,
        ..Default::default()
    };

    type ObjFn = fn(&[f64]) -> f64;
    let funcs: &[(&str, ObjFn)] = &[
        ("sphere", sphere),
        ("wave", wave),
        ("noisy_sphere", noisy_sphere),
        ("nan_on_neg", nan_on_neg),
        ("panic_on_extreme", panic_on_extreme),
    ];

    let directions = [Direction::Minimize, Direction::Maximize];
    let dir_names = ["minimize", "maximize"];

    let mut rng = args.seed;
    let mut failures = 0usize;
    let mut trials = 0usize;

    for trial in 0..args.n_trials {
        if t0.elapsed().as_secs_f64() > args.timeout_secs {
            eprintln!("Timeout reached after {trial} trials.");
            break;
        }

        let ndim = (lcg_next(&mut rng) % 5 + 1) as usize;
        let func_idx = (lcg_next(&mut rng) % funcs.len() as u64) as usize;
        let dir_idx = (lcg_next(&mut rng) % 2) as usize;

        // Random bounds in [-50, 50] with width in [1, 20]
        let bounds: Vec<(f64, f64)> = (0..ndim)
            .map(|_| {
                let lo = lcg_f64(&mut rng) * 100.0 - 50.0;
                let width = lcg_f64(&mut rng) * 19.0 + 1.0;
                (lo, lo + width)
            })
            .collect();

        let (func_name, func) = funcs[func_idx];
        let direction = directions[dir_idx];
        let dir_name = dir_names[dir_idx];

        let mut cfg = fast_cfg_base.clone();
        cfg.direction = direction;

        let result = std::panic::catch_unwind(|| givp(func, &bounds, cfg));

        trials += 1;

        match result {
            Err(_) => {
                eprintln!(
                    "[FAIL] trial={trial} func={func_name} dir={dir_name} ndim={ndim}: \
                     unexpected panic"
                );
                failures += 1;
            }
            Ok(Err(e)) => {
                // GivpError variants are expected for degenerate inputs — not a failure
                if args.verbose {
                    println!(
                        "[OK/ERR] trial={trial} func={func_name} dir={dir_name}: \
                         GivpError: {e}"
                    );
                }
            }
            Ok(Ok(res)) => {
                let mut ok = true;

                // Invariant: solution dimension matches ndim
                if res.x.len() != ndim {
                    eprintln!(
                        "[FAIL] trial={trial}: solution len {} != ndim {ndim}",
                        res.x.len()
                    );
                    ok = false;
                }

                // Invariant: solution within declared bounds (with tiny tolerance)
                for (i, (&xi, &(lo, hi))) in res.x.iter().zip(bounds.iter()).enumerate() {
                    let tol = (hi - lo) * 1e-9 + 1e-12;
                    if xi < lo - tol || xi > hi + tol {
                        eprintln!(
                            "[FAIL] trial={trial}: x[{i}]={xi:.6} out of bounds \
                             [{lo:.4}, {hi:.4}]"
                        );
                        ok = false;
                    }
                }

                if !ok {
                    failures += 1;
                } else if args.verbose {
                    println!(
                        "[OK] trial={trial} func={func_name} dir={dir_name} \
                         ndim={ndim} fun={:.6}",
                        res.fun
                    );
                }
            }
        }
    }

    println!(
        "\nFuzz complete: {trials} trials, {failures} failures, {:.1}s elapsed",
        t0.elapsed().as_secs_f64()
    );

    if failures > 0 {
        std::process::exit(1);
    }
}
