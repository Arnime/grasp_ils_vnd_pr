// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! Reproducible multi-run literature comparison experiment.
//!
//! Runs GIVP-full on six standard benchmark functions over N independent seeds
//! and writes results to a JSON file compatible with the Python/Julia
//! `generate_report` tool.
//!
//! # Usage
//!
//! ```bash
//! # Default: 30 seeds × 10-D × 6 functions
//! cargo run --example run_literature_comparison
//!
//! # Custom parameters
//! cargo run --example run_literature_comparison -- \
//!     --n-runs 30 --dims 10 --output results.json --verbose
//! ```
//!
//! # References
//!
//! - De Jong, K.A. (1975). Sphere / De Jong F1.
//! - Rosenbrock, H.H. (1960). The Computer Journal, 3(3), 175–184.
//! - Rastrigin, L.A. (1974). Systems of Extremal Control. Nauka, Moscow.
//! - Ackley, D.H. (1987). A Connectionist Machine for Genetic Hillclimbing.
//! - Griewank, A.O. (1981). J. Optim. Theory Appl., 34(1), 11–39.
//! - Schwefel, H.P. (1981). Numerical Optimization of Computer Models.

use givp::{givp, GivpConfig};
use std::env;
use std::fs;
use std::time::Instant;

// ── benchmark functions ──────────────────────────────────────────────────────

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / n;
    let sum_cos: f64 = x
        .iter()
        .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>()
        / n;
    -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp() + 20.0 + std::f64::consts::E
}

fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>() / 4000.0;
    let prod_cos: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
        .product();
    1.0 + sum_sq - prod_cos
}

fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|&xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
}

// ── function registry ────────────────────────────────────────────────────────

struct BenchFunc {
    name: &'static str,
    func: fn(&[f64]) -> f64,
    bounds_fn: fn(usize) -> Vec<(f64, f64)>,
    optimum: f64,
}

fn get_functions() -> Vec<BenchFunc> {
    vec![
        BenchFunc {
            name: "sphere",
            func: sphere,
            bounds_fn: |d| vec![(-5.12, 5.12); d],
            optimum: 0.0,
        },
        BenchFunc {
            name: "rosenbrock",
            func: rosenbrock,
            bounds_fn: |d| vec![(-5.0, 10.0); d],
            optimum: 0.0,
        },
        BenchFunc {
            name: "rastrigin",
            func: rastrigin,
            bounds_fn: |d| vec![(-5.12, 5.12); d],
            optimum: 0.0,
        },
        BenchFunc {
            name: "ackley",
            func: ackley,
            bounds_fn: |d| vec![(-32.768, 32.768); d],
            optimum: 0.0,
        },
        BenchFunc {
            name: "griewank",
            func: griewank,
            bounds_fn: |d| vec![(-600.0, 600.0); d],
            optimum: 0.0,
        },
        BenchFunc {
            name: "schwefel",
            func: schwefel,
            bounds_fn: |d| vec![(-500.0, 500.0); d],
            optimum: 0.0,
        },
    ]
}

// ── trial runner ─────────────────────────────────────────────────────────────

struct TrialResult {
    algorithm: &'static str,
    function: &'static str,
    seed: u64,
    best: f64,
    nfev: usize,
    elapsed_s: f64,
}

fn run_trial(
    func: fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    dims: usize,
    seed: u64,
) -> (f64, usize, f64) {
    let start = Instant::now();
    let cfg = GivpConfig {
        max_iterations: 50,
        seed: Some(seed),
        integer_split: Some(dims),
        ..Default::default()
    };
    match givp(func, bounds, cfg) {
        Ok(r) => (r.fun, r.nfev, start.elapsed().as_secs_f64()),
        Err(_) => (f64::INFINITY, 0, start.elapsed().as_secs_f64()),
    }
}

// ── JSON serialisation (no external deps) ───────────────────────────────────

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn results_to_json(entries: &[TrialResult]) -> String {
    let mut out = String::from("[\n");
    for (i, e) in entries.iter().enumerate() {
        let comma = if i + 1 < entries.len() { "," } else { "" };
        out.push_str(&format!(
            "  {{\"algorithm\":\"{}\",\"function\":\"{}\",\
\"seed\":{},\"best\":{:.10e},\"nfev\":{},\"elapsed_s\":{:.4}}}{}\n",
            json_escape(e.algorithm),
            json_escape(e.function),
            e.seed,
            e.best,
            e.nfev,
            e.elapsed_s,
            comma,
        ));
    }
    out.push(']');
    out
}

// ── CLI argument parsing ─────────────────────────────────────────────────────

struct Args {
    n_runs: usize,
    dims: usize,
    output: String,
    verbose: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut n_runs = 30usize;
    let mut dims = 10usize;
    let mut output = "rust/benchmarks/literature_comparison.json".to_string();
    let mut verbose = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--n-runs" if i + 1 < argv.len() => {
                n_runs = argv[i + 1].parse().unwrap_or(30);
                i += 2;
            }
            "--dims" if i + 1 < argv.len() => {
                dims = argv[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "--output" if i + 1 < argv.len() => {
                output = argv[i + 1].clone();
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
    Args {
        n_runs,
        dims,
        output,
        verbose,
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let functions = get_functions();

    println!("GIVP Literature Comparison (Rust)");
    println!(
        "  dims={}  runs/function={}  functions={}",
        args.dims,
        args.n_runs,
        functions.len()
    );
    println!("  output → {}", args.output);
    println!();

    let mut entries: Vec<TrialResult> = Vec::new();
    let total = functions.len() * args.n_runs;
    let mut done = 0usize;

    for bf in &functions {
        let bounds = (bf.bounds_fn)(args.dims);
        let mut bests: Vec<f64> = Vec::with_capacity(args.n_runs);

        for seed in 0..args.n_runs as u64 {
            let (best, nfev, elapsed) = run_trial(bf.func, &bounds, args.dims, seed);
            if args.verbose {
                println!(
                    "  {:>12}  seed={:>3}  best={:.6e}  nfev={}  {:.2}s",
                    bf.name, seed, best, nfev, elapsed
                );
            }
            bests.push(best);
            entries.push(TrialResult {
                algorithm: "GIVP-full",
                function: bf.name,
                seed,
                best,
                nfev,
                elapsed_s: elapsed,
            });
            done += 1;
            if !args.verbose && done % 30 == 0 {
                println!("  [{}/{}] last: {} seed={}", done, total, bf.name, seed);
            }
        }

        let mean = bests.iter().sum::<f64>() / args.n_runs as f64;
        let variance = bests.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / args.n_runs as f64;
        let std = variance.sqrt();
        let min = bests.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "  {:>12}  mean={:.4e}  std={:.4e}  best={:.4e}  gap={:.4e}",
            bf.name,
            mean,
            std,
            min,
            (min - bf.optimum).abs()
        );
    }

    // ensure output directory exists
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let json = results_to_json(&entries);
    match fs::write(&args.output, &json) {
        Ok(_) => println!("\nResults written to {}", args.output),
        Err(e) => {
            eprintln!("Failed to write {}: {}", args.output, e);
            std::process::exit(1);
        }
    }
}
