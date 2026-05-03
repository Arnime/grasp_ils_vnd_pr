// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! Reproducible multi-seed experiment helper for Rust.

use givp::{givp, GivpConfig};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn main() {
    let dims = 10usize;
    let n_runs = 30usize;
    let bounds = vec![(-5.12, 5.12); dims];

    let mut values: Vec<f64> = Vec::with_capacity(n_runs);
    let mut total_nfev = 0usize;
    let mut total_nit = 0usize;

    for seed in 0..n_runs as u64 {
        let cfg = GivpConfig {
            max_iterations: 50,
            seed: Some(seed),
            integer_split: Some(dims),
            ..Default::default()
        };

        let result = givp(sphere, &bounds, cfg).expect("seed_sweep run failed");
        println!(
            "seed={} best={:.6e} nfev={} nit={} success={}",
            seed, result.fun, result.nfev, result.nit, result.success
        );

        values.push(result.fun);
        total_nfev += result.nfev;
        total_nit += result.nit;
    }

    let mean = values.iter().sum::<f64>() / n_runs as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_runs as f64;
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("summary.fun.mean={:.6e}", mean);
    println!("summary.fun.std={:.6e}", std);
    println!("summary.fun.min={:.6e}", min);
    println!("summary.fun.max={:.6e}", max);
    println!("summary.nfev.mean={:.1}", total_nfev as f64 / n_runs as f64);
    println!("summary.nit.mean={:.1}", total_nit as f64 / n_runs as f64);
}
