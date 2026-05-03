// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! Experimental protocol helpers for multi-seed reproducible runs.

use crate::{givp, GivpConfig, OptimizeResult, Result};
use std::collections::BTreeMap;

/// Statistics summary from a multi-seed sweep.
#[derive(Debug, Clone)]
pub struct SweepSummary {
    /// Mean objective function value across all seeds.
    pub fun_mean: f64,
    /// Standard deviation of objective function values.
    pub fun_std: f64,
    /// Minimum objective function value.
    pub fun_min: f64,
    /// Maximum objective function value.
    pub fun_max: f64,
    /// Mean number of function evaluations.
    pub nfev_mean: f64,
    /// Mean number of iterations.
    pub nit_mean: f64,
}

/// Run an optimizer multiple times with different random seeds to estimate
/// reproducibility and convergence behavior.
///
/// # Arguments
///
/// * `func` — Objective function `&[f64] -> f64`.
/// * `bounds` — Variable bounds as `&[(lower, upper)]`.
/// * `base_config` — Base algorithm configuration (seed will be overridden for each run).
/// * `n_runs` — Number of seeds to execute (typically 30).
///
/// # Returns
///
/// A tuple of (results per seed, summary statistics).
///
/// # Example
///
/// ```no_run
/// use givp::{seed_sweep, GivpConfig};
///
/// let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
/// let bounds = vec![(-5.12, 5.12); 5];
/// let config = GivpConfig {
///     max_iterations: 50,
///     ..Default::default()
/// };
///
/// let (results, summary) = seed_sweep(&sphere, &bounds, config, 30)
///     .expect("sweep failed");
///
/// println!("Mean fitness: {:.6e}", summary.fun_mean);
/// println!("Std fitness: {:.6e}", summary.fun_std);
/// ```
pub fn seed_sweep<F>(
    func: F,
    bounds: &[(f64, f64)],
    base_config: GivpConfig,
    n_runs: usize,
) -> Result<(BTreeMap<u64, OptimizeResult>, SweepSummary)>
where
    F: Fn(&[f64]) -> f64 + Copy + Send + Sync,
{
    let mut results: BTreeMap<u64, OptimizeResult> = BTreeMap::new();
    let mut values: Vec<f64> = Vec::with_capacity(n_runs);
    let mut total_nfev = 0usize;
    let mut total_nit = 0usize;

    for seed in 0..n_runs as u64 {
        let mut cfg = base_config.clone();
        cfg.seed = Some(seed);

        let result = givp(func, bounds, cfg)?;
        values.push(result.fun);
        total_nfev += result.nfev;
        total_nit += result.nit;
        results.insert(seed, result);
    }

    let n = n_runs as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let summary = SweepSummary {
        fun_mean: mean,
        fun_std: std,
        fun_min: min,
        fun_max: max,
        nfev_mean: total_nfev as f64 / n,
        nit_mean: total_nit as f64 / n,
    };

    Ok((results, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_sweep_basic() {
        let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let bounds = vec![(-5.12, 5.12); 3];
        let cfg = GivpConfig {
            max_iterations: 10,
            ..Default::default()
        };

        let (results, summary) = seed_sweep(sphere, &bounds, cfg, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert!(summary.fun_mean.is_finite());
        assert!(summary.fun_std >= 0.0);
        assert!(summary.fun_min <= summary.fun_mean);
        assert!(summary.fun_max >= summary.fun_mean);
        assert!(summary.nfev_mean > 0.0);
    }

    #[test]
    fn test_seed_sweep_deterministic() {
        let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let bounds = vec![(-5.12, 5.12); 2];
        let cfg = GivpConfig {
            max_iterations: 5,
            ..Default::default()
        };

        let (_, summary1) = seed_sweep(sphere, &bounds, cfg.clone(), 3).unwrap();
        let (_, summary2) = seed_sweep(sphere, &bounds, cfg, 3).unwrap();

        // Same number of runs should produce deterministic seed assignment
        assert_eq!(summary1.fun_mean, summary2.fun_mean);
    }
}
