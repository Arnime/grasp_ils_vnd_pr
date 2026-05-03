// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::cache::EvaluationCache;
use crate::config::{Direction, GivpConfig};
use crate::convergence::ConvergenceMonitor;
use crate::elite::ElitePool;
use crate::error::{GivpError, Result};
use crate::grasp::{construct_grasp, evaluate_with_cache, get_current_alpha};
use crate::helpers::{child_rng, expired, get_half, new_rng, normalize_integer_tail};
use crate::ils::ils_search;
use crate::pr::bidirectional_path_relinking;
use crate::result::{OptimizeResult, TerminationReason};
use crate::vnd::local_search_vnd;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Validate bounds and initial guess.
fn validate_bounds(
    bounds: &[(f64, f64)],
    initial_guess: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if bounds.is_empty() {
        return Err(GivpError::InvalidBounds("bounds cannot be empty".into()));
    }
    let mut lower = Vec::with_capacity(bounds.len());
    let mut upper = Vec::with_capacity(bounds.len());
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        validate_single_bound(i, lo, hi)?;
        lower.push(lo);
        upper.push(hi);
    }
    if let Some(ig) = initial_guess {
        validate_initial_guess(bounds, ig)?;
    }
    Ok((lower, upper))
}

fn validate_single_bound(index: usize, lo: f64, hi: f64) -> Result<()> {
    if lo >= hi {
        return Err(GivpError::InvalidBounds(format!(
            "lower >= upper at index {index}: {lo} >= {hi}"
        )));
    }
    if !lo.is_finite() || !hi.is_finite() {
        return Err(GivpError::InvalidBounds(format!(
            "non-finite bound at index {index}"
        )));
    }
    Ok(())
}

fn validate_initial_guess(bounds: &[(f64, f64)], initial_guess: &[f64]) -> Result<()> {
    if initial_guess.len() != bounds.len() {
        return Err(GivpError::InvalidInitialGuess(format!(
            "expected {} values, got {}",
            bounds.len(),
            initial_guess.len()
        )));
    }

    for (i, (&v, &(lo, hi))) in initial_guess.iter().zip(bounds.iter()).enumerate() {
        if v < lo || v > hi {
            return Err(GivpError::InvalidInitialGuess(format!(
                "value {v} out of bounds [{lo}, {hi}] at index {i}"
            )));
        }
    }

    Ok(())
}

fn build_cache(config: &GivpConfig) -> Option<EvaluationCache> {
    if config.use_cache {
        Some(EvaluationCache::new(config.cache_size))
    } else {
        None
    }
}

fn build_convergence_monitor(config: &GivpConfig) -> Option<ConvergenceMonitor> {
    if config.use_convergence_monitor {
        Some(ConvergenceMonitor::new(20, 50))
    } else {
        None
    }
}

fn build_deadline(config: &GivpConfig) -> Option<Instant> {
    if config.time_limit > 0.0 {
        Some(Instant::now() + Duration::from_secs_f64(config.time_limit))
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize_best_solution<F>(
    config: &GivpConfig,
    num_vars: usize,
    lower: &[f64],
    upper: &[f64],
    func: &F,
    half: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut rand_chacha::ChaCha8Rng,
    deadline: Option<Instant>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    if let Some(ref ig) = config.initial_guess {
        let mut best_solution = ig.clone();
        normalize_integer_tail(&mut best_solution, half);
        let best_cost = evaluate_with_cache(&best_solution, func, cache, half);
        return (best_solution, best_cost);
    }

    let mut child = child_rng(rng);
    construct_grasp(
        num_vars,
        lower,
        upper,
        func,
        None,
        config.alpha,
        half,
        config.num_candidates_per_step,
        cache,
        &mut child,
        deadline,
        config.n_workers,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_and_improve_candidate<F>(
    iteration: usize,
    num_vars: usize,
    lower: &[f64],
    upper: &[f64],
    func: &F,
    config: &GivpConfig,
    alpha: f64,
    half: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut rand_chacha::ChaCha8Rng,
    deadline: Option<Instant>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let mut child = child_rng(rng);
    let ig = if iteration == 0 {
        config.initial_guess.as_deref()
    } else {
        None
    };
    let (mut candidate, _grasp_cost) = construct_grasp(
        num_vars,
        lower,
        upper,
        func,
        ig,
        alpha,
        half,
        config.num_candidates_per_step,
        cache,
        &mut child,
        deadline,
        config.n_workers,
    );

    let grasp_eval = evaluate_with_cache(&candidate, func, cache, half);
    let vnd_cost = local_search_vnd(
        func,
        &mut candidate,
        grasp_eval,
        half,
        lower,
        upper,
        config.vnd_iterations,
        cache,
        &mut child,
        deadline,
    );
    let ils_cost = ils_search(
        func,
        &mut candidate,
        vnd_cost,
        half,
        lower,
        upper,
        config.ils_iterations,
        config.vnd_iterations,
        config.perturbation_strength,
        cache,
        &mut child,
        deadline,
    );

    (candidate, ils_cost)
}

fn update_best(
    best_solution: &mut Vec<f64>,
    best_cost: &mut f64,
    candidate: &[f64],
    cost: f64,
) -> bool {
    if cost < *best_cost {
        *best_cost = cost;
        *best_solution = candidate.to_vec();
        return true;
    }
    false
}

fn update_convergence_state(
    conv_monitor: &mut Option<ConvergenceMonitor>,
    elite_pool: &mut ElitePool,
    cache: &mut Option<EvaluationCache>,
    best_cost: f64,
    stagnation: &mut usize,
) -> Option<usize> {
    let cm = conv_monitor.as_mut()?;

    let signal = cm.update(best_cost, Some(elite_pool));
    if signal.should_restart {
        elite_pool.keep_top(2);
        cm.reset_no_improve();
        *stagnation = 0;
        if let Some(c) = cache.as_mut() {
            c.clear();
        }
        return Some(0);
    }

    Some(signal.no_improve_count)
}

fn should_path_relink(config: &GivpConfig, iteration: usize, elite_pool: &ElitePool) -> bool {
    config.use_elite_pool
        && iteration > 0
        && iteration % config.path_relink_frequency == 0
        && elite_pool.len() >= 2
}

#[allow(clippy::too_many_arguments)]
fn maybe_restart_from_stagnation<F>(
    stagnation: &mut usize,
    config: &GivpConfig,
    num_vars: usize,
    lower: &[f64],
    upper: &[f64],
    func: &F,
    alpha: f64,
    half: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut rand_chacha::ChaCha8Rng,
    deadline: Option<Instant>,
    best_solution: &mut Vec<f64>,
    best_cost: &mut f64,
) where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    if *stagnation <= config.max_iterations / 4 {
        return;
    }

    let mut child = child_rng(rng);
    let mut restart = crate::grasp::construct_grasp(
        num_vars,
        lower,
        upper,
        func,
        None,
        alpha,
        half,
        config.num_candidates_per_step,
        cache,
        &mut child,
        deadline,
        config.n_workers,
    );
    let restart_cost = local_search_vnd(
        func,
        &mut restart.0,
        restart.1,
        half,
        lower,
        upper,
        config.vnd_iterations,
        cache,
        &mut child,
        deadline,
    );
    let restart_cost = ils_search(
        func,
        &mut restart.0,
        restart_cost,
        half,
        lower,
        upper,
        config.ils_iterations,
        config.vnd_iterations,
        config.perturbation_strength,
        cache,
        &mut child,
        deadline,
    );

    if restart_cost < *best_cost {
        *best_cost = restart_cost;
        *best_solution = restart.0;
    }
    *stagnation = 0;
}

#[allow(clippy::too_many_arguments)]
fn execute_iteration<F>(
    iteration: usize,
    config: &GivpConfig,
    num_vars: usize,
    lower: &[f64],
    upper: &[f64],
    wrapped: &F,
    half: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut rand_chacha::ChaCha8Rng,
    elite_pool: &mut ElitePool,
    conv_monitor: &mut Option<ConvergenceMonitor>,
    best_solution: &mut Vec<f64>,
    best_cost: &mut f64,
    stagnation: &mut usize,
    deadline: Option<Instant>,
) -> Option<&'static str>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    let alpha = get_current_alpha(
        iteration,
        config.max_iterations,
        config.alpha_min,
        config.alpha_max,
        config.adaptive_alpha,
        config.alpha,
    );

    let (candidate, ils_cost) = build_and_improve_candidate(
        iteration, num_vars, lower, upper, wrapped, config, alpha, half, cache, rng, deadline,
    );

    if update_best(best_solution, best_cost, &candidate, ils_cost) {
        *stagnation = 0;
    } else {
        *stagnation += 1;
    }

    if config.use_elite_pool {
        elite_pool.add(candidate, ils_cost);
    }

    let no_improve_count =
        update_convergence_state(conv_monitor, elite_pool, cache, *best_cost, stagnation);

    if should_path_relink(config, iteration, elite_pool) {
        let mut child = child_rng(rng);
        do_path_relinking(
            wrapped,
            elite_pool,
            best_solution,
            best_cost,
            half,
            lower,
            upper,
            config.vnd_iterations,
            cache,
            &mut child,
            deadline,
        );
    }

    maybe_restart_from_stagnation(
        stagnation,
        config,
        num_vars,
        lower,
        upper,
        wrapped,
        alpha,
        half,
        cache,
        rng,
        deadline,
        best_solution,
        best_cost,
    );

    if no_improve_count.is_some_and(|count| count >= config.early_stop_threshold) {
        return Some("early stop due to stagnation");
    }

    None
}

/// Perform path relinking on elite pool pairs.
#[allow(clippy::too_many_arguments)]
fn do_path_relinking<F>(
    func: &F,
    elite_pool: &ElitePool,
    best_solution: &mut Vec<f64>,
    best_cost: &mut f64,
    half: usize,
    lower: &[f64],
    upper: &[f64],
    vnd_iterations: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut rand_chacha::ChaCha8Rng,
    deadline: Option<Instant>,
) where
    F: Fn(&[f64]) -> f64,
{
    let all = elite_pool.get_all();
    // Caller guarantees elite_pool.len() >= 2
    let max_pairs = 3.min(all.len());
    for i in 0..max_pairs {
        for j in (i + 1)..all.len().min(i + 4) {
            if expired(deadline) {
                return;
            }
            let (mut pr_sol, pr_cost) = bidirectional_path_relinking(
                func, &all[i].0, &all[j].0, half, cache, rng, deadline,
            );

            // VND refinement on PR result
            let refined_cost = local_search_vnd(
                func,
                &mut pr_sol,
                pr_cost,
                half,
                lower,
                upper,
                vnd_iterations / 2,
                cache,
                rng,
                deadline,
            );

            if refined_cost < *best_cost {
                *best_cost = refined_cost;
                *best_solution = pr_sol;
            }
        }
    }
}

/// Main optimizer entry point.
pub(crate) fn run<F>(func: F, bounds: &[(f64, f64)], config: GivpConfig) -> Result<OptimizeResult>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    config.validate()?;

    let (lower, upper) = validate_bounds(bounds, config.initial_guess.as_deref())?;
    let num_vars = bounds.len();
    let half = get_half(num_vars, config.integer_split.or(Some(num_vars)));
    let is_maximize = config.direction == Direction::Maximize;

    // Wrap objective: flip sign for maximization, count evaluations
    let nfev = AtomicUsize::new(0);
    let wrapped = |x: &[f64]| -> f64 {
        nfev.fetch_add(1, Ordering::Relaxed);
        let v = func(x);
        if is_maximize {
            -v
        } else {
            v
        }
    };

    let mut rng = new_rng(config.seed);
    let mut cache = build_cache(&config);

    let mut elite_pool = ElitePool::new(config.elite_size, 0.05, &lower, &upper);
    let mut conv_monitor = build_convergence_monitor(&config);

    let deadline = build_deadline(&config);
    let (mut best_solution, mut best_cost) = initialize_best_solution(
        &config, num_vars, &lower, &upper, &wrapped, half, &mut cache, &mut rng, deadline,
    );

    if config.use_elite_pool {
        elite_pool.add(best_solution.clone(), best_cost);
    }

    let mut stagnation: usize = 0;
    let mut iterations_executed: usize = 0;
    let mut message = String::new();

    // Main loop
    for iteration in 0..config.max_iterations {
        if expired(deadline) {
            message = "time limit reached".into();
            break;
        }
        iterations_executed = iteration + 1;

        if let Some(stop_message) = execute_iteration(
            iteration,
            &config,
            num_vars,
            &lower,
            &upper,
            &wrapped,
            half,
            &mut cache,
            &mut rng,
            &mut elite_pool,
            &mut conv_monitor,
            &mut best_solution,
            &mut best_cost,
            &mut stagnation,
            deadline,
        ) {
            message = stop_message.into();
            break;
        }
    }

    if message.is_empty() && iterations_executed == config.max_iterations {
        message = "max iterations reached".into();
    }

    // Build result
    let final_cost = if is_maximize { -best_cost } else { best_cost };

    let mut result = OptimizeResult::new(config.direction);
    result.x = best_solution;
    result.fun = final_cost;
    result.nit = iterations_executed;
    result.nfev = nfev.load(Ordering::Relaxed);
    result.success = final_cost.is_finite();
    result.message = message.clone();
    result.termination = TerminationReason::from_message(&result.message);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::{Duration, Instant};

    #[test]
    fn test_run_with_invalid_config_errors() {
        // Covers the error branch of config.validate()? in run() at line 116.
        let func = |x: &[f64]| x.iter().sum::<f64>();
        let cfg = GivpConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(run(func, &[(-1.0, 1.0)], cfg).is_err());
    }

    #[test]
    fn test_do_path_relinking_expired_deadline_returns() {
        // Expired deadline causes the inner `return` in do_path_relinking (line 83).
        let func = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let mut pool = ElitePool::new(5, 0.01, &[-5.0, -5.0], &[5.0, 5.0]);
        pool.add(vec![1.0, 0.0], 1.0);
        pool.add(vec![-4.0, 3.0], 25.0);
        let mut best_solution = vec![1.0, 0.0];
        let mut best_cost = 1.0;
        assert!((func(&best_solution) - 1.0).abs() < 1e-10); // invoke closure body
        let deadline = Some(Instant::now() - Duration::from_secs(1));
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        do_path_relinking(
            &func,
            &pool,
            &mut best_solution,
            &mut best_cost,
            2,
            &[-5.0, -5.0],
            &[5.0, 5.0],
            10,
            &mut None,
            &mut rng,
            deadline,
        );
        // Best solution unchanged since deadline expired immediately
        assert!((best_cost - 1.0).abs() < 1e-10);
    }
}
