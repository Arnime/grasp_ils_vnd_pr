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
        if lo >= hi {
            return Err(GivpError::InvalidBounds(format!(
                "lower >= upper at index {i}: {lo} >= {hi}"
            )));
        }
        if !lo.is_finite() || !hi.is_finite() {
            return Err(GivpError::InvalidBounds(format!(
                "non-finite bound at index {i}"
            )));
        }
        lower.push(lo);
        upper.push(hi);
    }
    if let Some(ig) = initial_guess {
        if ig.len() != bounds.len() {
            return Err(GivpError::InvalidInitialGuess(format!(
                "expected {} values, got {}",
                bounds.len(),
                ig.len()
            )));
        }
        for (i, (&v, &(lo, hi))) in ig.iter().zip(bounds.iter()).enumerate() {
            if v < lo || v > hi {
                return Err(GivpError::InvalidInitialGuess(format!(
                    "value {v} out of bounds [{lo}, {hi}] at index {i}"
                )));
            }
        }
    }
    Ok((lower, upper))
}

/// Perform path relinking on elite pool pairs.
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
    if all.len() < 2 {
        return;
    }

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
    let mut cache = if config.use_cache {
        Some(EvaluationCache::new(config.cache_size))
    } else {
        None
    };

    let mut elite_pool = ElitePool::new(config.elite_size, 0.05, &lower, &upper);
    let mut conv_monitor = if config.use_convergence_monitor {
        Some(ConvergenceMonitor::new(20, 50))
    } else {
        None
    };

    let deadline = if config.time_limit > 0.0 {
        Some(Instant::now() + Duration::from_secs_f64(config.time_limit))
    } else {
        None
    };

    // Initialize best solution
    let mut best_solution: Vec<f64>;
    let mut best_cost: f64;

    if let Some(ref ig) = config.initial_guess {
        best_solution = ig.clone();
        normalize_integer_tail(&mut best_solution, half);
        best_cost = evaluate_with_cache(&best_solution, &wrapped, &mut cache, half);
    } else {
        let mut child = child_rng(&mut rng);
        let (sol, cost) = construct_grasp(
            num_vars,
            &lower,
            &upper,
            &wrapped,
            None,
            config.alpha,
            half,
            config.num_candidates_per_step,
            &mut cache,
            &mut child,
            deadline,
        );
        best_solution = sol;
        best_cost = cost;
    }

    if config.use_elite_pool {
        elite_pool.add(best_solution.clone(), best_cost);
    }

    let mut stagnation: usize = 0;
    let mut message = String::new();

    // Main loop
    for iteration in 0..config.max_iterations {
        if expired(deadline) {
            message = "time limit reached".into();
            break;
        }

        let alpha = get_current_alpha(
            iteration,
            config.max_iterations,
            config.alpha_min,
            config.alpha_max,
            config.adaptive_alpha,
            config.alpha,
        );

        // GRASP construction
        let mut child = child_rng(&mut rng);
        let ig = if iteration == 0 {
            config.initial_guess.as_deref()
        } else {
            None
        };
        let (mut candidate, _grasp_cost) = construct_grasp(
            num_vars,
            &lower,
            &upper,
            &wrapped,
            ig,
            alpha,
            half,
            config.num_candidates_per_step,
            &mut cache,
            &mut child,
            deadline,
        );

        // VND local search
        let grasp_eval = evaluate_with_cache(&candidate, &wrapped, &mut cache, half);
        let vnd_cost = local_search_vnd(
            &wrapped,
            &mut candidate,
            grasp_eval,
            half,
            &lower,
            &upper,
            config.vnd_iterations,
            &mut cache,
            &mut child,
            deadline,
        );

        // ILS
        let ils_cost = ils_search(
            &wrapped,
            &mut candidate,
            vnd_cost,
            half,
            &lower,
            &upper,
            config.ils_iterations,
            config.vnd_iterations,
            config.perturbation_strength,
            &mut cache,
            &mut child,
            deadline,
        );

        // Update best
        if ils_cost < best_cost {
            best_cost = ils_cost;
            best_solution = candidate.clone();
            stagnation = 0;
        } else {
            stagnation += 1;
        }

        // Elite pool
        if config.use_elite_pool {
            elite_pool.add(candidate, ils_cost);
        }

        // Convergence monitor
        if let Some(ref mut cm) = conv_monitor {
            let signal = cm.update(best_cost, Some(&elite_pool));

            if signal.should_restart {
                elite_pool.keep_top(2);
                cm.reset_no_improve();
                stagnation = 0;
                if let Some(ref mut c) = cache {
                    c.clear();
                }
            }
        }

        // Path relinking
        if config.use_elite_pool
            && iteration > 0
            && iteration % config.path_relink_frequency == 0
            && elite_pool.len() >= 2
        {
            do_path_relinking(
                &wrapped,
                &elite_pool,
                &mut best_solution,
                &mut best_cost,
                half,
                &lower,
                &upper,
                config.vnd_iterations,
                &mut cache,
                &mut child,
                deadline,
            );
        }

        // Stagnation restart
        if stagnation > config.max_iterations / 4 {
            let mut restart = crate::grasp::construct_grasp(
                num_vars,
                &lower,
                &upper,
                &wrapped,
                None,
                alpha,
                half,
                config.num_candidates_per_step,
                &mut cache,
                &mut child,
                deadline,
            );
            let restart_cost = local_search_vnd(
                &wrapped,
                &mut restart.0,
                restart.1,
                half,
                &lower,
                &upper,
                config.vnd_iterations,
                &mut cache,
                &mut child,
                deadline,
            );
            let restart_cost = ils_search(
                &wrapped,
                &mut restart.0,
                restart_cost,
                half,
                &lower,
                &upper,
                config.ils_iterations,
                config.vnd_iterations,
                config.perturbation_strength,
                &mut cache,
                &mut child,
                deadline,
            );
            if restart_cost < best_cost {
                best_cost = restart_cost;
                best_solution = restart.0;
            }
            stagnation = 0;
        }

        // Early stop
        if let Some(ref mut cm) = conv_monitor {
            let signal = cm.update(best_cost, Some(&elite_pool));
            if signal.no_improve_count >= config.early_stop_threshold {
                message = "early stop due to stagnation".into();
                break;
            }
        }

        if iteration == config.max_iterations - 1 {
            message = "max iterations reached".into();
        }
    }

    // Build result
    let final_cost = if is_maximize { -best_cost } else { best_cost };

    let mut result = OptimizeResult::new(config.direction);
    result.x = best_solution;
    result.fun = final_cost;
    result.nit = config.max_iterations;
    result.nfev = nfev.load(Ordering::Relaxed);
    result.success = final_cost.is_finite();
    result.message = if message.is_empty() {
        "optimization completed".into()
    } else {
        message.clone()
    };
    result.termination = TerminationReason::from_message(&result.message);

    Ok(result)
}
