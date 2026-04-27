// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::cache::EvaluationCache;
use crate::grasp::evaluate_with_cache;
use crate::helpers::{clamp, expired, normalize_integer_tail};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Try integer moves for a single variable: base-1, base, base+1.
fn try_integer_moves<F>(
    idx: usize,
    solution: &mut Vec<f64>,
    best_cost: f64,
    func: &F,
    lower: &[f64],
    upper: &[f64],
    cache: &mut Option<EvaluationCache>,
    half: usize,
) -> (f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let base = solution[idx].round();
    let mut best = best_cost;
    let mut improved = false;

    for delta in [-1.0, 0.0, 1.0] {
        let val = clamp(base + delta, lower[idx], upper[idx]).round();
        if (val - solution[idx]).abs() < 1e-12 {
            continue;
        }
        let old = solution[idx];
        solution[idx] = val;
        let cost = evaluate_with_cache(solution, func, cache, half);
        if cost < best {
            best = cost;
            improved = true;
        } else {
            solution[idx] = old;
        }
    }
    (best, improved)
}

/// Try a continuous move: ±5% of span.
fn try_continuous_move<F>(
    idx: usize,
    solution: &mut Vec<f64>,
    best_cost: f64,
    func: &F,
    rng: &mut ChaCha8Rng,
    lower: &[f64],
    upper: &[f64],
    cache: &mut Option<EvaluationCache>,
    half: usize,
) -> (f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let span = upper[idx] - lower[idx];
    let delta = rng.random_range(-0.05..=0.05) * span;
    let new_val = clamp(solution[idx] + delta, lower[idx], upper[idx]);
    let old = solution[idx];
    solution[idx] = new_val;
    let cost = evaluate_with_cache(solution, func, cache, half);
    if cost < best_cost {
        (cost, true)
    } else {
        solution[idx] = old;
        (best_cost, false)
    }
}

/// Perturb a single index (for swap/multiflip neighborhoods).
fn perturb_index(
    solution: &mut [f64],
    idx: usize,
    strength: usize,
    rng: &mut ChaCha8Rng,
    lower: &[f64],
    upper: &[f64],
    half: usize,
) {
    if idx >= half {
        let step = (strength as f64 / 2.0).max(1.0);
        let delta = rng.random_range(-step..=step);
        solution[idx] = clamp((solution[idx] + delta).round(), lower[idx], upper[idx]);
    } else {
        let span = upper[idx] - lower[idx];
        let delta = rng.random_range(-0.15..=0.15) * span;
        solution[idx] = clamp(solution[idx] + delta, lower[idx], upper[idx]);
    }
}

/// Neighborhood 1: single-variable flip (prioritized by sensitivity).
fn neighborhood_flip<F>(
    solution: &mut Vec<f64>,
    best_cost: f64,
    func: &F,
    sensitivity: &mut [f64],
    rng: &mut ChaCha8Rng,
    lower: &[f64],
    upper: &[f64],
    cache: &mut Option<EvaluationCache>,
    half: usize,
    deadline: Option<Instant>,
) -> (f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let n = solution.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| sensitivity[b].partial_cmp(&sensitivity[a]).unwrap());

    let mut current_best = best_cost;
    let mut any_improved = false;

    for &idx in &indices {
        if expired(deadline) {
            break;
        }
        let (new_cost, improved) = if idx >= half {
            try_integer_moves(idx, solution, current_best, func, lower, upper, cache, half)
        } else {
            try_continuous_move(
                idx,
                solution,
                current_best,
                func,
                rng,
                lower,
                upper,
                cache,
                half,
            )
        };
        if improved {
            let delta = current_best - new_cost;
            sensitivity[idx] = 0.9 * sensitivity[idx] + delta;
            current_best = new_cost;
            any_improved = true;
        }
    }
    (current_best, any_improved)
}

/// Neighborhood 2: paired variable swap.
fn neighborhood_swap<F>(
    solution: &mut Vec<f64>,
    best_cost: f64,
    func: &F,
    rng: &mut ChaCha8Rng,
    lower: &[f64],
    upper: &[f64],
    cache: &mut Option<EvaluationCache>,
    half: usize,
    deadline: Option<Instant>,
) -> (f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let n = solution.len();
    let mut current_best = best_cost;
    let mut any_improved = false;
    let max_attempts = 50.min(n * (n - 1) / 2);

    for _ in 0..max_attempts {
        if expired(deadline) {
            break;
        }
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        if i == j {
            continue;
        }
        let old_i = solution[i];
        let old_j = solution[j];
        perturb_index(solution, i, 4, rng, lower, upper, half);
        perturb_index(solution, j, 4, rng, lower, upper, half);
        normalize_integer_tail(solution, half);
        let cost = evaluate_with_cache(solution, func, cache, half);
        if cost < current_best {
            current_best = cost;
            any_improved = true;
        } else {
            solution[i] = old_i;
            solution[j] = old_j;
        }
    }
    (current_best, any_improved)
}

/// Neighborhood 3: simultaneous k-variable changes.
fn neighborhood_multiflip<F>(
    solution: &mut Vec<f64>,
    best_cost: f64,
    func: &F,
    rng: &mut ChaCha8Rng,
    lower: &[f64],
    upper: &[f64],
    cache: &mut Option<EvaluationCache>,
    half: usize,
    deadline: Option<Instant>,
) -> (f64, bool)
where
    F: Fn(&[f64]) -> f64,
{
    let n = solution.len();
    let k = 3.min(n);
    let mut current_best = best_cost;
    let mut any_improved = false;

    for _ in 0..50 {
        if expired(deadline) {
            break;
        }
        let backup = solution.to_vec();
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle for k elements
        for i in 0..k {
            let j = rng.random_range(i..n);
            indices.swap(i, j);
        }
        for &idx in &indices[..k] {
            perturb_index(solution, idx, 4, rng, lower, upper, half);
        }
        normalize_integer_tail(solution, half);
        let cost = evaluate_with_cache(solution, func, cache, half);
        if cost < current_best {
            current_best = cost;
            any_improved = true;
        } else {
            solution.copy_from_slice(&backup);
        }
    }
    (current_best, any_improved)
}

/// VND: Variable Neighborhood Descent.
pub(crate) fn local_search_vnd<F>(
    func: &F,
    solution: &mut Vec<f64>,
    current_cost: f64,
    half: usize,
    lower: &[f64],
    upper: &[f64],
    max_iter: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut ChaCha8Rng,
    deadline: Option<Instant>,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let n = solution.len();
    let mut sensitivity = vec![0.0_f64; n];
    let mut best_cost = current_cost;
    let no_improve_limit = 5;
    let mut no_improve_flip_count = 0;

    for _ in 0..max_iter {
        if expired(deadline) {
            break;
        }

        // Decay sensitivity
        for s in sensitivity.iter_mut() {
            *s *= 0.9;
        }

        // Try flip neighborhood
        let (cost, improved) = neighborhood_flip(
            solution,
            best_cost,
            func,
            &mut sensitivity,
            rng,
            lower,
            upper,
            cache,
            half,
            deadline,
        );
        best_cost = cost;

        if !improved {
            no_improve_flip_count += 1;

            // Try swap neighborhood
            let (cost2, improved2) = neighborhood_swap(
                solution, best_cost, func, rng, lower, upper, cache, half, deadline,
            );
            best_cost = cost2;

            if !improved2 && no_improve_flip_count >= no_improve_limit {
                // Try multiflip neighborhood
                let (cost3, _) = neighborhood_multiflip(
                    solution, best_cost, func, rng, lower, upper, cache, half, deadline,
                );
                best_cost = cost3;
            }

            if no_improve_flip_count >= no_improve_limit {
                break;
            }
        } else {
            no_improve_flip_count = 0;
        }
    }
    best_cost
}
