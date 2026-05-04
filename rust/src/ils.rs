// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::cache::EvaluationCache;
use crate::helpers::{clamp, expired, normalize_integer_tail};
use crate::vnd::local_search_vnd;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Perturb a solution by modifying a subset of variables.
fn perturb_solution(
    solution: &[f64],
    half: usize,
    strength: usize,
    lower: &[f64],
    upper: &[f64],
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let n = solution.len();
    let num_perturb = strength.min(n / 5).max(1);
    let mut perturbed = solution.to_vec();

    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..num_perturb {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }

    for &idx in &indices[..num_perturb] {
        if idx >= half {
            let step = (strength as f64 / 2.0).max(1.0);
            let delta = rng.random_range(-step..=step);
            perturbed[idx] = clamp((perturbed[idx] + delta).round(), lower[idx], upper[idx]);
        } else {
            let span = upper[idx] - lower[idx];
            let delta = rng.random_range(-0.15..=0.15) * span;
            perturbed[idx] = clamp(perturbed[idx] + delta, lower[idx], upper[idx]);
        }
    }

    normalize_integer_tail(&mut perturbed, half);
    perturbed
}

/// Iterated Local Search.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ils_search<F>(
    func: &F,
    solution: &mut Vec<f64>,
    current_cost: f64,
    half: usize,
    lower: &[f64],
    upper: &[f64],
    ils_iterations: usize,
    vnd_iterations: usize,
    perturbation_strength: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut ChaCha8Rng,
    deadline: Option<Instant>,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut best_cost = current_cost;
    let mut best_sol = solution.clone();

    for i in 0..ils_iterations {
        if expired(deadline) {
            break;
        }

        // Progressive adaptive strength
        let progress = i as f64 / ils_iterations.max(1) as f64;
        let effective_strength =
            perturbation_strength.max((perturbation_strength as f64 * (1.0 + progress)) as usize);

        let mut candidate =
            perturb_solution(&best_sol, half, effective_strength, lower, upper, rng);
        let perturbed_cost = crate::grasp::evaluate_with_cache(&candidate, func, cache, half);

        let vnd_cost = local_search_vnd(
            func,
            &mut candidate,
            perturbed_cost,
            half,
            lower,
            upper,
            vnd_iterations,
            cache,
            rng,
            deadline,
        );

        if vnd_cost < best_cost {
            best_cost = vnd_cost;
            best_sol = candidate;
        } else if vnd_cost < best_cost * 1.25 && rng.random_range(0.0..1.0) < 0.1 {
            // Accept slightly worse with 10% probability (diversification)
            best_sol = candidate;
            best_cost = vnd_cost;
        }
    }

    *solution = best_sol;
    best_cost
}
