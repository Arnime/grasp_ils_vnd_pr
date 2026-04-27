// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::cache::EvaluationCache;
use crate::helpers::{clamp, normalize_integer_tail, safe_evaluate};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Compute adaptive alpha via linear interpolation.
pub(crate) fn get_current_alpha(
    iter_idx: usize,
    max_iterations: usize,
    alpha_min: f64,
    alpha_max: f64,
    adaptive: bool,
    alpha: f64,
) -> f64 {
    if !adaptive {
        return alpha;
    }
    let progress = iter_idx as f64 / max_iterations.max(1) as f64;
    alpha_min + (alpha_max - alpha_min) * progress
}

/// Evaluate with optional cache.
pub(crate) fn evaluate_with_cache<F>(
    candidate: &[f64],
    func: &F,
    cache: &mut Option<EvaluationCache>,
    half: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    if let Some(ref mut c) = cache {
        if let Some(v) = c.get(candidate, half) {
            return v;
        }
        let cost = safe_evaluate(func, candidate);
        c.put(candidate, half, cost);
        cost
    } else {
        safe_evaluate(func, candidate)
    }
}

/// Select an index from the Restricted Candidate List.
fn select_from_rcl(costs: &[f64], alpha: f64, rng: &mut ChaCha8Rng) -> Option<usize> {
    if costs.is_empty() {
        return None;
    }
    let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_cost = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = min_cost + alpha * (max_cost - min_cost);

    let candidates: Vec<usize> = costs
        .iter()
        .enumerate()
        .filter(|(_, &c)| c <= threshold)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return Some(0);
    }
    let idx = rng.random_range(0..candidates.len());
    Some(candidates[idx])
}

/// Sample an integer uniformly from bounds.
fn sample_integer_from_bounds(lo: f64, hi: f64, rng: &mut ChaCha8Rng) -> f64 {
    let lo_i = lo.ceil() as i64;
    let hi_i = hi.floor() as i64;
    if lo_i > hi_i {
        return ((lo + hi) / 2.0).round();
    }
    rng.random_range(lo_i..=hi_i) as f64
}

/// Build a purely random candidate.
fn build_random_candidate(
    num_vars: usize,
    half: usize,
    lower: &[f64],
    upper: &[f64],
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let mut sol = vec![0.0; num_vars];
    for i in 0..half {
        sol[i] = rng.random_range(lower[i]..=upper[i]);
    }
    for i in half..num_vars {
        sol[i] = sample_integer_from_bounds(lower[i], upper[i], rng);
    }
    sol
}

/// Build a heuristic candidate (mid ± 15% noise).
fn build_heuristic_candidate(
    num_vars: usize,
    half: usize,
    lower: &[f64],
    upper: &[f64],
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let mut sol = vec![0.0; num_vars];
    for i in 0..half {
        let mid = (lower[i] + upper[i]) / 2.0;
        let span = upper[i] - lower[i];
        let noise = rng.random_range(-0.15..=0.15) * span;
        sol[i] = clamp(mid + noise, lower[i], upper[i]);
    }
    for i in half..num_vars {
        sol[i] = sample_integer_from_bounds(lower[i], upper[i], rng);
    }
    sol
}

/// GRASP construction phase.
#[allow(clippy::too_many_arguments)]
pub(crate) fn construct_grasp<F>(
    num_vars: usize,
    lower: &[f64],
    upper: &[f64],
    func: &F,
    initial_guess: Option<&[f64]>,
    alpha: f64,
    half: usize,
    num_candidates: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut ChaCha8Rng,
    deadline: Option<Instant>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(num_candidates);
    let mut costs: Vec<f64> = Vec::with_capacity(num_candidates);

    // Initial guess as first candidate
    if let Some(ig) = initial_guess {
        let mut sol = ig.to_vec();
        normalize_integer_tail(&mut sol, half);
        let cost = evaluate_with_cache(&sol, func, cache, half);
        candidates.push(sol);
        costs.push(cost);
    }

    // One heuristic candidate
    if candidates.len() < num_candidates {
        let mut sol = build_heuristic_candidate(num_vars, half, lower, upper, rng);
        normalize_integer_tail(&mut sol, half);
        let cost = evaluate_with_cache(&sol, func, cache, half);
        candidates.push(sol);
        costs.push(cost);
    }

    // Fill rest with random candidates
    while candidates.len() < num_candidates {
        if crate::helpers::expired(deadline) {
            break;
        }
        let mut sol = build_random_candidate(num_vars, half, lower, upper, rng);
        normalize_integer_tail(&mut sol, half);
        let cost = evaluate_with_cache(&sol, func, cache, half);
        candidates.push(sol);
        costs.push(cost);
    }

    // RCL selection — costs is always non-empty here (heuristic candidate always added)
    let idx = select_from_rcl(&costs, alpha, rng).expect("costs should not be empty");
    (candidates.swap_remove(idx), costs[idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::{Duration, Instant};

    #[test]
    fn test_select_from_rcl_empty_returns_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(select_from_rcl(&[], 0.5, &mut rng).is_none());
    }

    #[test]
    fn test_construct_grasp_expired_deadline_breaks_fill_loop() {
        // An already-expired deadline triggers the break in the fill-random loop.
        // The heuristic candidate is still returned because it was added before the loop.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut cache = None;
        let lower = [-5.0f64, -5.0];
        let upper = [5.0f64, 5.0];
        let func = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let deadline = Some(Instant::now() - Duration::from_secs(1));
        let (sol, cost) = construct_grasp(
            2, &lower, &upper, &func, None, 0.5, 2, 100, &mut cache, &mut rng, deadline,
        );
        assert_eq!(sol.len(), 2);
        assert!(cost.is_finite());
    }
}
