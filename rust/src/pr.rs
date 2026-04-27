// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::cache::EvaluationCache;
use crate::grasp::evaluate_with_cache;
use crate::helpers::expired;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const MAX_PR_VARS: usize = 25;

/// Best (greedy) path relinking: at each step pick the variable move with best cost.
fn path_relinking_best<F>(
    func: &F,
    source: &[f64],
    target: &[f64],
    diff_indices: &[usize],
    cache: &mut Option<EvaluationCache>,
    half: usize,
    deadline: Option<Instant>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut current = source.to_vec();
    let mut best = current.clone();
    let mut best_cost = evaluate_with_cache(&current, func, cache, half);
    let mut remaining: Vec<usize> = diff_indices.to_vec();

    while !remaining.is_empty() {
        if expired(deadline) {
            break;
        }
        let mut best_idx_pos = 0;
        let mut best_move_cost = f64::INFINITY;
        let mut best_move_val = 0.0;

        for (pos, &idx) in remaining.iter().enumerate() {
            let old = current[idx];
            current[idx] = target[idx];
            let cost = evaluate_with_cache(&current, func, cache, half);
            if cost < best_move_cost {
                best_move_cost = cost;
                best_idx_pos = pos;
                best_move_val = target[idx];
            }
            current[idx] = old;
        }

        let chosen_idx = remaining.swap_remove(best_idx_pos);
        current[chosen_idx] = best_move_val;

        if best_move_cost < best_cost {
            best_cost = best_move_cost;
            best = current.clone();
        }
    }
    (best, best_cost)
}

/// Bidirectional path relinking between two solutions.
pub(crate) fn bidirectional_path_relinking<F>(
    func: &F,
    sol1: &[f64],
    sol2: &[f64],
    half: usize,
    cache: &mut Option<EvaluationCache>,
    rng: &mut ChaCha8Rng,
    deadline: Option<Instant>,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = sol1.len();
    let mut diff_indices: Vec<usize> = (0..n)
        .filter(|&i| (sol1[i] - sol2[i]).abs() > 1e-12)
        .collect();

    if diff_indices.is_empty() {
        let cost = evaluate_with_cache(sol1, func, cache, half);
        return (sol1.to_vec(), cost);
    }

    // Limit to top MAX_PR_VARS most different variables
    if diff_indices.len() > MAX_PR_VARS {
        diff_indices.sort_by(|&a, &b| {
            let da = (sol1[a] - sol2[a]).abs();
            let db = (sol1[b] - sol2[b]).abs();
            db.partial_cmp(&da).unwrap()
        });
        diff_indices.truncate(MAX_PR_VARS);
    }

    // Shuffle order
    for i in (1..diff_indices.len()).rev() {
        let j = rng.random_range(0..=i);
        diff_indices.swap(i, j);
    }

    // Forward: sol1 → sol2 (best strategy)
    let (best_fwd, cost_fwd) =
        path_relinking_best(func, sol1, sol2, &diff_indices, cache, half, deadline);

    // Backward: sol2 → sol1 (best strategy)
    let (best_bwd, cost_bwd) =
        path_relinking_best(func, sol2, sol1, &diff_indices, cache, half, deadline);

    if cost_fwd <= cost_bwd {
        (best_fwd, cost_fwd)
    } else {
        (best_bwd, cost_bwd)
    }
}
