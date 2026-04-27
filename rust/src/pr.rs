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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::{Duration, Instant};

    #[test]
    fn test_identical_solutions_returns_immediately() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let sol = vec![1.0, 2.0, 3.0];
        let (result, cost) = bidirectional_path_relinking(
            &|x: &[f64]| x.iter().sum::<f64>(),
            &sol,
            &sol,
            3,
            &mut None,
            &mut rng,
            None,
        );
        assert_eq!(result, sol);
        assert!((cost - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_expired_deadline_breaks_inner_loop() {
        // deadline already expired → path_relinking_best loop breaks at line 33
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let sol1 = vec![0.0, 0.0, 0.0];
        let sol2 = vec![1.0, 1.0, 1.0];
        let func = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
        assert!((func(&sol1) - 0.0).abs() < 1e-10); // invoke closure body
        let deadline = Some(Instant::now() - Duration::from_secs(1));
        let (result, _cost) =
            bidirectional_path_relinking(&func, &sol1, &sol2, 3, &mut None, &mut rng, deadline);
        assert_eq!(result.len(), 3);
    }

    /// Constructs a function on the {0,1}^3 grid where the backward greedy path
    /// finds [0,1,1] (cost 2.0) — a point the forward greedy path never visits.
    /// Forward best = 5.0 (sol2), backward best = 2.0 → backward wins.
    #[test]
    fn test_backward_path_wins() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let func = |x: &[f64]| {
            let a = x[0] >= 0.5;
            let b = x[1] >= 0.5;
            let c = x[2] >= 0.5;
            match (a, b, c) {
                (false, false, false) => 10.0_f64, // sol1
                (true, false, false) => 8.0,
                (false, true, false) => 9.0,
                (false, false, true) => 11.0,
                (true, true, false) => 6.0,
                (true, false, true) => 7.0,
                (false, true, true) => 2.0, // backward step-1 finds this; forward never does
                (true, true, true) => 5.0,  // sol2
            }
        };
        let sol1 = vec![0.0, 0.0, 0.0];
        let sol2 = vec![1.0, 1.0, 1.0];
        let (result, cost) =
            bidirectional_path_relinking(&func, &sol1, &sol2, 3, &mut None, &mut rng, None);
        // Backward path: [1,1,1]→ set x[0]=0 → [0,1,1]=2.0 (best) → backward wins
        assert!(
            (cost - 2.0).abs() < 1e-10,
            "expected backward best 2.0, got {cost}"
        );
        assert!((result[0]).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }
}
