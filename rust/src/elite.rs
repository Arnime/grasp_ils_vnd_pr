// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::error::GivpError;

/// Diversity-aware elite pool of best solutions.
pub(crate) struct ElitePool {
    max_size: usize,
    min_distance: f64,
    pool: Vec<(Vec<f64>, f64)>,
    range: Option<Vec<f64>>,
}

impl ElitePool {
    pub fn new(max_size: usize, min_distance: f64, lower: &[f64], upper: &[f64]) -> Self {
        let range: Vec<f64> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lo, &hi)| (hi - lo).max(1e-12))
            .collect();
        Self {
            max_size,
            min_distance,
            pool: Vec::with_capacity(max_size + 1),
            range: Some(range),
        }
    }

    fn relative_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let r = self.range.as_deref().expect("range always set by new()");
        let n = a.len() as f64;
        a.iter()
            .zip(b.iter())
            .zip(r.iter())
            .map(|((&ai, &bi), &ri)| (ai - bi).abs() / ri)
            .sum::<f64>()
            / n
    }

    pub fn add(&mut self, solution: Vec<f64>, cost: f64) -> bool {
        // Reject if too close to any existing member
        for (existing, _) in &self.pool {
            if self.relative_distance(&solution, existing) < self.min_distance {
                return false;
            }
        }

        if self.pool.len() < self.max_size {
            self.pool.push((solution, cost));
            self.pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return true;
        }

        // Replace worst if new is better
        let worst_cost = self.pool.last().expect("pool non-empty when full").1;
        if cost < worst_cost {
            self.pool.pop();
            self.pool.push((solution, cost));
            self.pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return true;
        }
        false
    }

    #[allow(dead_code)]
    pub fn get_best(&self) -> Result<(&[f64], f64), GivpError> {
        self.pool
            .first()
            .map(|(s, c)| (s.as_slice(), *c))
            .ok_or_else(|| GivpError::EmptyPool("elite pool is empty".into()))
    }

    pub fn get_all(&self) -> &[(Vec<f64>, f64)] {
        &self.pool
    }

    pub fn len(&self) -> usize {
        self.pool.len()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Keep only the top `n` entries.
    pub fn keep_top(&mut self, n: usize) {
        self.pool.truncate(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(max_size: usize) -> ElitePool {
        ElitePool::new(max_size, 0.01, &[-5.0, -5.0], &[5.0, 5.0])
    }

    #[test]
    fn test_get_best_empty_returns_error() {
        let pool = make_pool(5);
        assert!(matches!(pool.get_best(), Err(GivpError::EmptyPool(_))));
    }

    #[test]
    fn test_get_best_returns_lowest_cost() {
        let mut pool = make_pool(5);
        pool.add(vec![1.0, 0.0], 5.0);
        pool.add(vec![-3.0, 2.0], 1.0);
        let (_, cost) = pool.get_best().unwrap();
        assert!((cost - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clear_empties_pool() {
        let mut pool = make_pool(5);
        pool.add(vec![1.0, 0.0], 5.0);
        assert_eq!(pool.len(), 1);
        pool.clear();
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_add_rejects_when_full_and_not_better() {
        let mut pool = make_pool(2);
        pool.add(vec![1.0, 0.0], 1.0); // cost 1
        pool.add(vec![-4.0, 3.0], 2.0); // cost 2 (diverse enough)
                                        // Pool is full; add something with worse cost (3.0) → rejected
        let added = pool.add(vec![4.0, -4.0], 3.0);
        assert!(!added);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_add_replaces_worst_when_better() {
        let mut pool = make_pool(2);
        pool.add(vec![1.0, 0.0], 2.0);
        pool.add(vec![-4.0, 3.0], 3.0);
        // Add something better (cost 0.5) that is diverse enough
        let added = pool.add(vec![0.0, 0.0], 0.5);
        assert!(added);
        let (_, cost) = pool.get_best().unwrap();
        assert!((cost - 0.5).abs() < 1e-10);
    }
}
