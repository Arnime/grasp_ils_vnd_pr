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
        match &self.range {
            Some(r) => {
                let n = a.len() as f64;
                a.iter()
                    .zip(b.iter())
                    .zip(r.iter())
                    .map(|((&ai, &bi), &ri)| (ai - bi).abs() / ri)
                    .sum::<f64>()
                    / n
            }
            None => a
                .iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| (ai - bi).powi(2))
                .sum::<f64>()
                .sqrt(),
        }
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
        if let Some(worst) = self.pool.last() {
            if cost < worst.1 {
                self.pool.pop();
                self.pool.push((solution, cost));
                self.pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                return true;
            }
        }
        false
    }

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

    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Keep only the top `n` entries.
    pub fn keep_top(&mut self, n: usize) {
        self.pool.truncate(n);
    }
}
