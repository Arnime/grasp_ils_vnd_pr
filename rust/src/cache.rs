// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use std::collections::HashMap;

/// LRU evaluation cache to avoid redundant objective function calls.
pub(crate) struct EvaluationCache {
    maxsize: usize,
    cache: HashMap<u64, f64>,
    insertion_order: Vec<u64>,
    hits: usize,
    misses: usize,
}

impl EvaluationCache {
    pub fn new(maxsize: usize) -> Self {
        Self {
            maxsize,
            cache: HashMap::with_capacity(maxsize),
            insertion_order: Vec::with_capacity(maxsize),
            hits: 0,
            misses: 0,
        }
    }

    fn hash_solution(&self, solution: &[f64], half: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (i, &v) in solution.iter().enumerate() {
            let rounded = if i < half {
                (v * 1000.0).round() as i64
            } else {
                v.round() as i64
            };
            rounded.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn get(&mut self, solution: &[f64], half: usize) -> Option<f64> {
        let key = self.hash_solution(solution, half);
        match self.cache.get(&key) {
            Some(&v) => {
                self.hits += 1;
                Some(v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn put(&mut self, solution: &[f64], half: usize, cost: f64) {
        let key = self.hash_solution(solution, half);
        if self.cache.contains_key(&key) {
            return;
        }
        if self.cache.len() >= self.maxsize {
            if let Some(oldest) = self.insertion_order.first().copied() {
                self.cache.remove(&oldest);
                self.insertion_order.remove(0);
            }
        }
        self.cache.insert(key, cost);
        self.insertion_order.push(key);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.insertion_order.clear();
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> (usize, usize, f64, usize) {
        let total = self.hits + self.misses;
        let rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, rate, self.cache.len())
    }
}
