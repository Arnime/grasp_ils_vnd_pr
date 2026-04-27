// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::elite::ElitePool;

/// Tracks convergence and triggers restarts / intensification.
pub(crate) struct ConvergenceMonitor {
    window_size: usize,
    restart_threshold: usize,
    history: Vec<f64>,
    no_improve_count: usize,
    best_ever: f64,
    diversity_scores: Vec<f64>,
}

pub(crate) struct ConvergenceSignal {
    pub should_restart: bool,
    pub should_intensify: bool,
    pub diversity: f64,
    pub no_improve_count: usize,
}

impl ConvergenceMonitor {
    pub fn new(window_size: usize, restart_threshold: usize) -> Self {
        Self {
            window_size,
            restart_threshold,
            history: Vec::new(),
            no_improve_count: 0,
            best_ever: f64::INFINITY,
            diversity_scores: Vec::new(),
        }
    }

    pub fn update(&mut self, current_cost: f64, elite_pool: Option<&ElitePool>) -> ConvergenceSignal {
        if current_cost < self.best_ever {
            self.best_ever = current_cost;
            self.no_improve_count = 0;
        } else {
            self.no_improve_count += 1;
        }

        self.history.push(current_cost);
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }

        let diversity = if let Some(pool) = elite_pool {
            self.compute_diversity(pool)
        } else {
            1.0
        };

        self.diversity_scores.push(diversity);

        let should_restart = self.no_improve_count >= self.restart_threshold;
        let should_intensify =
            self.no_improve_count >= self.restart_threshold / 2 && diversity < 0.5;

        ConvergenceSignal {
            should_restart,
            should_intensify,
            diversity,
            no_improve_count: self.no_improve_count,
        }
    }

    fn compute_diversity(&self, pool: &ElitePool) -> f64 {
        let solutions = pool.get_all();
        if solutions.len() < 2 {
            return 1.0;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let dist: f64 = solutions[i]
                    .0
                    .iter()
                    .zip(solutions[j].0.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total += dist;
                count += 1;
            }
        }
        if count > 0 {
            total / count as f64
        } else {
            1.0
        }
    }

    pub fn reset_no_improve(&mut self) {
        self.no_improve_count = 0;
    }
}
