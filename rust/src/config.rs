// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::error::{GivpError, Result};

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    Minimize,
    Maximize,
}

impl Default for Direction {
    fn default() -> Self {
        Self::Minimize
    }
}

/// Algorithm hyper-parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GivpConfig {
    pub max_iterations: usize,
    pub alpha: f64,
    pub vnd_iterations: usize,
    pub ils_iterations: usize,
    pub perturbation_strength: usize,
    pub use_elite_pool: bool,
    pub elite_size: usize,
    pub path_relink_frequency: usize,
    pub adaptive_alpha: bool,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub num_candidates_per_step: usize,
    pub use_cache: bool,
    pub cache_size: usize,
    pub early_stop_threshold: usize,
    pub use_convergence_monitor: bool,
    pub n_workers: usize,
    pub time_limit: f64,
    pub direction: Direction,
    pub integer_split: Option<usize>,
    pub group_size: Option<usize>,
    pub initial_guess: Option<Vec<f64>>,
    pub seed: Option<u64>,
    pub verbose: bool,
}

impl Default for GivpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            alpha: 0.12,
            vnd_iterations: 200,
            ils_iterations: 10,
            perturbation_strength: 4,
            use_elite_pool: true,
            elite_size: 7,
            path_relink_frequency: 8,
            adaptive_alpha: true,
            alpha_min: 0.08,
            alpha_max: 0.18,
            num_candidates_per_step: 20,
            use_cache: true,
            cache_size: 10_000,
            early_stop_threshold: 80,
            use_convergence_monitor: true,
            n_workers: 1,
            time_limit: 0.0,
            direction: Direction::Minimize,
            integer_split: None,
            group_size: None,
            initial_guess: None,
            seed: None,
            verbose: false,
        }
    }
}

impl GivpConfig {
    /// Validate all numeric ranges.
    pub fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(GivpError::InvalidConfig(
                "max_iterations must be > 0".into(),
            ));
        }
        if self.vnd_iterations == 0 {
            return Err(GivpError::InvalidConfig(
                "vnd_iterations must be > 0".into(),
            ));
        }
        if self.ils_iterations == 0 {
            return Err(GivpError::InvalidConfig(
                "ils_iterations must be > 0".into(),
            ));
        }
        if self.elite_size == 0 {
            return Err(GivpError::InvalidConfig("elite_size must be > 0".into()));
        }
        if self.path_relink_frequency == 0 {
            return Err(GivpError::InvalidConfig(
                "path_relink_frequency must be > 0".into(),
            ));
        }
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(GivpError::InvalidConfig(
                "alpha must be in [0.0, 1.0]".into(),
            ));
        }
        if self.alpha_min < 0.0 || self.alpha_min > 1.0 {
            return Err(GivpError::InvalidConfig(
                "alpha_min must be in [0.0, 1.0]".into(),
            ));
        }
        if self.alpha_max < 0.0 || self.alpha_max > 1.0 {
            return Err(GivpError::InvalidConfig(
                "alpha_max must be in [0.0, 1.0]".into(),
            ));
        }
        if self.alpha_min > self.alpha_max {
            return Err(GivpError::InvalidConfig(
                "alpha_min must be <= alpha_max".into(),
            ));
        }
        if self.perturbation_strength == 0 {
            return Err(GivpError::InvalidConfig(
                "perturbation_strength must be > 0".into(),
            ));
        }
        if self.cache_size == 0 {
            return Err(GivpError::InvalidConfig("cache_size must be > 0".into()));
        }
        if self.early_stop_threshold == 0 {
            return Err(GivpError::InvalidConfig(
                "early_stop_threshold must be > 0".into(),
            ));
        }
        if self.num_candidates_per_step == 0 {
            return Err(GivpError::InvalidConfig(
                "num_candidates_per_step must be > 0".into(),
            ));
        }
        Ok(())
    }
}
