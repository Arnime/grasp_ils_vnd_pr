// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::error::{GivpError, Result};

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    #[default]
    Minimize,
    Maximize,
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
    fn validate_positive(value: usize, field: &str) -> Result<()> {
        if value == 0 {
            return Err(GivpError::InvalidConfig(format!("{field} must be > 0")));
        }
        Ok(())
    }

    fn validate_unit_interval(value: f64, field: &str) -> Result<()> {
        if !(0.0..=1.0).contains(&value) {
            return Err(GivpError::InvalidConfig(format!(
                "{field} must be in [0.0, 1.0]"
            )));
        }
        Ok(())
    }

    /// Validate all numeric ranges.
    pub fn validate(&self) -> Result<()> {
        Self::validate_positive(self.max_iterations, "max_iterations")?;
        Self::validate_positive(self.vnd_iterations, "vnd_iterations")?;
        Self::validate_positive(self.ils_iterations, "ils_iterations")?;
        Self::validate_positive(self.elite_size, "elite_size")?;
        Self::validate_positive(self.path_relink_frequency, "path_relink_frequency")?;
        Self::validate_unit_interval(self.alpha, "alpha")?;
        Self::validate_unit_interval(self.alpha_min, "alpha_min")?;
        Self::validate_unit_interval(self.alpha_max, "alpha_max")?;
        if self.alpha_min > self.alpha_max {
            return Err(GivpError::InvalidConfig(
                "alpha_min must be <= alpha_max".into(),
            ));
        }
        Self::validate_positive(self.perturbation_strength, "perturbation_strength")?;
        Self::validate_positive(self.cache_size, "cache_size")?;
        Self::validate_positive(self.early_stop_threshold, "early_stop_threshold")?;
        Self::validate_positive(self.num_candidates_per_step, "num_candidates_per_step")?;
        if self.n_workers == 0 {
            return Err(GivpError::InvalidConfig("n_workers must be >= 1".into()));
        }
        Ok(())
    }
}
