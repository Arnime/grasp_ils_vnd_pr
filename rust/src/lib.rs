// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! # GIVP — GRASP-ILS-VND with Path Relinking
//!
//! A metaheuristic optimizer for continuous and mixed-integer black-box
//! optimization problems.
//!
//! ## Quick start
//!
//! ```rust
//! use givp::{givp, GivpConfig, Direction};
//!
//! let sphere = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
//! let bounds: Vec<(f64, f64)> = vec![(-5.12, 5.12); 5];
//!
//! let result = givp(sphere, &bounds, GivpConfig::default()).unwrap();
//! println!("Best: {:.6} at {:?}", result.fun, result.x);
//! ```

mod cache;
mod config;
mod convergence;
mod elite;
mod error;
mod grasp;
mod helpers;
mod ils;
mod impl_core;
mod pr;
mod result;
mod vnd;

pub use config::{Direction, GivpConfig};
pub use error::{GivpError, Result};
pub use result::{OptimizeResult, TerminationReason};

/// Run the GRASP-ILS-VND with Path Relinking optimizer.
///
/// # Arguments
///
/// * `func` — Objective function `&[f64] -> f64`.
/// * `bounds` — Variable bounds as `&[(lower, upper)]`.
/// * `config` — Algorithm configuration.
///
/// # Returns
///
/// An [`OptimizeResult`] containing the best solution found.
pub fn givp<F>(func: F, bounds: &[(f64, f64)], config: GivpConfig) -> Result<OptimizeResult>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    impl_core::run(func, bounds, config)
}
