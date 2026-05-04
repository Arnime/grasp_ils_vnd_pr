// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use crate::config::Direction;
use std::collections::HashMap;

/// Why the optimizer stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TerminationReason {
    Converged,
    MaxIterationsReached,
    TimeLimitReached,
    EarlyStop,
    NoFeasible,
    Unknown,
}

impl TerminationReason {
    pub fn from_message(msg: &str) -> Self {
        let lower = msg.to_lowercase();
        if lower.contains("converged") {
            Self::Converged
        } else if lower.contains("max") && lower.contains("iteration") {
            Self::MaxIterationsReached
        } else if lower.contains("time") {
            Self::TimeLimitReached
        } else if lower.contains("early") || lower.contains("stagnation") {
            Self::EarlyStop
        } else if lower.contains("no feasible") || lower.contains("no_feasible") {
            Self::NoFeasible
        } else {
            Self::Unknown
        }
    }
}

/// Optimization result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizeResult {
    /// Best solution found.
    pub x: Vec<f64>,
    /// Best objective value.
    pub fun: f64,
    /// Number of iterations.
    pub nit: usize,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the optimization succeeded.
    pub success: bool,
    /// Human-readable termination message.
    pub message: String,
    /// Optimization direction used.
    pub direction: Direction,
    /// Termination reason.
    pub termination: TerminationReason,
    /// Extra metadata.
    pub meta: HashMap<String, String>,
}

impl OptimizeResult {
    pub(crate) fn new(direction: Direction) -> Self {
        Self {
            x: Vec::new(),
            fun: f64::INFINITY,
            nit: 0,
            nfev: 0,
            success: true,
            message: String::new(),
            direction,
            termination: TerminationReason::Unknown,
            meta: HashMap::new(),
        }
    }
}
