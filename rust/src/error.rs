// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

use thiserror::Error;

/// All errors produced by the GIVP library.
#[derive(Debug, Error)]
pub enum GivpError {
    #[error("invalid bounds: {0}")]
    InvalidBounds(String),

    #[error("invalid initial guess: {0}")]
    InvalidInitialGuess(String),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("evaluator error: {0}")]
    Evaluator(String),

    #[error("empty elite pool: {0}")]
    EmptyPool(String),
}

pub type Result<T> = std::result::Result<T, GivpError>;
