// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

//! Minimal CLI for running GIVP on built-in benchmark functions.

use givp::{givp, Direction, GivpConfig};
use std::env;

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn parse_args() -> (String, usize, u64, Direction) {
    let args: Vec<String> = env::args().collect();
    let mut function = String::from("sphere");
    let mut dims = 10usize;
    let mut seed = 42u64;
    let mut direction = Direction::Minimize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--function" if i + 1 < args.len() => {
                function = args[i + 1].clone();
                i += 2;
            }
            "--dims" if i + 1 < args.len() => {
                dims = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(42);
                i += 2;
            }
            "--direction" if i + 1 < args.len() => {
                direction = if args[i + 1].eq_ignore_ascii_case("maximize") {
                    Direction::Maximize
                } else {
                    Direction::Minimize
                };
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    (function, dims.max(1), seed, direction)
}

struct CliResult<'a> {
    function: &'a str,
    dims: usize,
    seed: u64,
    best: f64,
    nfev: usize,
    nit: usize,
    success: bool,
    message: String,
}

fn to_json(record: &CliResult<'_>) -> String {
    format!(
        "{{\"function\":\"{}\",\"dims\":{},\"seed\":{},\"best\":{:.10e},\"nfev\":{},\"nit\":{},\"success\":{},\"message\":\"{}\"}}",
        record.function,
        record.dims,
        record.seed,
        record.best,
        record.nfev,
        record.nit,
        record.success,
        record.message.replace('"', "\\\"")
    )
}

fn main() {
    let (fun_name, dims, seed, direction) = parse_args();
    let bounds = match fun_name.as_str() {
        "rosenbrock" => vec![(-5.0, 10.0); dims],
        _ => vec![(-5.12, 5.12); dims],
    };

    let f: fn(&[f64]) -> f64 = match fun_name.as_str() {
        "rosenbrock" => rosenbrock,
        _ => sphere,
    };

    let cfg = GivpConfig {
        max_iterations: 50,
        seed: Some(seed),
        direction,
        integer_split: Some(dims),
        ..Default::default()
    };

    match givp(f, &bounds, cfg) {
        Ok(result) => {
            let payload = CliResult {
                function: &fun_name,
                dims,
                seed,
                best: result.fun,
                nfev: result.nfev,
                nit: result.nit,
                success: result.success,
                message: result.message,
            };
            println!(
                "{}",
                to_json(&payload)
            );
        }
        Err(e) => {
            let payload = CliResult {
                function: &fun_name,
                dims,
                seed,
                best: f64::INFINITY,
                nfev: 0,
                nit: 0,
                success: false,
                message: e.to_string(),
            };
            eprintln!(
                "{}",
                to_json(&payload)
            );
            std::process::exit(1);
        }
    }
}
