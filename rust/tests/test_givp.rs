// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod tests {
    use givp::{givp, Direction, GivpConfig, GivpError, TerminationReason};

    // ── Test functions ──────────────────────────────────────────────

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // ── Config validation ───────────────────────────────────────────

    #[test]
    fn test_config_default_is_valid() {
        let cfg = GivpConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_max_iterations() {
        let cfg = GivpConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_vnd_iterations() {
        let cfg = GivpConfig {
            vnd_iterations: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_alpha() {
        let cfg = GivpConfig {
            alpha: -0.1,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_alpha_min_gt_max() {
        let cfg = GivpConfig {
            alpha_min: 0.5,
            alpha_max: 0.1,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    // ── Bounds validation ───────────────────────────────────────────

    #[test]
    fn test_empty_bounds() {
        let cfg = GivpConfig::default();
        let result = givp(sphere, &[], cfg);
        assert!(matches!(result, Err(GivpError::InvalidBounds(_))));
    }

    #[test]
    fn test_inverted_bounds() {
        let cfg = GivpConfig::default();
        let result = givp(sphere, &[(5.0, -5.0)], cfg);
        assert!(matches!(result, Err(GivpError::InvalidBounds(_))));
    }

    #[test]
    fn test_invalid_initial_guess_length() {
        let cfg = GivpConfig {
            initial_guess: Some(vec![0.0, 0.0]),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg);
        assert!(matches!(result, Err(GivpError::InvalidInitialGuess(_))));
    }

    #[test]
    fn test_initial_guess_out_of_bounds() {
        let cfg = GivpConfig {
            initial_guess: Some(vec![100.0]),
            ..Default::default()
        };
        let result = givp(sphere, &[(-5.0, 5.0)], cfg);
        assert!(matches!(result, Err(GivpError::InvalidInitialGuess(_))));
    }

    // ── Optimization ────────────────────────────────────────────────

    #[test]
    fn test_sphere_minimize() {
        let cfg = GivpConfig {
            max_iterations: 30,
            seed: Some(42),
            integer_split: Some(5), // all continuous
            ..Default::default()
        };
        let bounds = vec![(-5.12, 5.12); 5];
        let result = givp(sphere, &bounds, cfg).unwrap();

        assert!(result.success);
        assert!(result.fun < 1.0, "sphere result too high: {}", result.fun);
        assert_eq!(result.x.len(), 5);
    }

    #[test]
    fn test_rosenbrock_minimize() {
        let cfg = GivpConfig {
            max_iterations: 50,
            seed: Some(123),
            integer_split: Some(3), // all continuous
            ..Default::default()
        };
        let bounds = vec![(-5.0, 10.0); 3];
        let result = givp(rosenbrock, &bounds, cfg).unwrap();

        assert!(result.success);
        assert!(result.fun < 50.0, "rosenbrock too high: {}", result.fun);
    }

    #[test]
    fn test_maximize() {
        let neg_sphere = |x: &[f64]| -> f64 { -sphere(x) };
        let cfg = GivpConfig {
            max_iterations: 20,
            seed: Some(42),
            direction: Direction::Maximize,
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.12, 5.12); 3];
        let result = givp(neg_sphere, &bounds, cfg).unwrap();

        assert!(result.success);
        // Maximizing -sphere should find near-zero (maximized value is near 0)
        assert!(result.fun > -1.0, "maximize result: {}", result.fun);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let make_cfg = || GivpConfig {
            max_iterations: 20,
            seed: Some(999),
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.12, 5.12); 3];

        let r1 = givp(sphere, &bounds, make_cfg()).unwrap();
        let r2 = givp(sphere, &bounds, make_cfg()).unwrap();

        assert!(
            (r1.fun - r2.fun).abs() < 1e-10,
            "not deterministic: {} vs {}",
            r1.fun,
            r2.fun
        );
    }

    #[test]
    fn test_time_limit() {
        let cfg = GivpConfig {
            max_iterations: 100_000,
            time_limit: 0.5,
            seed: Some(42),
            integer_split: Some(5),
            ..Default::default()
        };
        let bounds = vec![(-5.12, 5.12); 5];
        let result = givp(sphere, &bounds, cfg).unwrap();

        assert!(result.success);
        assert_eq!(result.termination, TerminationReason::TimeLimitReached);
    }

    #[test]
    fn test_result_fields() {
        let cfg = GivpConfig {
            max_iterations: 10,
            seed: Some(42),
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let result = givp(sphere, &bounds, cfg).unwrap();

        assert!(!result.x.is_empty());
        assert!(result.nfev > 0);
        assert!(!result.message.is_empty());
    }

    // ── TerminationReason ───────────────────────────────────────────

    #[test]
    fn test_termination_from_message() {
        assert_eq!(
            TerminationReason::from_message("converged"),
            TerminationReason::Converged
        );
        assert_eq!(
            TerminationReason::from_message("max iterations reached"),
            TerminationReason::MaxIterationsReached
        );
        assert_eq!(
            TerminationReason::from_message("time limit reached"),
            TerminationReason::TimeLimitReached
        );
        assert_eq!(
            TerminationReason::from_message("early stop due to stagnation"),
            TerminationReason::EarlyStop
        );
        assert_eq!(
            TerminationReason::from_message("unknown thing"),
            TerminationReason::Unknown
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_single_variable() {
        let cfg = GivpConfig {
            max_iterations: 20,
            seed: Some(42),
            integer_split: Some(1),
            ..Default::default()
        };
        let result = givp(sphere, &[(-10.0, 10.0)], cfg).unwrap();
        assert!(result.success);
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_no_cache() {
        let cfg = GivpConfig {
            max_iterations: 10,
            use_cache: false,
            seed: Some(42),
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_no_elite_pool() {
        let cfg = GivpConfig {
            max_iterations: 10,
            use_elite_pool: false,
            seed: Some(42),
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_no_convergence_monitor() {
        let cfg = GivpConfig {
            max_iterations: 10,
            use_convergence_monitor: false,
            seed: Some(42),
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }
}
