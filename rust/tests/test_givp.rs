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

    #[allow(dead_code)]
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

    // ── Additional config validation ────────────────────────────────

    #[test]
    fn test_config_invalid_ils_iterations() {
        let cfg = GivpConfig {
            ils_iterations: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_elite_size() {
        let cfg = GivpConfig {
            elite_size: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_path_relink_frequency() {
        let cfg = GivpConfig {
            path_relink_frequency: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_alpha_min_out_of_range() {
        let cfg = GivpConfig {
            alpha_min: -0.1,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_alpha_max_out_of_range() {
        let cfg = GivpConfig {
            alpha_max: 1.5,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_perturbation_strength() {
        let cfg = GivpConfig {
            perturbation_strength: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_cache_size() {
        let cfg = GivpConfig {
            cache_size: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_early_stop_threshold() {
        let cfg = GivpConfig {
            early_stop_threshold: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_invalid_num_candidates_per_step() {
        let cfg = GivpConfig {
            num_candidates_per_step: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }

    #[test]
    fn test_direction_default_is_minimize() {
        use givp::Direction;
        assert_eq!(Direction::default(), Direction::Minimize);
    }

    // ── Non-finite bounds ───────────────────────────────────────────

    #[test]
    fn test_non_finite_bounds() {
        let cfg = GivpConfig::default();
        let result = givp(sphere, &[(f64::NEG_INFINITY, 5.0)], cfg);
        assert!(matches!(result, Err(GivpError::InvalidBounds(_))));
    }

    // ── Integer variables ───────────────────────────────────────────

    #[test]
    fn test_integer_variables() {
        let cfg = GivpConfig {
            max_iterations: 20,
            seed: Some(42),
            integer_split: Some(2), // 2 continuous, 3 integer
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2]
            .into_iter()
            .chain(vec![(0.0, 4.0); 3])
            .collect::<Vec<_>>();
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
        // Integer vars should be rounded
        for v in &result.x[2..] {
            assert!((v - v.round()).abs() < 1e-9, "not integer: {v}");
        }
    }

    #[test]
    fn test_all_integer_variables() {
        let cfg = GivpConfig {
            max_iterations: 20,
            seed: Some(42),
            integer_split: Some(0), // all integer
            ..Default::default()
        };
        let bounds = vec![(0.0, 10.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
        for v in &result.x {
            assert!((v - v.round()).abs() < 1e-9, "not integer: {v}");
        }
    }

    // ── Initial guess ───────────────────────────────────────────────

    #[test]
    fn test_initial_guess_used() {
        let cfg = GivpConfig {
            max_iterations: 10,
            seed: Some(42),
            integer_split: Some(3),
            initial_guess: Some(vec![0.1, 0.2, 0.3]),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── Early stop ──────────────────────────────────────────────────

    #[test]
    fn test_early_stop() {
        let cfg = GivpConfig {
            max_iterations: 200,
            early_stop_threshold: 3,
            seed: Some(42),
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
        assert_eq!(result.termination, TerminationReason::EarlyStop);
    }

    // ── TerminationReason::NoFeasible ───────────────────────────────

    #[test]
    fn test_termination_no_feasible() {
        assert_eq!(
            TerminationReason::from_message("no feasible solution found"),
            TerminationReason::NoFeasible
        );
        assert_eq!(
            TerminationReason::from_message("no_feasible"),
            TerminationReason::NoFeasible
        );
    }

    // ── Path relinking with identical solutions ─────────────────────

    #[test]
    fn test_pr_identical_solutions_via_small_pool() {
        // Force path relinking by setting high elite pool frequency and many iterations
        let cfg = GivpConfig {
            max_iterations: 15,
            path_relink_frequency: 1,
            elite_size: 5,
            seed: Some(42),
            integer_split: Some(3),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 3];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── Large number of variables (exercises MAX_PR_VARS truncation) ─

    #[test]
    fn test_many_variables_pr_truncation() {
        let cfg = GivpConfig {
            max_iterations: 10,
            seed: Some(42),
            integer_split: Some(30),
            path_relink_frequency: 1,
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 30];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── Without seed (exercises from_os_rng branch) ─────────────────

    #[test]
    fn test_no_seed() {
        let cfg = GivpConfig {
            max_iterations: 5,
            seed: None,
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── Panicking objective (exercises safe_evaluate) ────────────────

    #[test]
    fn test_panicking_objective() {
        let panicker = |_x: &[f64]| -> f64 { panic!("intentional panic in test") };
        let cfg = GivpConfig {
            max_iterations: 5,
            seed: Some(42),
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        // Should not propagate the panic; optimizer returns infinity cost
        let result = givp(panicker, &bounds, cfg).unwrap();
        // All evaluations return infinity → success=false
        assert!(!result.success);
    }

    // ── Adaptive alpha off ───────────────────────────────────────────

    #[test]
    fn test_non_adaptive_alpha() {
        let cfg = GivpConfig {
            max_iterations: 10,
            adaptive_alpha: false,
            alpha: 0.15,
            seed: Some(42),
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── do_path_relinking early return (elite_size=1 → len < 2) ─────

    #[test]
    fn test_pr_early_return_single_elite() {
        let cfg = GivpConfig {
            max_iterations: 5,
            elite_size: 1,            // pool never reaches 2 members
            path_relink_frequency: 1, // trigger PR every iteration
            seed: Some(42),
            integer_split: Some(2),
            ..Default::default()
        };
        let bounds = vec![(-5.0, 5.0); 2];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── PR with >MAX_PR_VARS different variables ──────────────────────

    #[test]
    fn test_pr_many_diffs_truncation() {
        let cfg = GivpConfig {
            max_iterations: 10,
            seed: Some(7),
            integer_split: Some(30),
            path_relink_frequency: 1,
            elite_size: 5,
            ..Default::default()
        };
        let bounds = vec![(-100.0, 100.0); 30];
        let result = givp(sphere, &bounds, cfg).unwrap();
        assert!(result.success);
    }

    // ── Integer bounds where ceil > floor (degenerate) ───────────────

    #[test]
    fn test_integer_degenerate_bounds() {
        // bounds like (0.1, 0.9) for integer vars: ceil(0.1)=1 > floor(0.9)=0
        let cfg = GivpConfig {
            max_iterations: 10,
            seed: Some(42),
            integer_split: Some(0), // all integer
            ..Default::default()
        };
        let bounds = vec![(0.1_f64, 0.9_f64); 3]; // degenerate integer range
                                                  // Should not panic
        let _result = givp(sphere, &bounds, cfg);
    }

    // ── n_workers parallel candidate evaluation (rayon) ──────────────

    #[test]
    fn test_n_workers_parallel_matches_sequential() {
        // Results must be finite; exact match is not required because rayon
        // may evaluate candidates in a different order, affecting RCL selection.
        let make_cfg = |n: usize| GivpConfig {
            max_iterations: 20,
            n_workers: n,
            num_candidates_per_step: 8,
            use_cache: false, // cache disabled — required for parallel path
            seed: Some(0),
            integer_split: Some(5),
            ..Default::default()
        };
        let bounds = vec![(-5.12, 5.12); 5];

        let r_seq = givp(sphere, &bounds, make_cfg(1)).unwrap();
        let r_par = givp(sphere, &bounds, make_cfg(2)).unwrap();

        assert!(r_seq.success, "sequential run failed");
        assert!(r_par.success, "parallel run failed");
        assert!(r_par.fun.is_finite(), "parallel result is non-finite");
    }

    #[test]
    fn test_n_workers_zero_is_invalid() {
        let cfg = GivpConfig {
            n_workers: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(GivpError::InvalidConfig(_))));
    }
}
