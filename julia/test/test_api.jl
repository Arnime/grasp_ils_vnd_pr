# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "API" begin
    @testset "givp minimization (pair bounds)" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 30,
            vnd_iterations = 50,
            ils_iterations = 3,
            early_stop_threshold = 20,
            integer_split = 4,
        )

        result = givp(sphere, bounds; direction = minimize, config = config, seed = 42)
        @test result.success
        @test result.fun < 1.0
        @test length(result.x) == 4
        @test 0 < result.nit <= 30
        @test result.nfev > 0
        @test result.direction == minimize
        @test result.message in ("max iterations reached", "early stop triggered")
    end

    @testset "givp maximization" begin
        neg_sphere(x) = -sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 15,
            vnd_iterations = 30,
            ils_iterations = 2,
            integer_split = 2,
        )

        result = givp(neg_sphere, bounds; direction = maximize, config = config, seed = 42)
        @test result.success
        @test result.fun < 0.0
        @test result.direction == maximize
    end

    @testset "givp tuple bounds" begin
        sphere(x) = sum(x .^ 2)
        lower = [-5.0, -5.0]
        upper = [5.0, 5.0]
        config = GIVPConfig(;
            max_iterations = 5,
            vnd_iterations = 10,
            ils_iterations = 1,
            integer_split = 2,
        )

        result = givp(sphere, (lower, upper); config = config, seed = 1)
        @test result.success
        @test length(result.x) == 2
    end

    @testset "givp with num_vars" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 3,
            vnd_iterations = 5,
            ils_iterations = 1,
            integer_split = 2,
        )

        result = givp(sphere, bounds; num_vars = 2, config = config, seed = 1)
        @test length(result.x) == 2
    end

    @testset "givp seed reproducibility" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 10,
            vnd_iterations = 20,
            ils_iterations = 2,
            integer_split = 2,
        )

        r1 = givp(sphere, bounds; config = config, seed = 123)
        r2 = givp(sphere, bounds; config = config, seed = 123)
        @test r1.x == r2.x
        @test r1.fun == r2.fun
    end

    @testset "givp with initial_guess and callback" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 5,
            vnd_iterations = 10,
            ils_iterations = 1,
            integer_split = 2,
        )
        cb_count = Ref(0)

        result = givp(
            sphere,
            bounds;
            config = config,
            seed = 42,
            initial_guess = [1.0, 1.0],
            iteration_callback = (i, c, s) -> (cb_count[] += 1),
        )
        @test result.success
        @test cb_count[] == 5
    end

    @testset "givp verbose mode" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 3,
            vnd_iterations = 5,
            ils_iterations = 1,
            integer_split = 2,
        )

        result = givp(sphere, bounds; config = config, seed = 42, verbose = true)
        @test result.success
    end

    @testset "givp default config" begin
        sphere(x) = sum(x .^ 2)
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        # No config provided → uses defaults (but restrict iterations for speed)
        config = GIVPConfig(; max_iterations = 3, vnd_iterations = 5, ils_iterations = 1)
        result = givp(sphere, bounds; config = config, seed = 42)
        @test result.success
    end

    @testset "givp with evaluator that throws" begin
        bad_fn(x) = error("evaluator crash")
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        config = GIVPConfig(;
            max_iterations = 2,
            vnd_iterations = 5,
            ils_iterations = 1,
            integer_split = 2,
        )
        result = givp(bad_fn, bounds; config = config, seed = 42)
        # Should handle gracefully (all evaluations return Inf)
        @test !result.success || result.fun == Inf
    end

    @testset "_normalize_bounds errors" begin
        # Pair bounds, wrong num_vars
        @test_throws ArgumentError GIVPOptimizer._normalize_bounds(
            [(-1.0, 1.0), (-1.0, 1.0)],
            3,
        )

        # Tuple bounds, wrong num_vars
        @test_throws ArgumentError GIVPOptimizer._normalize_bounds(
            ([-1.0, -1.0], [1.0, 1.0]),
            3,
        )

        # Tuple bounds, mismatched lengths
        @test_throws ArgumentError GIVPOptimizer._normalize_bounds(
            ([-1.0], [1.0, 1.0]),
            nothing,
        )

        # Vector{Vector} bounds — happy path
        lo, hi, n = GIVPOptimizer._normalize_bounds([[-1.0, -2.0], [1.0, 2.0]], nothing)
        @test lo == [-1.0, -2.0]
        @test hi == [1.0, 2.0]
        @test n == 2

        # Vector{Vector} bounds, wrong element count
        @test_throws ArgumentError GIVPOptimizer._normalize_bounds(
            [[-1.0], [1.0], [0.0]],
            nothing,
        )

        # Vector{Vector} bounds, mismatched lengths
        @test_throws ArgumentError GIVPOptimizer._normalize_bounds(
            [[-1.0], [1.0, 2.0]],
            nothing,
        )
    end
end

# ── GIVPOptimizerWrapper ─────────────────────────────────────────────────────

@testset "GIVPOptimizerWrapper" begin
    sphere(x) = sum(xi^2 for xi in x)
    bounds = [(-5.0, 5.0) for _ in 1:3]
    fast_cfg = GIVPConfig(; max_iterations = 3, vnd_iterations = 5, ils_iterations = 1)

    @testset "basic optimize! returns OptimizeResult" begin
        opt = GIVPOptimizerWrapper(sphere, bounds; config = fast_cfg, seed = 1)
        result = optimize!(opt)
        @test result isa OptimizeResult
        @test result.success
        @test isfinite(result.fun)
        @test length(result.x) == 3
    end

    @testset "history accumulates across multiple runs" begin
        opt = GIVPOptimizerWrapper(sphere, bounds; config = fast_cfg, seed = 2)
        optimize!(opt)
        optimize!(opt)
        optimize!(opt)
        @test length(opt.history) == 3
    end

    @testset "best_fun and best_x track the best across runs" begin
        opt = GIVPOptimizerWrapper(sphere, bounds; config = fast_cfg, seed = 3)
        r1 = optimize!(opt)
        @test opt.best_fun == r1.fun
        @test opt.best_x == r1.x

        r2 = optimize!(opt)
        @test opt.best_fun <= min(r1.fun, r2.fun)
    end

    @testset "maximize direction tracks maximum" begin
        neg_sphere(x) = -sum(xi^2 for xi in x)
        opt = GIVPOptimizerWrapper(neg_sphere, bounds; direction = maximize, config = fast_cfg, seed = 4)
        r = optimize!(opt)
        @test r.direction == maximize
        @test opt.best_fun == r.fun
        @test r.fun <= 0.0
    end

    @testset "config is forwarded" begin
        cfg = GIVPConfig(; max_iterations = 5, vnd_iterations = 10, ils_iterations = 2)
        opt = GIVPOptimizerWrapper(sphere, bounds; config = cfg, seed = 5)
        r = optimize!(opt)
        @test r.success
        @test r.nit <= 5
    end

    @testset "evaluator exception is logged and handled" begin
        throwing_func(x) = x[1] > 0 ? error("boom") : sum(x .^ 2)
        opt = GIVPOptimizerWrapper(throwing_func, bounds; config = fast_cfg, seed = 6)
        # Should not throw — exceptions are caught and treated as Inf
        r = optimize!(opt)
        @test r isa OptimizeResult
    end
end
