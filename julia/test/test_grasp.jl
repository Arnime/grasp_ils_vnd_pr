# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "GRASP" begin
    @testset "validate_bounds_and_initial!" begin
        lower = [0.0, 0.0]
        upper = [10.0, 10.0]
        # valid
        GIVPOptimizer.validate_bounds_and_initial!(lower, upper, nothing, 2)
        GIVPOptimizer.validate_bounds_and_initial!(lower, upper, [5.0, 5.0], 2)

        # lower length mismatch
        @test_throws InvalidBoundsError GIVPOptimizer.validate_bounds_and_initial!(
            [0.0],
            upper,
            nothing,
            2,
        )
        # upper length mismatch
        @test_throws InvalidBoundsError GIVPOptimizer.validate_bounds_and_initial!(
            lower,
            [10.0],
            nothing,
            2,
        )
        # initial_guess wrong length
        @test_throws InvalidInitialGuessError GIVPOptimizer.validate_bounds_and_initial!(
            lower,
            upper,
            [5.0],
            2,
        )
        # initial_guess out of bounds
        @test_throws InvalidInitialGuessError GIVPOptimizer.validate_bounds_and_initial!(
            lower,
            upper,
            [0.0, 5.0],
            2,
        )
        @test_throws InvalidInitialGuessError GIVPOptimizer.validate_bounds_and_initial!(
            lower,
            upper,
            [5.0, 10.0],
            2,
        )
    end

    @testset "get_current_alpha" begin
        cfg = GIVPConfig(;
            adaptive_alpha = true,
            alpha_min = 0.1,
            alpha_max = 0.5,
            max_iterations = 10,
        )
        # iter 0 → alpha_min
        @test GIVPOptimizer.get_current_alpha(0, cfg) ≈ 0.1
        # iter 9 → alpha_max
        @test GIVPOptimizer.get_current_alpha(9, cfg) ≈ 0.5

        cfg2 = GIVPConfig(; adaptive_alpha = false, alpha = 0.3)
        @test GIVPOptimizer.get_current_alpha(0, cfg2) == 0.3
        @test GIVPOptimizer.get_current_alpha(99, cfg2) == 0.3
    end

    @testset "evaluate_with_cache" begin
        GIVPOptimizer.set_integer_split!(2)
        # without cache
        @test GIVPOptimizer.evaluate_with_cache([1.0, 2.0], x -> sum(x .^ 2), nothing) ==
              5.0

        # with cache
        cache = GIVPOptimizer.EvaluationCache()
        @test GIVPOptimizer.evaluate_with_cache([1.0, 2.0], x -> sum(x .^ 2), cache) == 5.0
        # second call hits cache
        @test GIVPOptimizer.evaluate_with_cache([1.0, 2.0], x -> 999.0, cache) == 5.0  # cached
        @test cache.hits == 1

        # non-finite not cached
        cache2 = GIVPOptimizer.EvaluationCache()
        @test GIVPOptimizer.evaluate_with_cache([1.0, 2.0], x -> error("boom"), cache2) ==
              Inf
        @test GIVPOptimizer.cache_stats(cache2)["size"] == 0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "select_from_rcl" begin
        rng = MersenneTwister(42)
        # all infinite → nothing
        @test GIVPOptimizer.select_from_rcl([Inf, Inf], 0.5, rng) === nothing

        # single valid
        @test GIVPOptimizer.select_from_rcl([Inf, 5.0], 0.5, rng) == 2

        # alpha=0 → greedy (selects minimum)
        idx = GIVPOptimizer.select_from_rcl([10.0, 1.0, 5.0], 0.0, rng)
        @test idx == 2
    end

    @testset "normalize_integer_tail!" begin
        GIVPOptimizer.set_integer_split!(2)
        sol = [1.5, 2.7, 3.4, 4.6]
        GIVPOptimizer.normalize_integer_tail!(sol, 2)
        @test sol[1] == 1.5  # continuous, unchanged
        @test sol[2] == 2.7  # continuous, unchanged
        @test sol[3] == 3.0  # rounded
        @test sol[4] == 5.0  # rounded
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "sample_integer_from_bounds" begin
        rng = MersenneTwister(42)
        # normal case
        val = GIVPOptimizer.sample_integer_from_bounds(1.0, 5.0, rng)
        @test val == round(val)
        @test 1.0 <= val <= 5.0

        # hi < lo (degenerate) → midpoint fallback
        # ceil(3.1)=4, floor(3.9)=3, so hi(3) < lo(4) → fallback
        val = GIVPOptimizer.sample_integer_from_bounds(3.1, 3.9, rng)
        @test val == round(Int, (3.1 + 3.9) / 2.0)  # midpoint = 4
    end

    @testset "build_random_candidate" begin
        GIVPOptimizer.set_integer_split!(2)
        rng = MersenneTwister(42)
        lower = [0.0, 0.0, 1.0, 1.0]
        upper = [10.0, 10.0, 5.0, 5.0]
        sol = GIVPOptimizer.build_random_candidate(4, 2, lower, upper, rng)
        @test length(sol) == 4
        @test all(lower .<= sol .<= upper)
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "build_heuristic_candidate" begin
        GIVPOptimizer.set_integer_split!(2)
        rng = MersenneTwister(42)
        lower = [0.0, 0.0, 1.0, 1.0]
        upper = [10.0, 10.0, 5.0, 5.0]
        sol = GIVPOptimizer.build_heuristic_candidate(4, 2, lower, upper, rng)
        @test length(sol) == 4
        @test all(lower .<= sol .<= upper)

        # Fully continuous (half == num_vars)
        sol2 = GIVPOptimizer.build_heuristic_candidate(2, 2, [0.0, 0.0], [10.0, 10.0], rng)
        @test length(sol2) == 2
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "build_heuristic_candidate edge: hi <= lo" begin
        GIVPOptimizer.set_integer_split!(1)
        rng = MersenneTwister(42)
        # Integer bounds [2.9, 3.1] → lo=3, hi=3, so hi == lo
        sol = GIVPOptimizer.build_heuristic_candidate(2, 1, [0.0, 2.9], [10.0, 3.1], rng)
        @test sol[2] == 3.0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "construct_grasp" begin
        GIVPOptimizer.set_integer_split!(4)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        lower = [-5.0, -5.0, -5.0, -5.0]
        upper = [5.0, 5.0, 5.0, 5.0]

        sol = GIVPOptimizer.construct_grasp(
            4,
            lower,
            upper,
            sphere,
            nothing,
            0.3;
            num_candidates_per_step = 10,
        )
        @test length(sol) == 4
        @test all(lower .<= sol .<= upper)

        # with initial guess
        sol2 = GIVPOptimizer.construct_grasp(
            4,
            lower,
            upper,
            sphere,
            [1.0, 1.0, 1.0, 1.0],
            0.3;
            num_candidates_per_step = 10,
        )
        @test length(sol2) == 4

        # with cache
        cache = GIVPOptimizer.EvaluationCache()
        sol3 = GIVPOptimizer.construct_grasp(
            4,
            lower,
            upper,
            sphere,
            nothing,
            0.3;
            cache = cache,
            num_candidates_per_step = 10,
        )
        @test GIVPOptimizer.cache_stats(cache)["size"] > 0

        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "construct_grasp with infeasible initial guess" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        bad_eval(x) = Inf  # always infeasible
        lower = [0.0, 0.0]
        upper = [10.0, 10.0]
        sol = GIVPOptimizer.construct_grasp(
            2,
            lower,
            upper,
            bad_eval,
            [5.0, 5.0],
            0.3;
            num_candidates_per_step = 5,
        )
        @test length(sol) == 2
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "construct_grasp n_workers parallel" begin
        GIVPOptimizer.set_integer_split!(4)  # all continuous
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        lower = [-5.0, -5.0, -5.0, -5.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        # n_workers=2 should return a valid candidate (parallel evaluation path)
        sol = GIVPOptimizer.construct_grasp(
            4,
            lower,
            upper,
            sphere,
            nothing,
            0.3;
            num_candidates_per_step = 10,
            n_workers = 2,
        )
        @test length(sol) == 4
        @test all(lower .<= sol .<= upper)
        GIVPOptimizer.set_integer_split!(nothing)
    end
end
