# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

using Test
using GIVP
using Random
using LinearAlgebra

@testset "GIVP.jl" begin

    # =========================================================================
    # exceptions.jl
    # =========================================================================
    @testset "Exceptions" begin
        @testset "Config validation errors" begin
            @test_throws InvalidConfigError validate_config!(GIVPConfig(max_iterations=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(vnd_iterations=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(ils_iterations=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(elite_size=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(path_relink_frequency=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(num_candidates_per_step=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(cache_size=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(early_stop_threshold=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(n_workers=0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha=-0.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha=1.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_min=-0.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_min=1.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_max=-0.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_max=1.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_min=0.5, alpha_max=0.1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(time_limit=-1.0))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(perturbation_strength=-1))
            @test_throws InvalidConfigError validate_config!(GIVPConfig(integer_split=-1))
        end

        @testset "showerror coverage" begin
            buf = IOBuffer()
            Base.showerror(buf, InvalidBoundsError("test bounds"))
            @test occursin("InvalidBoundsError", String(take!(buf)))

            Base.showerror(buf, InvalidInitialGuessError("test guess"))
            @test occursin("InvalidInitialGuessError", String(take!(buf)))

            Base.showerror(buf, InvalidConfigError("test config"))
            @test occursin("InvalidConfigError", String(take!(buf)))

            Base.showerror(buf, EvaluatorError("test eval"))
            @test occursin("EvaluatorError", String(take!(buf)))

            Base.showerror(buf, EmptyPoolError("test pool"))
            @test occursin("EmptyPoolError", String(take!(buf)))
        end

        @testset "Exception hierarchy" begin
            @test InvalidBoundsError <: GivpError
            @test InvalidInitialGuessError <: GivpError
            @test InvalidConfigError <: GivpError
            @test EvaluatorError <: GivpError
            @test EmptyPoolError <: GivpError
            @test GivpError <: Exception
        end
    end

    # =========================================================================
    # config.jl
    # =========================================================================
    @testset "GIVPConfig" begin
        @testset "defaults" begin
            cfg = GIVPConfig()
            @test cfg.max_iterations == 100
            @test cfg.alpha == 0.12
            @test cfg.vnd_iterations == 200
            @test cfg.ils_iterations == 10
            @test cfg.perturbation_strength == 4
            @test cfg.use_elite_pool == true
            @test cfg.elite_size == 7
            @test cfg.path_relink_frequency == 8
            @test cfg.adaptive_alpha == true
            @test cfg.alpha_min == 0.08
            @test cfg.alpha_max == 0.18
            @test cfg.num_candidates_per_step == 20
            @test cfg.use_cache == true
            @test cfg.cache_size == 10000
            @test cfg.early_stop_threshold == 80
            @test cfg.use_convergence_monitor == true
            @test cfg.n_workers == 1
            @test cfg.time_limit == 0.0
            @test cfg.direction == minimize
            @test cfg.integer_split === nothing
            @test cfg.group_size === nothing
            validate_config!(cfg)
        end

        @testset "valid edge values" begin
            cfg = validate_config!(GIVPConfig(alpha=0.0, alpha_min=0.0, alpha_max=0.0))
            @test cfg.alpha == 0.0
            cfg = validate_config!(GIVPConfig(alpha=1.0, alpha_min=0.0, alpha_max=1.0))
            @test cfg.alpha == 1.0
            cfg = validate_config!(GIVPConfig(perturbation_strength=0))
            @test cfg.perturbation_strength == 0
            cfg = validate_config!(GIVPConfig(time_limit=0.0))
            @test cfg.time_limit == 0.0
            cfg = validate_config!(GIVPConfig(integer_split=0))
            @test cfg.integer_split == 0
        end

        @testset "Direction enum" begin
            @test minimize isa Direction
            @test maximize isa Direction
        end
    end

    # =========================================================================
    # result.jl
    # =========================================================================
    @testset "OptimizeResult" begin
        @testset "construction and fields" begin
            r = OptimizeResult(; x=[1.0, 2.0], fun=3.0, nit=10, nfev=100, success=true,
                              message="converged", direction=minimize)
            @test r.x == [1.0, 2.0]
            @test r.fun == 3.0
            @test r.nit == 10
            @test r.nfev == 100
            @test r.success == true
            @test r.message == "converged"
            @test r.direction == minimize
            @test r.meta == Dict{String,Any}()
        end

        @testset "default constructor" begin
            r = OptimizeResult()
            @test r.x == Float64[]
            @test r.fun == Inf
            @test r.nit == 0
            @test r.nfev == 0
            @test r.success == true
            @test r.message == ""
            @test r.direction == minimize
        end

        @testset "to_dict" begin
            r = OptimizeResult(; x=[1.0], fun=2.0, nit=5, nfev=50,
                              success=false, message="no feasible", direction=maximize)
            d = to_dict(r)
            @test d["x"] == [1.0]
            @test d["fun"] == 2.0
            @test d["nit"] == 5
            @test d["nfev"] == 50
            @test d["success"] == false
            @test d["termination"] == "no_feasible"
            @test d["direction"] == "maximize"
        end

        @testset "tuple unpacking (iterate)" begin
            r = OptimizeResult(; x=[1.0, 2.0], fun=3.0)
            x, fun = r
            @test x == [1.0, 2.0]
            @test fun == 3.0
            @test length(r) == 2
            # Explicit state-by-state iteration to cover all branches
            @test iterate(r) == ([1.0, 2.0], 2)       # state=1
            @test iterate(r, 2) == (3.0, 3)            # state=2
            @test iterate(r, 3) === nothing             # state=3
        end
    end

    @testset "TerminationReason" begin
        @test GIVP.termination_from_message("converged successfully") == GIVP.converged
        @test GIVP.termination_from_message("time limit reached") == GIVP.time_limit_reached
        @test GIVP.termination_from_message("early stop triggered") == GIVP.early_stop
        @test GIVP.termination_from_message("threshold exceeded") == GIVP.early_stop
        @test GIVP.termination_from_message("no feasible solution") == GIVP.no_feasible
        @test GIVP.termination_from_message("no solution found") == GIVP.no_feasible
        @test GIVP.termination_from_message("max iterations reached") == GIVP.max_iterations_reached
        @test GIVP.termination_from_message("iteration limit") == GIVP.max_iterations_reached
        @test GIVP.termination_from_message("something else entirely") == GIVP.unknown
        @test GIVP.termination_from_message("") == GIVP.unknown
    end

    # =========================================================================
    # helpers.jl
    # =========================================================================
    @testset "Helpers" begin
        @testset "set_seed! and new_rng determinism" begin
            GIVP.set_seed!(42)
            rng1 = GIVP.new_rng()
            GIVP.set_seed!(42)
            rng2 = GIVP.new_rng()
            @test rand(rng1) == rand(rng2)

            # explicit seed overrides master
            rng3 = GIVP.new_rng(99)
            rng4 = GIVP.new_rng(99)
            @test rand(rng3) == rand(rng4)

            # no seed, no master → non-deterministic (just verify it works)
            GIVP.set_seed!(nothing)
            rng5 = GIVP.new_rng()
            @test rng5 isa AbstractRNG
        end

        @testset "get_half / set_integer_split!" begin
            GIVP.set_integer_split!(nothing)
            @test GIVP.get_half(10) == 5
            @test GIVP.get_half(7) == 3

            GIVP.set_integer_split!(3)
            @test GIVP.get_half(10) == 3
            @test GIVP.get_half(3) == 3

            # split > n → falls back to n÷2
            GIVP.set_integer_split!(20)
            @test GIVP.get_half(10) == 5

            GIVP.set_integer_split!(0)
            @test GIVP.get_half(10) == 0

            GIVP.set_integer_split!(nothing)
        end

        @testset "set_group_size! / get_group_size" begin
            GIVP.set_group_size!(nothing)
            @test GIVP.get_group_size() === nothing
            GIVP.set_group_size!(24)
            @test GIVP.get_group_size() == 24
            GIVP.set_group_size!(nothing)
        end

        @testset "safe_evaluate" begin
            @test GIVP.safe_evaluate(x -> sum(x.^2), [1.0, 2.0]) == 5.0
            @test GIVP.safe_evaluate(x -> 0.0, [1.0]) == 0.0
            @test GIVP.safe_evaluate(x -> error("fail"), [1.0]) == Inf
            @test GIVP.safe_evaluate(x -> NaN, [1.0]) == Inf
            @test GIVP.safe_evaluate(x -> Inf, [1.0]) == Inf
            @test GIVP.safe_evaluate(x -> -Inf, [1.0]) == Inf
        end

        @testset "expired" begin
            @test !GIVP.expired(0.0)  # 0 = no deadline
            @test !GIVP.expired(time() + 1000)
            @test GIVP.expired(time() - 1.0)
        end
    end

    # =========================================================================
    # cache.jl
    # =========================================================================
    @testset "EvaluationCache" begin
        @testset "basic operations" begin
            GIVP.set_integer_split!(4)  # all continuous
            cache = GIVP.EvaluationCache(; maxsize=3)

            sol1 = [1.0, 2.0, 3.0, 4.0]
            @test GIVP.cache_get(cache, sol1) === nothing
            @test cache.misses == 1

            GIVP.cache_put!(cache, sol1, 10.0)
            @test GIVP.cache_get(cache, sol1) == 10.0
            @test cache.hits == 1

            stats = GIVP.cache_stats(cache)
            @test stats["hits"] == 1
            @test stats["misses"] == 1
            @test stats["size"] == 1
            @test stats["hit_rate"] == 50.0
        end

        @testset "LRU eviction" begin
            GIVP.set_integer_split!(2)
            cache = GIVP.EvaluationCache(; maxsize=2)

            GIVP.cache_put!(cache, [1.0, 2.0], 10.0)
            GIVP.cache_put!(cache, [3.0, 4.0], 20.0)
            @test GIVP.cache_stats(cache)["size"] == 2

            # Adding third entry evicts the oldest (first)
            GIVP.cache_put!(cache, [5.0, 6.0], 30.0)
            @test GIVP.cache_stats(cache)["size"] == 2
            @test GIVP.cache_get(cache, [1.0, 2.0]) === nothing  # evicted
            @test GIVP.cache_get(cache, [3.0, 4.0]) == 20.0  # still there
            @test GIVP.cache_get(cache, [5.0, 6.0]) == 30.0  # still there
        end

        @testset "overwrite existing key" begin
            GIVP.set_integer_split!(2)
            cache = GIVP.EvaluationCache(; maxsize=5)
            GIVP.cache_put!(cache, [1.0, 2.0], 10.0)
            GIVP.cache_put!(cache, [1.0, 2.0], 99.0)  # same key, different value
            @test GIVP.cache_get(cache, [1.0, 2.0]) == 99.0
            @test GIVP.cache_stats(cache)["size"] == 1  # no duplicate entries
        end

        @testset "clear" begin
            GIVP.set_integer_split!(2)
            cache = GIVP.EvaluationCache(; maxsize=10)
            GIVP.cache_put!(cache, [1.0, 2.0], 10.0)
            GIVP.cache_clear!(cache)
            @test cache.hits == 0
            @test cache.misses == 0
            # After clear, previously cached entry is gone
            @test GIVP.cache_get(cache, [1.0, 2.0]) === nothing
            @test cache.misses == 1  # incremented by cache_get miss
        end

        @testset "hit_rate when empty" begin
            cache = GIVP.EvaluationCache()
            stats = GIVP.cache_stats(cache)
            @test stats["hit_rate"] == 0.0
        end

        @testset "integer rounding in hash" begin
            GIVP.set_integer_split!(1)  # first var continuous, rest integer
            cache = GIVP.EvaluationCache()
            # Two solutions that round to the same thing
            GIVP.cache_put!(cache, [1.001, 2.4], 10.0)
            @test GIVP.cache_get(cache, [1.001, 2.6]) === nothing  # different integer part
            @test GIVP.cache_get(cache, [1.001, 2.3]) == 10.0  # rounds to same
        end

        GIVP.set_integer_split!(nothing)
    end

    # =========================================================================
    # elite.jl
    # =========================================================================
    @testset "ElitePool" begin
        @testset "basic operations" begin
            ep = GIVP.ElitePool(; max_size=3, min_distance=0.01,
                               lower=[0.0, 0.0], upper=[10.0, 10.0])
            @test GIVP.pool_size(ep) == 0
            @test_throws EmptyPoolError GIVP.get_best(ep)

            @test GIVP.add!(ep, [1.0, 1.0], 5.0)
            @test GIVP.add!(ep, [5.0, 5.0], 3.0)
            @test GIVP.pool_size(ep) == 2

            best_sol, best_cost = GIVP.get_best(ep)
            @test best_cost == 3.0
            @test best_sol == [5.0, 5.0]

            all_sols = GIVP.get_all(ep)
            @test length(all_sols) == 2
            @test all_sols[1][2] <= all_sols[2][2]  # sorted by cost
        end

        @testset "min_distance rejection" begin
            ep = GIVP.ElitePool(; max_size=5, min_distance=0.5,
                               lower=[0.0], upper=[10.0])
            @test GIVP.add!(ep, [5.0], 1.0)
            # Too close — should be rejected
            @test !GIVP.add!(ep, [5.01], 0.5)
        end

        @testset "replace worst when full" begin
            ep = GIVP.ElitePool(; max_size=2, min_distance=0.01,
                               lower=[0.0], upper=[100.0])
            @test GIVP.add!(ep, [10.0], 10.0)
            @test GIVP.add!(ep, [50.0], 20.0)
            @test GIVP.pool_size(ep) == 2

            # Better than worst → replaces it
            @test GIVP.add!(ep, [90.0], 15.0)
            @test GIVP.pool_size(ep) == 2
            best_sol, best_cost = GIVP.get_best(ep)
            @test best_cost == 10.0
        end

        @testset "reject when full and worse" begin
            ep = GIVP.ElitePool(; max_size=2, min_distance=0.01,
                               lower=[0.0], upper=[100.0])
            GIVP.add!(ep, [10.0], 5.0)
            GIVP.add!(ep, [50.0], 10.0)
            # Worse than worst → rejected
            @test !GIVP.add!(ep, [90.0], 100.0)
        end

        @testset "without bounds (norm distance)" begin
            ep = GIVP.ElitePool(; max_size=5, min_distance=0.01)
            @test ep._range === nothing
            @test GIVP.add!(ep, [1.0, 2.0], 5.0)
            @test GIVP.add!(ep, [100.0, 200.0], 3.0)
        end

        @testset "pool_clear!" begin
            ep = GIVP.ElitePool(; max_size=3)
            GIVP.add!(ep, [1.0], 1.0)
            GIVP.pool_clear!(ep)
            @test GIVP.pool_size(ep) == 0
        end
    end

    # =========================================================================
    # convergence.jl
    # =========================================================================
    @testset "ConvergenceMonitor" begin
        @testset "restart trigger" begin
            cm = GIVP.ConvergenceMonitor(; restart_threshold=5)
            ep = GIVP.ElitePool(; max_size=3)

            # First update improves best_ever from Inf
            status = GIVP.update!(cm, 10.0, ep)
            @test !status["should_restart"]
            @test cm.no_improve_count == 0

            for _ in 1:4
                status = GIVP.update!(cm, 10.0, ep)
            end
            @test cm.no_improve_count == 4
            @test !status["should_restart"]

            status = GIVP.update!(cm, 10.0, ep)
            @test status["should_restart"]
            @test cm.no_improve_count == 5
        end

        @testset "improvement resets counter" begin
            cm = GIVP.ConvergenceMonitor(; restart_threshold=10)
            GIVP.update!(cm, 10.0)
            GIVP.update!(cm, 10.0)
            GIVP.update!(cm, 10.0)
            @test cm.no_improve_count == 2
            GIVP.update!(cm, 5.0)  # improvement
            @test cm.no_improve_count == 0
        end

        @testset "should_intensify" begin
            cm = GIVP.ConvergenceMonitor(; restart_threshold=10)
            ep = GIVP.ElitePool(; max_size=3)
            GIVP.update!(cm, 10.0, ep)
            for _ in 1:5
                GIVP.update!(cm, 10.0, ep)
            end
            status = GIVP.update!(cm, 10.0, ep)
            # no_improve_count >= 5 (threshold//2) and diversity < 0.5
            @test status["should_intensify"]
        end

        @testset "diversity with elite pool" begin
            cm = GIVP.ConvergenceMonitor(; restart_threshold=100)
            ep = GIVP.ElitePool(; max_size=5, min_distance=0.01,
                               lower=[0.0, 0.0], upper=[100.0, 100.0])
            GIVP.add!(ep, [10.0, 10.0], 1.0)
            GIVP.add!(ep, [90.0, 90.0], 2.0)
            status = GIVP.update!(cm, 5.0, ep)
            @test status["diversity"] > 0.0
        end

        @testset "without elite pool" begin
            cm = GIVP.ConvergenceMonitor()
            status = GIVP.update!(cm, 10.0, nothing)
            @test status["diversity"] == 0.0
        end
    end

    # =========================================================================
    # grasp.jl
    # =========================================================================
    @testset "GRASP" begin
        @testset "validate_bounds_and_initial!" begin
            lower = [0.0, 0.0]
            upper = [10.0, 10.0]
            # valid
            GIVP.validate_bounds_and_initial!(lower, upper, nothing, 2)
            GIVP.validate_bounds_and_initial!(lower, upper, [5.0, 5.0], 2)

            # lower length mismatch
            @test_throws InvalidBoundsError GIVP.validate_bounds_and_initial!([0.0], upper, nothing, 2)
            # upper length mismatch
            @test_throws InvalidBoundsError GIVP.validate_bounds_and_initial!(lower, [10.0], nothing, 2)
            # initial_guess wrong length
            @test_throws InvalidInitialGuessError GIVP.validate_bounds_and_initial!(lower, upper, [5.0], 2)
            # initial_guess out of bounds
            @test_throws InvalidInitialGuessError GIVP.validate_bounds_and_initial!(lower, upper, [0.0, 5.0], 2)
            @test_throws InvalidInitialGuessError GIVP.validate_bounds_and_initial!(lower, upper, [5.0, 10.0], 2)
        end

        @testset "get_current_alpha" begin
            cfg = GIVPConfig(; adaptive_alpha=true, alpha_min=0.1, alpha_max=0.5, max_iterations=10)
            # iter 0 → alpha_min
            @test GIVP.get_current_alpha(0, cfg) ≈ 0.1
            # iter 9 → alpha_max
            @test GIVP.get_current_alpha(9, cfg) ≈ 0.5

            cfg2 = GIVPConfig(; adaptive_alpha=false, alpha=0.3)
            @test GIVP.get_current_alpha(0, cfg2) == 0.3
            @test GIVP.get_current_alpha(99, cfg2) == 0.3
        end

        @testset "evaluate_with_cache" begin
            GIVP.set_integer_split!(2)
            # without cache
            @test GIVP.evaluate_with_cache([1.0, 2.0], x -> sum(x.^2), nothing) == 5.0

            # with cache
            cache = GIVP.EvaluationCache()
            @test GIVP.evaluate_with_cache([1.0, 2.0], x -> sum(x.^2), cache) == 5.0
            # second call hits cache
            @test GIVP.evaluate_with_cache([1.0, 2.0], x -> 999.0, cache) == 5.0  # cached
            @test cache.hits == 1

            # non-finite not cached
            cache2 = GIVP.EvaluationCache()
            @test GIVP.evaluate_with_cache([1.0, 2.0], x -> error("boom"), cache2) == Inf
            @test GIVP.cache_stats(cache2)["size"] == 0
            GIVP.set_integer_split!(nothing)
        end

        @testset "select_from_rcl" begin
            rng = MersenneTwister(42)
            # all infinite → nothing
            @test GIVP.select_from_rcl([Inf, Inf], 0.5, rng) === nothing

            # single valid
            @test GIVP.select_from_rcl([Inf, 5.0], 0.5, rng) == 2

            # alpha=0 → greedy (selects minimum)
            idx = GIVP.select_from_rcl([10.0, 1.0, 5.0], 0.0, rng)
            @test idx == 2
        end

        @testset "normalize_integer_tail!" begin
            GIVP.set_integer_split!(2)
            sol = [1.5, 2.7, 3.4, 4.6]
            GIVP.normalize_integer_tail!(sol, 2)
            @test sol[1] == 1.5  # continuous, unchanged
            @test sol[2] == 2.7  # continuous, unchanged
            @test sol[3] == 3.0  # rounded
            @test sol[4] == 5.0  # rounded
            GIVP.set_integer_split!(nothing)
        end

        @testset "sample_integer_from_bounds" begin
            rng = MersenneTwister(42)
            # normal case
            val = GIVP.sample_integer_from_bounds(1.0, 5.0, rng)
            @test val == round(val)
            @test 1.0 <= val <= 5.0

            # hi < lo (degenerate) → midpoint fallback
            # ceil(3.1)=4, floor(3.9)=3, so hi(3) < lo(4) → fallback
            val = GIVP.sample_integer_from_bounds(3.1, 3.9, rng)
            @test val == round(Int, (3.1 + 3.9) / 2.0)  # midpoint = 4
        end

        @testset "build_random_candidate" begin
            GIVP.set_integer_split!(2)
            rng = MersenneTwister(42)
            lower = [0.0, 0.0, 1.0, 1.0]
            upper = [10.0, 10.0, 5.0, 5.0]
            sol = GIVP.build_random_candidate(4, 2, lower, upper, rng)
            @test length(sol) == 4
            @test all(lower .<= sol .<= upper)
            GIVP.set_integer_split!(nothing)
        end

        @testset "build_heuristic_candidate" begin
            GIVP.set_integer_split!(2)
            rng = MersenneTwister(42)
            lower = [0.0, 0.0, 1.0, 1.0]
            upper = [10.0, 10.0, 5.0, 5.0]
            sol = GIVP.build_heuristic_candidate(4, 2, lower, upper, rng)
            @test length(sol) == 4
            @test all(lower .<= sol .<= upper)

            # Fully continuous (half == num_vars)
            sol2 = GIVP.build_heuristic_candidate(2, 2, [0.0, 0.0], [10.0, 10.0], rng)
            @test length(sol2) == 2
            GIVP.set_integer_split!(nothing)
        end

        @testset "build_heuristic_candidate edge: hi <= lo" begin
            GIVP.set_integer_split!(1)
            rng = MersenneTwister(42)
            # Integer bounds [2.9, 3.1] → lo=3, hi=3, so hi == lo
            sol = GIVP.build_heuristic_candidate(2, 1, [0.0, 2.9], [10.0, 3.1], rng)
            @test sol[2] == 3.0
            GIVP.set_integer_split!(nothing)
        end

        @testset "construct_grasp" begin
            GIVP.set_integer_split!(4)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            lower = [-5.0, -5.0, -5.0, -5.0]
            upper = [5.0, 5.0, 5.0, 5.0]

            sol = GIVP.construct_grasp(4, lower, upper, sphere, nothing, 0.3;
                num_candidates_per_step=10)
            @test length(sol) == 4
            @test all(lower .<= sol .<= upper)

            # with initial guess
            sol2 = GIVP.construct_grasp(4, lower, upper, sphere, [1.0, 1.0, 1.0, 1.0], 0.3;
                num_candidates_per_step=10)
            @test length(sol2) == 4

            # with cache
            cache = GIVP.EvaluationCache()
            sol3 = GIVP.construct_grasp(4, lower, upper, sphere, nothing, 0.3;
                cache=cache, num_candidates_per_step=10)
            @test GIVP.cache_stats(cache)["size"] > 0

            GIVP.set_integer_split!(nothing)
        end

        @testset "construct_grasp with infeasible initial guess" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            bad_eval(x) = Inf  # always infeasible
            lower = [0.0, 0.0]
            upper = [10.0, 10.0]
            sol = GIVP.construct_grasp(2, lower, upper, bad_eval, [5.0, 5.0], 0.3;
                num_candidates_per_step=5)
            @test length(sol) == 2
            GIVP.set_integer_split!(nothing)
        end
    end

    # =========================================================================
    # vnd.jl
    # =========================================================================
    @testset "VND" begin
        @testset "create_cached_cost_fn" begin
            GIVP.set_integer_split!(2)
            cache = GIVP.EvaluationCache()
            fn = GIVP.create_cached_cost_fn(x -> sum(x.^2), cache)
            @test fn([1.0, 2.0]) == 5.0
            @test fn([1.0, 2.0]) == 5.0  # cache hit
            @test cache.hits == 1

            # without cache
            fn2 = GIVP.create_cached_cost_fn(x -> sum(x.^2), nothing)
            @test fn2([1.0, 2.0]) == 5.0
            GIVP.set_integer_split!(nothing)
        end

        @testset "try_integer_moves" begin
            GIVP.set_integer_split!(0)  # all integer
            cost_fn(x) = sum(x.^2)

            # with bounds — should improve (covers return result, c, true with bounds)
            sol = [5.0, 5.0]
            new_sol, new_cost, improved = GIVP.try_integer_moves(
                1, copy(sol), cost_fn(sol), cost_fn, [0.0, 0.0], [10.0, 10.0])
            @test improved
            @test new_cost < 50.0

            # with no bounds
            sol2 = [5.0]
            new_sol2, new_cost2, imp2 = GIVP.try_integer_moves(
                1, copy(sol2), cost_fn(sol2), cost_fn, nothing, nothing)
            @test imp2

            # no improvement possible — already at optimal for that index
            sol3 = [0.0, 0.0]
            new_sol3, new_cost3, imp3 = GIVP.try_integer_moves(
                1, copy(sol3), cost_fn(sol3), cost_fn, [0.0, 0.0], [0.0, 0.0])
            @test !imp3
            GIVP.set_integer_split!(nothing)
        end

        @testset "try_continuous_move" begin
            GIVP.set_integer_split!(2)
            rng = MersenneTwister(42)
            sol = [3.0, 3.0]
            cost_fn(x) = sum(x.^2)
            # Try multiple times to get an improvement
            improved = false
            for _ in 1:100
                changed, _ = GIVP.try_continuous_move(
                    1, sol, cost_fn(sol), cost_fn, rng, [0.0, 0.0], [10.0, 10.0])
                if changed
                    improved = true
                    break
                end
            end
            @test improved

            # without bounds
            rng2 = MersenneTwister(42)
            sol2 = [3.0]
            GIVP.try_continuous_move(1, sol2, cost_fn(sol2), cost_fn, rng2, nothing, nothing)
            GIVP.set_integer_split!(nothing)
        end

        @testset "perturb_index! continuous" begin
            GIVP.set_integer_split!(2)
            rng = MersenneTwister(42)
            sol = [5.0, 5.0, 3.0, 3.0]
            # Continuous index (1)
            GIVP.perturb_index!(sol, 1, 4, rng, [0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 5.0, 5.0])
            @test sol[1] != 5.0
            @test 0.0 <= sol[1] <= 10.0

            # Without bounds
            sol2 = [5.0, 5.0]
            GIVP.perturb_index!(sol2, 1, 4, rng, nothing, nothing)
            @test sol2[1] != 5.0
            GIVP.set_integer_split!(nothing)
        end

        @testset "perturb_index! integer" begin
            GIVP.set_integer_split!(2)
            rng = MersenneTwister(42)
            sol = [5.0, 5.0, 3.0, 3.0]
            # Integer index (3 > half=2)
            GIVP.perturb_index!(sol, 3, 4, rng, [0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 5.0, 5.0])
            @test sol[3] == round(sol[3])
            @test 1.0 <= sol[3] <= 5.0

            # Without bounds
            sol2 = [5.0, 5.0, 3.0, 3.0]
            GIVP.perturb_index!(sol2, 3, 4, rng, nothing, nothing)
            @test sol2[3] == round(sol2[3])
            GIVP.set_integer_split!(nothing)
        end

        @testset "neighborhood_flip" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 2.0, 2.0]
            lower = [-5.0, -5.0, 0.0, 0.0]
            upper = [5.0, 5.0, 5.0, 5.0]
            new_sol, new_cost = GIVP.neighborhood_flip(sphere, sol, sphere(sol), 4;
                lower=lower, upper=upper, seed=42)
            @test new_cost <= sphere(sol)

            # with sensitivity
            sens = [10.0, 0.1, 5.0, 0.5]
            new_sol2, new_cost2 = GIVP.neighborhood_flip(sphere, sol, sphere(sol), 4;
                lower=lower, upper=upper, sensitivity=sens, seed=42)
            @test new_cost2 <= sphere(sol)

            # best improvement (first_improvement=false)
            new_sol3, new_cost3 = GIVP.neighborhood_flip(sphere, sol, sphere(sol), 4;
                lower=lower, upper=upper, first_improvement=false, seed=42)
            @test new_cost3 <= sphere(sol)
            GIVP.set_integer_split!(nothing)
        end

        @testset "neighborhood_swap" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 2.0, 2.0]
            lower = [-5.0, -5.0, 0.0, 0.0]
            upper = [5.0, 5.0, 5.0, 5.0]
            new_sol, new_cost = GIVP.neighborhood_swap(sphere, sol, sphere(sol), 4;
                lower=lower, upper=upper)
            @test new_cost <= sphere(sol)

            # no improvement scenario (covers final return at end of loop)
            # Use a function where current sol is already optimal-ish
            flat_fn(x) = 0.0  # always returns 0, no swap can improve
            ns_sol, ns_cost = GIVP.neighborhood_swap(flat_fn, [1.0, 1.0, 1.0, 1.0],
                0.0, 4; lower=[-5.0,-5.0,0.0,0.0], upper=[5.0,5.0,5.0,5.0],
                max_attempts=5, first_improvement=false)
            @test ns_cost == 0.0

            # without bounds
            new_sol2, _ = GIVP.neighborhood_swap(sphere, sol, sphere(sol), 4)
            @test length(new_sol2) == 4

            # skip when half==0 or half==num_vars
            GIVP.set_integer_split!(0)
            new_sol3, nc3 = GIVP.neighborhood_swap(sphere, sol, sphere(sol), 4)
            @test nc3 == sphere(sol)  # no change

            GIVP.set_integer_split!(4)
            new_sol4, nc4 = GIVP.neighborhood_swap(sphere, sol, sphere(sol), 4)
            @test nc4 == sphere(sol)  # no change

            GIVP.set_integer_split!(nothing)
        end

        @testset "neighborhood_multiflip" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 2.0, 2.0]
            lower = [-5.0, -5.0, 0.0, 0.0]
            upper = [5.0, 5.0, 5.0, 5.0]
            new_sol, new_cost = GIVP.neighborhood_multiflip(sphere, sol, sphere(sol), 4;
                lower=lower, upper=upper, seed=42)
            @test length(new_sol) == 4

            # without bounds
            new_sol2, _ = GIVP.neighborhood_multiflip(sphere, sol, sphere(sol), 4;
                seed=42)
            @test length(new_sol2) == 4
            GIVP.set_integer_split!(nothing)
        end

        @testset "try_neighborhoods" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 2.0, 2.0]
            lower = [-5.0, -5.0, 0.0, 0.0]
            upper = [5.0, 5.0, 5.0, 5.0]
            cached_fn = GIVP.create_cached_cost_fn(sphere, nothing)

            new_sol, new_cost, improved = GIVP.try_neighborhoods(
                cached_fn, sol, sphere(sol), 4;
                lower=lower, upper=upper)
            @test new_cost <= sphere(sol)

            # trigger multiflip (iteration divisible by no_improve_flip_limit)
            new_sol2, new_cost2, _ = GIVP.try_neighborhoods(
                cached_fn, sol, sphere(sol), 4;
                lower=lower, upper=upper, iteration=3, no_improve_flip_limit=3)
            @test length(new_sol2) == 4

            # expired deadline
            new_sol3, new_cost3, improved3 = GIVP.try_neighborhoods(
                cached_fn, sol, sphere(sol), 4;
                deadline=time() - 1.0)
            @test !improved3
            GIVP.set_integer_split!(nothing)
        end

        @testset "local_search_vnd" begin
            GIVP.set_integer_split!(4)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 3.0, 3.0]
            lower = [-5.0, -5.0, -5.0, -5.0]
            upper = [5.0, 5.0, 5.0, 5.0]

            improved = GIVP.local_search_vnd(sphere, sol, 4;
                max_iter=50, lower=lower, upper=upper)
            @test sphere(improved) <= sphere(sol)

            # with cache
            cache = GIVP.EvaluationCache()
            improved2 = GIVP.local_search_vnd(sphere, sol, 4;
                max_iter=50, lower=lower, upper=upper, cache=cache)
            @test sphere(improved2) <= sphere(sol)

            # with deadline (expired)
            improved3 = GIVP.local_search_vnd(sphere, sol, 4;
                deadline=time() - 1.0)
            @test improved3 == sol
            GIVP.set_integer_split!(nothing)
        end
    end

    # =========================================================================
    # ils.jl
    # =========================================================================
    @testset "ILS" begin
        @testset "perturb_solution" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sol = [5.0, 5.0, 3.0, 3.0]
            lower = [0.0, 0.0, 1.0, 1.0]
            upper = [10.0, 10.0, 5.0, 5.0]

            perturbed = GIVP.perturb_solution(sol, 4; strength=4, seed=42,
                lower=lower, upper=upper)
            @test length(perturbed) == 4
            @test perturbed != sol  # should be different
            @test all(lower .<= perturbed .<= upper)

            # without bounds
            perturbed2 = GIVP.perturb_solution(sol, 4; strength=4, seed=42)
            @test length(perturbed2) == 4
            GIVP.set_integer_split!(nothing)
        end

        @testset "ils_search" begin
            GIVP.set_integer_split!(4)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            sol = [3.0, 3.0, 3.0, 3.0]
            lower = [-5.0, -5.0, -5.0, -5.0]
            upper = [5.0, 5.0, 5.0, 5.0]
            config = GIVPConfig(; ils_iterations=5, vnd_iterations=20,
                               perturbation_strength=2, integer_split=4)

            best_sol, best_cost = GIVP.ils_search(sol, sphere(sol), 4, sphere, config;
                lower=lower, upper=upper)
            @test best_cost <= sphere(sol)
            @test length(best_sol) == 4

            # with cache (large enough to always hit)
            cache = GIVP.EvaluationCache()
            best_sol2, best_cost2 = GIVP.ils_search(sol, sphere(sol), 4, sphere, config;
                lower=lower, upper=upper, cache=cache)
            @test best_cost2 <= sphere(sol)

            # with tiny cache (maxsize=1) to force cache misses after VND evicts entries
            tiny_cache = GIVP.EvaluationCache(; maxsize=1)
            best_sol5, best_cost5 = GIVP.ils_search(sol, sphere(sol), 4, sphere, config;
                lower=lower, upper=upper, cache=tiny_cache)
            @test best_cost5 <= sphere(sol)

            # without cache (covers else branch: cost_fn(perturbed))
            best_sol4, best_cost4 = GIVP.ils_search(sol, sphere(sol), 4, sphere, config;
                lower=lower, upper=upper)
            @test best_cost4 <= sphere(sol)

            # expired deadline
            best_sol3, best_cost3 = GIVP.ils_search(sol, sphere(sol), 4, sphere, config;
                deadline=time() - 1.0)
            @test best_cost3 == sphere(sol)  # no improvement
            GIVP.set_integer_split!(nothing)
        end
    end

    # =========================================================================
    # pr.jl
    # =========================================================================
    @testset "Path Relinking" begin
        @testset "path_relinking forward" begin
            sphere(x) = sum(x.^2)
            source = [5.0, 5.0, 5.0]
            target = [0.1, 0.1, 0.1]
            best_sol, best_cost = GIVP.path_relinking(sphere, source, target;
                strategy=:forward, seed=42)
            @test best_cost <= sphere(source)
        end

        @testset "path_relinking best" begin
            sphere(x) = sum(x.^2)
            source = [5.0, 5.0, 5.0]
            target = [0.1, 0.1, 0.1]
            best_sol, best_cost = GIVP.path_relinking(sphere, source, target;
                strategy=:best, seed=42)
            @test best_cost <= sphere(source)
        end

        @testset "path_relinking_best no improving move (break)" begin
            # cost function where moving towards target never improves
            # source is at optimum, target is worse in every component
            opt_fn(x) = sum((x .- 1.0).^2)  # optimum at [1,1,1]
            source = [1.0, 1.0, 1.0]
            target = [10.0, 10.0, 10.0]
            best_sol, best_cost = GIVP.path_relinking(opt_fn, source, target;
                strategy=:best, seed=42)
            @test best_cost == opt_fn(source)  # can't improve
        end

        @testset "identical solutions" begin
            sphere(x) = sum(x.^2)
            source = [1.0, 2.0, 3.0]
            best_sol, best_cost = GIVP.path_relinking(sphere, source, copy(source);
                seed=42)
            @test best_cost == sphere(source)
            @test best_sol == source
        end

        @testset "many diff indices (> MAX_PR_VARS)" begin
            sphere(x) = sum(x.^2)
            n = 50
            source = fill(5.0, n)
            target = fill(0.1, n)
            best_sol, best_cost = GIVP.path_relinking(sphere, source, target;
                strategy=:forward, seed=42)
            @test best_cost <= sphere(source)
        end

        @testset "bidirectional_path_relinking" begin
            sphere(x) = sum(x.^2)
            sol1 = [5.0, 5.0]
            sol2 = [0.1, 0.1]
            best_sol, best_cost = GIVP.bidirectional_path_relinking(sphere, sol1, sol2)
            @test best_cost <= sphere(sol1)
        end

        @testset "with expired deadline" begin
            sphere(x) = sum(x.^2)
            source = [5.0, 5.0]
            target = [0.1, 0.1]
            best_sol, best_cost = GIVP.path_relinking(sphere, source, target;
                deadline=time() - 1.0)
            # Should still return something valid
            @test isfinite(best_cost)
        end
    end

    # =========================================================================
    # impl.jl
    # =========================================================================
    @testset "Impl" begin
        @testset "evaluate_with_cache_impl" begin
            GIVP.set_integer_split!(2)
            # without cache
            @test GIVP.evaluate_with_cache_impl([1.0, 2.0], x -> sum(x.^2), nothing) == 5.0

            # with cache
            cache = GIVP.EvaluationCache()
            @test GIVP.evaluate_with_cache_impl([1.0, 2.0], x -> sum(x.^2), cache) == 5.0
            # cached hit
            @test GIVP.evaluate_with_cache_impl([1.0, 2.0], x -> 999.0, cache) == 5.0
            GIVP.set_integer_split!(nothing)
        end

        @testset "do_path_relinking! skip conditions" begin
            cfg = GIVPConfig(; use_elite_pool=false)
            bc, bs, st = GIVP.do_path_relinking!(0, 10.0, [1.0], 0, cfg, nothing,
                x -> sum(x.^2), 1)
            @test bc == 10.0  # unchanged

            # iteration 0 → skip
            cfg2 = GIVPConfig(; use_elite_pool=true, path_relink_frequency=5)
            ep = GIVP.ElitePool(; max_size=5, lower=[0.0], upper=[10.0])
            GIVP.add!(ep, [1.0], 1.0)
            GIVP.add!(ep, [5.0], 5.0)
            bc2, _, _ = GIVP.do_path_relinking!(0, 10.0, [3.0], 0, cfg2, ep,
                x -> sum(x.^2), 1)
            @test bc2 == 10.0  # skip because iteration=0

            # not on frequency
            bc3, _, _ = GIVP.do_path_relinking!(3, 10.0, [3.0], 0, cfg2, ep,
                x -> sum(x.^2), 1)
            @test bc3 == 10.0  # skip because 3 % 5 != 0
        end

        @testset "do_path_relinking! executes" begin
            GIVP.set_integer_split!(2)
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            cfg = GIVPConfig(; use_elite_pool=true, path_relink_frequency=5,
                            vnd_iterations=10, integer_split=2)
            ep = GIVP.ElitePool(; max_size=5, min_distance=0.01,
                               lower=[0.0, 0.0], upper=[10.0, 10.0])
            GIVP.add!(ep, [1.0, 1.0], 2.0)
            GIVP.add!(ep, [5.0, 5.0], 50.0)

            bc, bs, st = GIVP.do_path_relinking!(5, 100.0, [7.0, 7.0], 3, cfg, ep,
                sphere, 2; cache=GIVP.EvaluationCache())
            @test bc <= 100.0
            GIVP.set_integer_split!(nothing)
        end

        @testset "grasp_ils_vnd bounds validation" begin
            cfg = GIVPConfig(; max_iterations=1)
            @test_throws InvalidBoundsError GIVP.grasp_ils_vnd(x -> 0.0, 2, cfg)
            @test_throws InvalidBoundsError GIVP.grasp_ils_vnd(x -> 0.0, 2, cfg;
                lower=[0.0], upper=[1.0, 1.0])
            @test_throws InvalidBoundsError GIVP.grasp_ils_vnd(x -> 0.0, 2, cfg;
                lower=[5.0, 5.0], upper=[1.0, 1.0])
        end

        @testset "grasp_ils_vnd with all features" begin
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)

            # With verbose, time_limit, callback, initial_guess
            callback_count = Ref(0)
            callback_fn(iter, cost, sol) = (callback_count[] += 1)

            cfg = GIVPConfig(; max_iterations=5, vnd_iterations=20, ils_iterations=2,
                            early_stop_threshold=3, use_elite_pool=true, elite_size=3,
                            path_relink_frequency=2, use_cache=true, cache_size=100,
                            use_convergence_monitor=true, integer_split=2)

            sol, cost = GIVP.grasp_ils_vnd(sphere, 2, cfg;
                verbose=true, iteration_callback=callback_fn,
                lower=[0.0, 0.0], upper=[10.0, 10.0],
                initial_guess=[5.0, 5.0])
            @test isfinite(cost)
            @test callback_count[] > 0
        end

        @testset "grasp_ils_vnd with time_limit" begin
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            cfg = GIVPConfig(; max_iterations=1000, vnd_iterations=10, ils_iterations=1,
                            time_limit=0.5, integer_split=2)

            t0 = time()
            sol, cost = GIVP.grasp_ils_vnd(sphere, 2, cfg;
                lower=[0.0, 0.0], upper=[10.0, 10.0])
            elapsed = time() - t0
            @test elapsed < 5.0  # should stop well before 1000 iterations
            @test isfinite(cost)
        end

        @testset "grasp_ils_vnd with time_limit verbose" begin
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            # Very short time limit + verbose to cover the TIME LIMIT log lines
            cfg = GIVPConfig(; max_iterations=10000, vnd_iterations=50, ils_iterations=5,
                            time_limit=0.01, integer_split=2)
            sol, cost = GIVP.grasp_ils_vnd(sphere, 2, cfg;
                verbose=true, lower=[0.0, 0.0], upper=[10.0, 10.0])
            @test isfinite(cost)
        end

        @testset "grasp_ils_vnd stagnation restart improves best" begin
            # Flat function that forces stagnation (all iterations get same cost),
            # then switches to real landscape so restart can improve.
            call_count = Ref(0)
            function flat_then_real(x)
                call_count[] += 1
                if call_count[] < 800
                    return 1000.0  # flat → forces stagnation
                else
                    return sum(x.^2)  # restart evaluations get real cost
                end
            end
            GIVP.set_seed!(42)
            # max_iterations=8 → stagnation threshold = 2
            cfg = GIVPConfig(; max_iterations=8, vnd_iterations=5, ils_iterations=1,
                            early_stop_threshold=200, use_elite_pool=false,
                            use_cache=false, use_convergence_monitor=false,
                            integer_split=2, perturbation_strength=1,
                            num_candidates_per_step=3)
            sol, cost = GIVP.grasp_ils_vnd(flat_then_real, 2, cfg;
                lower=[0.0, 0.0], upper=[10.0, 10.0])
            @test isfinite(cost)
            @test cost < 1000.0  # restart found better than flat
        end

        @testset "grasp_ils_vnd without optional components" begin
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            cfg = GIVPConfig(; max_iterations=3, vnd_iterations=10, ils_iterations=1,
                            use_elite_pool=false, use_cache=false,
                            use_convergence_monitor=false, integer_split=2)

            sol, cost = GIVP.grasp_ils_vnd(sphere, 2, cfg;
                lower=[0.0, 0.0], upper=[10.0, 10.0])
            @test isfinite(cost)
        end

        @testset "grasp_ils_vnd callback error handling" begin
            GIVP.set_seed!(42)
            sphere(x) = sum(x.^2)
            bad_callback(iter, cost, sol) = error("callback crash")

            cfg = GIVPConfig(; max_iterations=2, vnd_iterations=5, ils_iterations=1,
                            integer_split=2)
            # Should not throw despite callback error
            sol, cost = GIVP.grasp_ils_vnd(sphere, 2, cfg;
                lower=[0.0, 0.0], upper=[10.0, 10.0],
                iteration_callback=bad_callback)
            @test isfinite(cost)
        end
    end

    # =========================================================================
    # api.jl
    # =========================================================================
    @testset "API" begin
        @testset "givp minimization (pair bounds)" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=30, vnd_iterations=50, ils_iterations=3,
                               early_stop_threshold=20, integer_split=4)

            result = givp(sphere, bounds; direction=minimize, config=config, seed=42)
            @test result.success
            @test result.fun < 1.0
            @test length(result.x) == 4
            @test result.nit == 30
            @test result.nfev > 0
            @test result.direction == minimize
            @test occursin("completed", result.message)
        end

        @testset "givp maximization" begin
            neg_sphere(x) = -sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=15, vnd_iterations=30, ils_iterations=2,
                               integer_split=2)

            result = givp(neg_sphere, bounds; direction=maximize, config=config, seed=42)
            @test result.success
            @test result.fun < 0.0
            @test result.direction == maximize
        end

        @testset "givp tuple bounds" begin
            sphere(x) = sum(x .^ 2)
            lower = [-5.0, -5.0]
            upper = [5.0, 5.0]
            config = GIVPConfig(; max_iterations=5, vnd_iterations=10, ils_iterations=1,
                               integer_split=2)

            result = givp(sphere, (lower, upper); config=config, seed=1)
            @test result.success
            @test length(result.x) == 2
        end

        @testset "givp with num_vars" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=3, vnd_iterations=5, ils_iterations=1,
                               integer_split=2)

            result = givp(sphere, bounds; num_vars=2, config=config, seed=1)
            @test length(result.x) == 2
        end

        @testset "givp seed reproducibility" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=10, vnd_iterations=20, ils_iterations=2,
                               integer_split=2)

            r1 = givp(sphere, bounds; config=config, seed=123)
            r2 = givp(sphere, bounds; config=config, seed=123)
            @test r1.x == r2.x
            @test r1.fun == r2.fun
        end

        @testset "givp with initial_guess and callback" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=5, vnd_iterations=10, ils_iterations=1,
                               integer_split=2)
            cb_count = Ref(0)

            result = givp(sphere, bounds; config=config, seed=42,
                         initial_guess=[1.0, 1.0],
                         iteration_callback=(i, c, s) -> (cb_count[] += 1))
            @test result.success
            @test cb_count[] == 5
        end

        @testset "givp verbose mode" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=3, vnd_iterations=5, ils_iterations=1,
                               integer_split=2)

            result = givp(sphere, bounds; config=config, seed=42, verbose=true)
            @test result.success
        end

        @testset "givp default config" begin
            sphere(x) = sum(x .^ 2)
            bounds = [(-2.0, 2.0), (-2.0, 2.0)]
            # No config provided → uses defaults (but restrict iterations for speed)
            config = GIVPConfig(; max_iterations=3, vnd_iterations=5, ils_iterations=1)
            result = givp(sphere, bounds; config=config, seed=42)
            @test result.success
        end

        @testset "givp with evaluator that throws" begin
            bad_fn(x) = error("evaluator crash")
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
            config = GIVPConfig(; max_iterations=2, vnd_iterations=5, ils_iterations=1,
                               integer_split=2)
            result = givp(bad_fn, bounds; config=config, seed=42)
            # Should handle gracefully (all evaluations return Inf)
            @test !result.success
        end

        @testset "_normalize_bounds errors" begin
            # Pair bounds, wrong num_vars
            @test_throws ArgumentError GIVP._normalize_bounds(
                [(-1.0, 1.0), (-1.0, 1.0)], 3)

            # Tuple bounds, wrong num_vars
            @test_throws ArgumentError GIVP._normalize_bounds(
                ([-1.0, -1.0], [1.0, 1.0]), 3)

            # Tuple bounds, mismatched lengths
            @test_throws ArgumentError GIVP._normalize_bounds(
                ([-1.0], [1.0, 1.0]), nothing)
        end
    end

end
