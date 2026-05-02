# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "Impl" begin
    @testset "evaluate_with_cache_impl" begin
        GIVPOptimizer.set_integer_split!(2)
        # without cache
        @test GIVPOptimizer.evaluate_with_cache_impl(
            [1.0, 2.0],
            x -> sum(x .^ 2),
            nothing,
        ) == 5.0

        # with cache
        cache = GIVPOptimizer.EvaluationCache()
        @test GIVPOptimizer.evaluate_with_cache_impl([1.0, 2.0], x -> sum(x .^ 2), cache) ==
              5.0
        # cached hit
        @test GIVPOptimizer.evaluate_with_cache_impl([1.0, 2.0], x -> 999.0, cache) == 5.0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "do_path_relinking! skip conditions" begin
        cfg = GIVPConfig(; use_elite_pool = false)
        bc, bs, st = GIVPOptimizer.do_path_relinking!(
            0,
            10.0,
            [1.0],
            0,
            cfg,
            nothing,
            x -> sum(x .^ 2),
            1,
        )
        @test bc == 10.0  # unchanged

        # iteration 0 → skip
        cfg2 = GIVPConfig(; use_elite_pool = true, path_relink_frequency = 5)
        ep = GIVPOptimizer.ElitePool(; max_size = 5, lower = [0.0], upper = [10.0])
        GIVPOptimizer.add!(ep, [1.0], 1.0)
        GIVPOptimizer.add!(ep, [5.0], 5.0)
        bc2, _, _ = GIVPOptimizer.do_path_relinking!(
            0,
            10.0,
            [3.0],
            0,
            cfg2,
            ep,
            x -> sum(x .^ 2),
            1,
        )
        @test bc2 == 10.0  # skip because iteration=0

        # not on frequency
        bc3, _, _ = GIVPOptimizer.do_path_relinking!(
            3,
            10.0,
            [3.0],
            0,
            cfg2,
            ep,
            x -> sum(x .^ 2),
            1,
        )
        @test bc3 == 10.0  # skip because 3 % 5 != 0
    end

    @testset "do_path_relinking! executes" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        cfg = GIVPConfig(;
            use_elite_pool = true,
            path_relink_frequency = 5,
            vnd_iterations = 10,
            integer_split = 2,
        )
        ep = GIVPOptimizer.ElitePool(;
            max_size = 5,
            min_distance = 0.01,
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        GIVPOptimizer.add!(ep, [1.0, 1.0], 2.0)
        GIVPOptimizer.add!(ep, [5.0, 5.0], 50.0)

        bc, bs, st = GIVPOptimizer.do_path_relinking!(
            5,
            100.0,
            [7.0, 7.0],
            3,
            cfg,
            ep,
            sphere,
            2;
            cache = GIVPOptimizer.EvaluationCache(),
        )
        @test bc <= 100.0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "do_path_relinking! respects expired deadline" begin
        GIVPOptimizer.set_integer_split!(2)
        sphere(x) = sum(x .^ 2)
        cfg = GIVPConfig(;
            use_elite_pool = true,
            path_relink_frequency = 1,
            vnd_iterations = 4,
            integer_split = 2,
        )
        ep = GIVPOptimizer.ElitePool(;
            max_size = 5,
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        GIVPOptimizer.add!(ep, [1.0, 1.0], 2.0)
        GIVPOptimizer.add!(ep, [2.0, 2.0], 8.0)

        bc, bs, st = GIVPOptimizer.do_path_relinking!(
            1,
            5.0,
            [1.5, 1.5],
            2,
            cfg,
            ep,
            sphere,
            2;
            deadline = time() - 1.0,
        )
        @test bc == 5.0
        @test bs == [1.5, 1.5]
        @test st == 2
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "grasp_ils_vnd bounds validation" begin
        cfg = GIVPConfig(; max_iterations = 1)
        @test_throws InvalidBoundsError GIVPOptimizer.grasp_ils_vnd(x -> 0.0, 2, cfg)
        @test_throws InvalidBoundsError GIVPOptimizer.grasp_ils_vnd(
            x -> 0.0,
            2,
            cfg;
            lower = [0.0],
            upper = [1.0, 1.0],
        )
        @test_throws InvalidBoundsError GIVPOptimizer.grasp_ils_vnd(
            x -> 0.0,
            2,
            cfg;
            lower = [5.0, 5.0],
            upper = [1.0, 1.0],
        )
    end

    @testset "grasp_ils_vnd with all features" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)

        # With verbose, time_limit, callback, initial_guess
        callback_count = Ref(0)
        callback_fn(iter, cost, sol) = (callback_count[] += 1)

        cfg = GIVPConfig(;
            max_iterations = 5,
            vnd_iterations = 20,
            ils_iterations = 2,
            early_stop_threshold = 3,
            use_elite_pool = true,
            elite_size = 3,
            path_relink_frequency = 2,
            use_cache = true,
            cache_size = 100,
            use_convergence_monitor = true,
            integer_split = 2,
        )

        sol, cost, nit, msg = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg;
            verbose = true,
            iteration_callback = callback_fn,
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
            initial_guess = [5.0, 5.0],
        )
        @test isfinite(cost)
        @test callback_count[] > 0
        @test msg ∈ ("max iterations reached", "early stop triggered")
    end

    @testset "grasp_ils_vnd with time_limit" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        # Use high iteration counts + moderate VND to guarantee that time_limit
        # fires before natural termination (even with JIT warmup on Windows).
        # Disable convergence monitor and early stop so ONLY time_limit can fire.
        cfg = GIVPConfig(;
            max_iterations = 100_000,
            vnd_iterations = 200,
            ils_iterations = 5,
            time_limit = 2.0,
            integer_split = 2,
            use_convergence_monitor = false,
            early_stop_threshold = 100_000,
        )

        t0 = time()
        sol, cost, nit, msg = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        elapsed = time() - t0
        @test elapsed < 30.0  # generous upper bound for slow CI
        @test isfinite(cost)
        @test nit < cfg.max_iterations
        @test msg == "time limit reached"
    end

    @testset "grasp_ils_vnd with time_limit verbose" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        # Very short time limit + verbose to cover the TIME LIMIT log lines
        cfg = GIVPConfig(;
            max_iterations = 10000,
            vnd_iterations = 50,
            ils_iterations = 5,
            time_limit = 0.01,
            integer_split = 2,
        )
        sol, cost, _nit, _msg = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg;
            verbose = true,
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
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
                return sum(x .^ 2)  # restart evaluations get real cost
            end
        end
        GIVPOptimizer.set_seed!(42)
        # max_iterations=8 → stagnation threshold = 2
        cfg = GIVPConfig(;
            max_iterations = 8,
            vnd_iterations = 5,
            ils_iterations = 1,
            early_stop_threshold = 200,
            use_elite_pool = false,
            use_cache = false,
            use_convergence_monitor = false,
            integer_split = 2,
            perturbation_strength = 1,
            num_candidates_per_step = 3,
        )
        sol, cost, _nit, _msg = GIVPOptimizer.grasp_ils_vnd(
            flat_then_real,
            2,
            cfg;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test isfinite(cost)
        @test cost < 1000.0  # restart found better than flat
    end

    @testset "grasp_ils_vnd without optional components" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        cfg = GIVPConfig(;
            max_iterations = 3,
            vnd_iterations = 10,
            ils_iterations = 1,
            use_elite_pool = false,
            use_cache = false,
            use_convergence_monitor = false,
            integer_split = 2,
        )

        sol, cost, nit, msg = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test isfinite(cost)
        @test msg == "max iterations reached"
        @test nit == cfg.max_iterations
    end

    @testset "grasp_ils_vnd callback error handling" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        bad_callback(iter, cost, sol) = error("callback crash")

        cfg = GIVPConfig(;
            max_iterations = 2,
            vnd_iterations = 5,
            ils_iterations = 1,
            integer_split = 2,
        )
        # Should not throw despite callback error
        sol, cost, _nit, _msg = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
            iteration_callback = bad_callback,
        )
        @test isfinite(cost)
    end

    @testset "grasp_ils_vnd convergence restart keeps top-2 elite" begin
        GIVPOptimizer.set_seed!(42)
        flat_cost(x) = 1.0
        cfg = GIVPConfig(;
            max_iterations = 6,
            vnd_iterations = 3,
            ils_iterations = 1,
            early_stop_threshold = 1,
            use_convergence_monitor = true,
            use_elite_pool = true,
            elite_size = 6,
            use_cache = false,
            integer_split = 3,
        )

        sol, cost, nit, msg = GIVPOptimizer.grasp_ils_vnd(
            flat_cost,
            3,
            cfg;
            lower = [0.0, 0.0, 0.0],
            upper = [10.0, 10.0, 10.0],
            initial_guess = [9.0, 1.0, 3.0],
        )
        @test isfinite(cost)
        @test nit <= cfg.max_iterations
        @test msg in ("early stop triggered", "max iterations reached")
    end

    @testset "grasp_ils_vnd stagnation restart integer fallback clamp" begin
        GIVPOptimizer.set_seed!(123)
        flat_cost(x) = 1.0
        cfg = GIVPConfig(;
            max_iterations = 8,
            vnd_iterations = 3,
            ils_iterations = 1,
            use_convergence_monitor = false,
            use_elite_pool = false,
            use_cache = false,
            integer_split = 1,
        )

        sol, cost, nit, msg = GIVPOptimizer.grasp_ils_vnd(
            flat_cost,
            2,
            cfg;
            # Variable 2 is integer by split, but [0.2, 0.8] has no valid integer.
            lower = [0.0, 0.2],
            upper = [1.0, 0.8],
        )
        @test isfinite(cost)
        @test 0.2 <= sol[2] <= 0.8
        @test nit <= cfg.max_iterations
    end

    @testset "grasp_ils_vnd termination tracking" begin
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        # Completes all iterations: msg must be "max iterations reached"
        cfg_full = GIVPConfig(;
            max_iterations = 3,
            vnd_iterations = 5,
            ils_iterations = 1,
            use_convergence_monitor = false,
            use_elite_pool = false,
            use_cache = false,
            integer_split = 2,
        )
        sol_f, _, nit_f, msg_f = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg_full;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test msg_f == "max iterations reached"
        @test nit_f == cfg_full.max_iterations

        # Early stop: threshold=1 stops after first stagnation
        GIVPOptimizer.set_seed!(42)
        cfg_es = GIVPConfig(;
            max_iterations = 100,
            vnd_iterations = 5,
            ils_iterations = 1,
            early_stop_threshold = 1,
            use_convergence_monitor = true,
            use_elite_pool = false,
            use_cache = false,
            integer_split = 2,
        )
        sol_es, _, nit_es, msg_es = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg_es;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test isfinite(sol_es[1])
        @test nit_es < cfg_es.max_iterations
        @test msg_es == "early stop triggered"

        # Time limit: stops early with nit < max_iterations
        GIVPOptimizer.set_seed!(42)
        cfg_tl = GIVPConfig(;
            max_iterations = 100000,
            vnd_iterations = 50,
            ils_iterations = 5,
            time_limit = 0.01,
            integer_split = 2,
        )
        sol_tl, _, nit_tl, msg_tl = GIVPOptimizer.grasp_ils_vnd(
            sphere,
            2,
            cfg_tl;
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test isfinite(sol_tl[1])
        @test nit_tl < cfg_tl.max_iterations
        @test msg_tl == "time limit reached"
    end
end
