# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "VND" begin
    @testset "create_cached_cost_fn" begin
        GIVPOptimizer.set_integer_split!(2)
        cache = GIVPOptimizer.EvaluationCache()
        fn = GIVPOptimizer.create_cached_cost_fn(x -> sum(x .^ 2), cache)
        @test fn([1.0, 2.0]) == 5.0
        @test fn([1.0, 2.0]) == 5.0  # cache hit
        @test cache.hits == 1

        # without cache
        fn2 = GIVPOptimizer.create_cached_cost_fn(x -> sum(x .^ 2), nothing)
        @test fn2([1.0, 2.0]) == 5.0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "try_integer_moves" begin
        GIVPOptimizer.set_integer_split!(0)  # all integer
        cost_fn(x) = sum(x .^ 2)

        # with bounds — should improve
        sol = [5.0, 5.0]
        new_sol, new_cost, improved = GIVPOptimizer.try_integer_moves(
            1,
            copy(sol),
            cost_fn(sol),
            cost_fn,
            [0.0, 0.0],
            [10.0, 10.0],
        )
        @test improved
        @test new_cost < 50.0

        # with no bounds
        sol2 = [5.0]
        new_sol2, new_cost2, imp2 = GIVPOptimizer.try_integer_moves(
            1,
            copy(sol2),
            cost_fn(sol2),
            cost_fn,
            nothing,
            nothing,
        )
        @test imp2

        # no improvement possible — already at optimal for that index
        sol3 = [0.0, 0.0]
        new_sol3, new_cost3, imp3 = GIVPOptimizer.try_integer_moves(
            1,
            copy(sol3),
            cost_fn(sol3),
            cost_fn,
            [0.0, 0.0],
            [0.0, 0.0],
        )
        @test !imp3
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "try_continuous_move" begin
        GIVPOptimizer.set_integer_split!(2)
        rng = MersenneTwister(42)
        sol = [3.0, 3.0]
        cost_fn(x) = sum(x .^ 2)
        # Try multiple times to get an improvement
        improved = false
        for _ in 1:100
            changed, _ = GIVPOptimizer.try_continuous_move(
                1,
                sol,
                cost_fn(sol),
                cost_fn,
                rng,
                [0.0, 0.0],
                [10.0, 10.0],
            )
            if changed
                improved = true
                break
            end
        end
        @test improved

        # without bounds
        rng2 = MersenneTwister(42)
        sol2 = [3.0]
        GIVPOptimizer.try_continuous_move(
            1,
            sol2,
            cost_fn(sol2),
            cost_fn,
            rng2,
            nothing,
            nothing,
        )
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "perturb_index! continuous" begin
        GIVPOptimizer.set_integer_split!(2)
        rng = MersenneTwister(42)
        sol = [5.0, 5.0, 3.0, 3.0]
        # Continuous index (1)
        GIVPOptimizer.perturb_index!(
            sol,
            1,
            4,
            rng,
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 10.0, 5.0, 5.0],
        )
        @test sol[1] != 5.0
        @test 0.0 <= sol[1] <= 10.0

        # Without bounds
        sol2 = [5.0, 5.0]
        GIVPOptimizer.perturb_index!(sol2, 1, 4, rng, nothing, nothing)
        @test sol2[1] != 5.0
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "perturb_index! integer" begin
        GIVPOptimizer.set_integer_split!(2)
        rng = MersenneTwister(42)
        sol = [5.0, 5.0, 3.0, 3.0]
        # Integer index (3 > half=2)
        GIVPOptimizer.perturb_index!(
            sol,
            3,
            4,
            rng,
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 10.0, 5.0, 5.0],
        )
        @test sol[3] == round(sol[3])
        @test 1.0 <= sol[3] <= 5.0

        # Without bounds
        sol2 = [5.0, 5.0, 3.0, 3.0]
        GIVPOptimizer.perturb_index!(sol2, 3, 4, rng, nothing, nothing)
        @test sol2[3] == round(sol2[3])
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "neighborhood_flip" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 2.0, 2.0]
        lower = [-5.0, -5.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        new_sol, new_cost = GIVPOptimizer.neighborhood_flip(
            sphere,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
            seed = 42,
        )
        @test new_cost <= sphere(sol)

        # with sensitivity
        sens = [10.0, 0.1, 5.0, 0.5]
        new_sol2, new_cost2 = GIVPOptimizer.neighborhood_flip(
            sphere,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
            sensitivity = sens,
            seed = 42,
        )
        @test new_cost2 <= sphere(sol)

        # best improvement (first_improvement=false)
        new_sol3, new_cost3 = GIVPOptimizer.neighborhood_flip(
            sphere,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
            first_improvement = false,
            seed = 42,
        )
        @test new_cost3 <= sphere(sol)
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "neighborhood_swap" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 2.0, 2.0]
        lower = [-5.0, -5.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        new_sol, new_cost = GIVPOptimizer.neighborhood_swap(
            sphere,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
        )
        @test new_cost <= sphere(sol)

        # no improvement scenario (covers final return at end of loop)
        flat_fn(x) = 0.0  # always returns 0, no swap can improve
        ns_sol, ns_cost = GIVPOptimizer.neighborhood_swap(
            flat_fn,
            [1.0, 1.0, 1.0, 1.0],
            0.0,
            4;
            lower = [-5.0, -5.0, 0.0, 0.0],
            upper = [5.0, 5.0, 5.0, 5.0],
            max_attempts = 5,
            first_improvement = false,
        )
        @test ns_cost == 0.0

        # without bounds
        new_sol2, _ = GIVPOptimizer.neighborhood_swap(sphere, sol, sphere(sol), 4)
        @test length(new_sol2) == 4

        # skip when half==0 or half==num_vars
        GIVPOptimizer.set_integer_split!(0)
        new_sol3, nc3 = GIVPOptimizer.neighborhood_swap(sphere, sol, sphere(sol), 4)
        @test nc3 == sphere(sol)  # no change

        GIVPOptimizer.set_integer_split!(4)
        new_sol4, nc4 = GIVPOptimizer.neighborhood_swap(sphere, sol, sphere(sol), 4)
        @test nc4 == sphere(sol)  # no change

        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "neighborhood_multiflip" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 2.0, 2.0]
        lower = [-5.0, -5.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        new_sol, new_cost = GIVPOptimizer.neighborhood_multiflip(
            sphere,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
            seed = 42,
        )
        @test length(new_sol) == 4

        # without bounds
        new_sol2, _ =
            GIVPOptimizer.neighborhood_multiflip(sphere, sol, sphere(sol), 4; seed = 42)
        @test length(new_sol2) == 4
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "try_neighborhoods" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 2.0, 2.0]
        lower = [-5.0, -5.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        cached_fn = GIVPOptimizer.create_cached_cost_fn(sphere, nothing)

        new_sol, new_cost, improved = GIVPOptimizer.try_neighborhoods(
            cached_fn,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
        )
        @test new_cost <= sphere(sol)

        # trigger multiflip (iteration divisible by no_improve_flip_limit)
        new_sol2, new_cost2, _ = GIVPOptimizer.try_neighborhoods(
            cached_fn,
            sol,
            sphere(sol),
            4;
            lower = lower,
            upper = upper,
            iteration = 3,
            no_improve_flip_limit = 3,
        )
        @test length(new_sol2) == 4

        # expired deadline
        new_sol3, new_cost3, improved3 = GIVPOptimizer.try_neighborhoods(
            cached_fn,
            sol,
            sphere(sol),
            4;
            deadline = time() - 1.0,
        )
        @test !improved3
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "local_search_vnd" begin
        GIVPOptimizer.set_integer_split!(4)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 3.0, 3.0]
        lower = [-5.0, -5.0, -5.0, -5.0]
        upper = [5.0, 5.0, 5.0, 5.0]

        improved = GIVPOptimizer.local_search_vnd(
            sphere,
            sol,
            4;
            max_iter = 50,
            lower = lower,
            upper = upper,
        )
        @test sphere(improved) <= sphere(sol)

        # with cache
        cache = GIVPOptimizer.EvaluationCache()
        improved2 = GIVPOptimizer.local_search_vnd(
            sphere,
            sol,
            4;
            max_iter = 50,
            lower = lower,
            upper = upper,
            cache = cache,
        )
        @test sphere(improved2) <= sphere(sol)

        # with deadline (expired)
        improved3 = GIVPOptimizer.local_search_vnd(sphere, sol, 4; deadline = time() - 1.0)
        @test improved3 == sol
        GIVPOptimizer.set_integer_split!(nothing)
    end
end
