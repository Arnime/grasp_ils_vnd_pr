# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "ILS" begin
    @testset "perturb_solution" begin
        GIVPOptimizer.set_integer_split!(2)
        GIVPOptimizer.set_seed!(42)
        sol = [5.0, 5.0, 3.0, 3.0]
        lower = [0.0, 0.0, 1.0, 1.0]
        upper = [10.0, 10.0, 5.0, 5.0]

        perturbed = GIVPOptimizer.perturb_solution(
            sol,
            4;
            strength = 4,
            seed = 42,
            lower = lower,
            upper = upper,
        )
        @test length(perturbed) == 4
        @test perturbed != sol  # should be different
        @test all(lower .<= perturbed .<= upper)

        # without bounds
        perturbed2 = GIVPOptimizer.perturb_solution(sol, 4; strength = 4, seed = 42)
        @test length(perturbed2) == 4
        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "ils_search" begin
        GIVPOptimizer.set_integer_split!(4)
        GIVPOptimizer.set_seed!(42)
        sphere(x) = sum(x .^ 2)
        sol = [3.0, 3.0, 3.0, 3.0]
        lower = [-5.0, -5.0, -5.0, -5.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        config = GIVPConfig(;
            ils_iterations = 5,
            vnd_iterations = 20,
            perturbation_strength = 2,
            integer_split = 4,
        )

        best_sol, best_cost = GIVPOptimizer.ils_search(
            sol,
            sphere(sol),
            4,
            sphere,
            config;
            lower = lower,
            upper = upper,
        )
        @test best_cost <= sphere(sol)
        @test length(best_sol) == 4

        # with cache (large enough to always hit)
        cache = GIVPOptimizer.EvaluationCache()
        best_sol2, best_cost2 = GIVPOptimizer.ils_search(
            sol,
            sphere(sol),
            4,
            sphere,
            config;
            lower = lower,
            upper = upper,
            cache = cache,
        )
        @test best_cost2 <= sphere(sol)

        # with tiny cache (maxsize=1) to force cache misses after VND evicts entries
        tiny_cache = GIVPOptimizer.EvaluationCache(; maxsize = 1)
        best_sol5, best_cost5 = GIVPOptimizer.ils_search(
            sol,
            sphere(sol),
            4,
            sphere,
            config;
            lower = lower,
            upper = upper,
            cache = tiny_cache,
        )
        @test best_cost5 <= sphere(sol)

        # without cache (covers else branch: cost_fn(perturbed))
        best_sol4, best_cost4 = GIVPOptimizer.ils_search(
            sol,
            sphere(sol),
            4,
            sphere,
            config;
            lower = lower,
            upper = upper,
        )
        @test best_cost4 <= sphere(sol)

        # expired deadline
        best_sol3, best_cost3 = GIVPOptimizer.ils_search(
            sol,
            sphere(sol),
            4,
            sphere,
            config;
            deadline = time() - 1.0,
        )
        @test best_cost3 == sphere(sol)  # no improvement
        GIVPOptimizer.set_integer_split!(nothing)
    end
end
