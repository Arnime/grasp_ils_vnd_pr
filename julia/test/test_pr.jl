# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "Path Relinking" begin
    @testset "path_relinking forward" begin
        sphere(x) = sum(x .^ 2)
        source = [5.0, 5.0, 5.0]
        target = [0.1, 0.1, 0.1]
        best_sol, best_cost = GIVPOptimizer.path_relinking(
            sphere,
            source,
            target;
            strategy = :forward,
            seed = 42,
        )
        @test best_cost <= sphere(source)
    end

    @testset "path_relinking best" begin
        sphere(x) = sum(x .^ 2)
        source = [5.0, 5.0, 5.0]
        target = [0.1, 0.1, 0.1]
        best_sol, best_cost = GIVPOptimizer.path_relinking(
            sphere,
            source,
            target;
            strategy = :best,
            seed = 42,
        )
        @test best_cost <= sphere(source)
    end

    @testset "path_relinking_best no improving move (break)" begin
        # cost function where moving towards target never improves
        opt_fn(x) = sum((x .- 1.0) .^ 2)  # optimum at [1,1,1]
        source = [1.0, 1.0, 1.0]
        target = [10.0, 10.0, 10.0]
        best_sol, best_cost = GIVPOptimizer.path_relinking(
            opt_fn,
            source,
            target;
            strategy = :best,
            seed = 42,
        )
        @test best_cost == opt_fn(source)  # can't improve
    end

    @testset "identical solutions" begin
        sphere(x) = sum(x .^ 2)
        source = [1.0, 2.0, 3.0]
        best_sol, best_cost =
            GIVPOptimizer.path_relinking(sphere, source, copy(source); seed = 42)
        @test best_cost == sphere(source)
        @test best_sol == source
    end

    @testset "many diff indices (> MAX_PR_VARS)" begin
        sphere(x) = sum(x .^ 2)
        n = 50
        source = fill(5.0, n)
        target = fill(0.1, n)
        best_sol, best_cost = GIVPOptimizer.path_relinking(
            sphere,
            source,
            target;
            strategy = :forward,
            seed = 42,
        )
        @test best_cost <= sphere(source)
    end

    @testset "bidirectional_path_relinking" begin
        sphere(x) = sum(x .^ 2)
        sol1 = [5.0, 5.0]
        sol2 = [0.1, 0.1]
        best_sol, best_cost = GIVPOptimizer.bidirectional_path_relinking(sphere, sol1, sol2)
        @test best_cost <= sphere(sol1)
    end

    @testset "with expired deadline" begin
        sphere(x) = sum(x .^ 2)
        source = [5.0, 5.0]
        target = [0.1, 0.1]
        best_sol, best_cost =
            GIVPOptimizer.path_relinking(sphere, source, target; deadline = time() - 1.0)
        # Should still return something valid
        @test isfinite(best_cost)
    end
end
