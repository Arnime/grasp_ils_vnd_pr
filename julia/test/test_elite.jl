# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "ElitePool" begin
    @testset "basic operations" begin
        ep = GIVPOptimizer.ElitePool(;
            max_size = 3,
            min_distance = 0.01,
            lower = [0.0, 0.0],
            upper = [10.0, 10.0],
        )
        @test GIVPOptimizer.pool_size(ep) == 0
        @test_throws EmptyPoolError GIVPOptimizer.get_best(ep)

        @test GIVPOptimizer.add!(ep, [1.0, 1.0], 5.0)
        @test GIVPOptimizer.add!(ep, [5.0, 5.0], 3.0)
        @test GIVPOptimizer.pool_size(ep) == 2

        best_sol, best_cost = GIVPOptimizer.get_best(ep)
        @test best_cost == 3.0
        @test best_sol == [5.0, 5.0]

        all_sols = GIVPOptimizer.get_all(ep)
        @test length(all_sols) == 2
        @test all_sols[1][2] <= all_sols[2][2]  # sorted by cost
    end

    @testset "min_distance rejection" begin
        ep = GIVPOptimizer.ElitePool(;
            max_size = 5,
            min_distance = 0.5,
            lower = [0.0],
            upper = [10.0],
        )
        @test GIVPOptimizer.add!(ep, [5.0], 1.0)
        # Too close — should be rejected
        @test !GIVPOptimizer.add!(ep, [5.01], 0.5)
    end

    @testset "replace worst when full" begin
        ep = GIVPOptimizer.ElitePool(;
            max_size = 2,
            min_distance = 0.01,
            lower = [0.0],
            upper = [100.0],
        )
        @test GIVPOptimizer.add!(ep, [10.0], 10.0)
        @test GIVPOptimizer.add!(ep, [50.0], 20.0)
        @test GIVPOptimizer.pool_size(ep) == 2

        # Better than worst → replaces it
        @test GIVPOptimizer.add!(ep, [90.0], 15.0)
        @test GIVPOptimizer.pool_size(ep) == 2
        best_sol, best_cost = GIVPOptimizer.get_best(ep)
        @test best_cost == 10.0
    end

    @testset "reject when full and worse" begin
        ep = GIVPOptimizer.ElitePool(;
            max_size = 2,
            min_distance = 0.01,
            lower = [0.0],
            upper = [100.0],
        )
        GIVPOptimizer.add!(ep, [10.0], 5.0)
        GIVPOptimizer.add!(ep, [50.0], 10.0)
        # Worse than worst → rejected
        @test !GIVPOptimizer.add!(ep, [90.0], 100.0)
    end

    @testset "without bounds (norm distance)" begin
        ep = GIVPOptimizer.ElitePool(; max_size = 5, min_distance = 0.01)
        @test ep._range === nothing
        @test GIVPOptimizer.add!(ep, [1.0, 2.0], 5.0)
        @test GIVPOptimizer.add!(ep, [100.0, 200.0], 3.0)
    end

    @testset "pool_clear!" begin
        ep = GIVPOptimizer.ElitePool(; max_size = 3)
        GIVPOptimizer.add!(ep, [1.0], 1.0)
        GIVPOptimizer.pool_clear!(ep)
        @test GIVPOptimizer.pool_size(ep) == 0
    end
end
