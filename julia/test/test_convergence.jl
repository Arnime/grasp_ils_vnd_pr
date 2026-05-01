# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "ConvergenceMonitor" begin
    @testset "restart trigger" begin
        cm = GIVPOptimizer.ConvergenceMonitor(; restart_threshold = 5)
        ep = GIVPOptimizer.ElitePool(; max_size = 3)

        # First update improves best_ever from Inf
        status = GIVPOptimizer.update!(cm, 10.0, ep)
        @test !status["should_restart"]
        @test cm.no_improve_count == 0

        for _ in 1:4
            status = GIVPOptimizer.update!(cm, 10.0, ep)
        end
        @test cm.no_improve_count == 4
        @test !status["should_restart"]

        status = GIVPOptimizer.update!(cm, 10.0, ep)
        @test status["should_restart"]
        @test cm.no_improve_count == 5
    end

    @testset "improvement resets counter" begin
        cm = GIVPOptimizer.ConvergenceMonitor(; restart_threshold = 10)
        GIVPOptimizer.update!(cm, 10.0)
        GIVPOptimizer.update!(cm, 10.0)
        GIVPOptimizer.update!(cm, 10.0)
        @test cm.no_improve_count == 2
        GIVPOptimizer.update!(cm, 5.0)  # improvement
        @test cm.no_improve_count == 0
    end

    @testset "should_intensify" begin
        cm = GIVPOptimizer.ConvergenceMonitor(; restart_threshold = 10)
        ep = GIVPOptimizer.ElitePool(; max_size = 3)
        GIVPOptimizer.update!(cm, 10.0, ep)
        for _ in 1:5
            GIVPOptimizer.update!(cm, 10.0, ep)
        end
        status = GIVPOptimizer.update!(cm, 10.0, ep)
        # no_improve_count >= 5 (threshold//2) and diversity < 0.5
        @test status["should_intensify"]
    end

    @testset "diversity with elite pool" begin
        cm = GIVPOptimizer.ConvergenceMonitor(; restart_threshold = 100)
        ep = GIVPOptimizer.ElitePool(;
            max_size = 5,
            min_distance = 0.01,
            lower = [0.0, 0.0],
            upper = [100.0, 100.0],
        )
        GIVPOptimizer.add!(ep, [10.0, 10.0], 1.0)
        GIVPOptimizer.add!(ep, [90.0, 90.0], 2.0)
        status = GIVPOptimizer.update!(cm, 5.0, ep)
        @test status["diversity"] > 0.0
    end

    @testset "without elite pool" begin
        cm = GIVPOptimizer.ConvergenceMonitor()
        status = GIVPOptimizer.update!(cm, 10.0, nothing)
        @test status["diversity"] == 0.0
    end
end
