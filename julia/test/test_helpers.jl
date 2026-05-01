# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "Helpers" begin
    @testset "set_seed! and new_rng determinism" begin
        GIVPOptimizer.set_seed!(42)
        rng1 = GIVPOptimizer.new_rng()
        GIVPOptimizer.set_seed!(42)
        rng2 = GIVPOptimizer.new_rng()
        @test rand(rng1) == rand(rng2)

        # explicit seed overrides master
        rng3 = GIVPOptimizer.new_rng(99)
        rng4 = GIVPOptimizer.new_rng(99)
        @test rand(rng3) == rand(rng4)

        # no seed, no master → non-deterministic (just verify it works)
        GIVPOptimizer.set_seed!(nothing)
        rng5 = GIVPOptimizer.new_rng()
        @test rng5 isa AbstractRNG
    end

    @testset "get_half / set_integer_split!" begin
        GIVPOptimizer.set_integer_split!(nothing)
        @test GIVPOptimizer.get_half(10) == 5
        @test GIVPOptimizer.get_half(7) == 3

        GIVPOptimizer.set_integer_split!(3)
        @test GIVPOptimizer.get_half(10) == 3
        @test GIVPOptimizer.get_half(3) == 3

        # split > n → falls back to n÷2
        GIVPOptimizer.set_integer_split!(20)
        @test GIVPOptimizer.get_half(10) == 5

        GIVPOptimizer.set_integer_split!(0)
        @test GIVPOptimizer.get_half(10) == 0

        GIVPOptimizer.set_integer_split!(nothing)
    end

    @testset "set_group_size! / get_group_size" begin
        GIVPOptimizer.set_group_size!(nothing)
        @test GIVPOptimizer.get_group_size() === nothing
        GIVPOptimizer.set_group_size!(24)
        @test GIVPOptimizer.get_group_size() == 24
        GIVPOptimizer.set_group_size!(nothing)
    end

    @testset "safe_evaluate" begin
        @test GIVPOptimizer.safe_evaluate(x -> sum(x .^ 2), [1.0, 2.0]) == 5.0
        @test GIVPOptimizer.safe_evaluate(x -> 0.0, [1.0]) == 0.0
        @test GIVPOptimizer.safe_evaluate(x -> error("fail"), [1.0]) == Inf
        @test GIVPOptimizer.safe_evaluate(x -> NaN, [1.0]) == Inf
        @test GIVPOptimizer.safe_evaluate(x -> Inf, [1.0]) == Inf
        @test GIVPOptimizer.safe_evaluate(x -> -Inf, [1.0]) == Inf
    end

    @testset "expired" begin
        @test !GIVPOptimizer.expired(0.0)  # 0 = no deadline
        @test !GIVPOptimizer.expired(time() + 1000)
        @test GIVPOptimizer.expired(time() - 1.0)
    end
end
