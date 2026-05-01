# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "OptimizeResult" begin
    @testset "construction and fields" begin
        r = OptimizeResult(;
            x = [1.0, 2.0],
            fun = 3.0,
            nit = 10,
            nfev = 100,
            success = true,
            message = "converged",
            direction = minimize,
        )
        @test r.x == [1.0, 2.0]
        @test r.fun == 3.0
        @test r.nit == 10
        @test r.nfev == 100
        @test r.success == true
        @test r.message == "converged"
        @test r.direction == minimize
        @test r.meta == Dict{String, Any}()
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
        r = OptimizeResult(;
            x = [1.0],
            fun = 2.0,
            nit = 5,
            nfev = 50,
            success = false,
            message = "no feasible",
            direction = maximize,
        )
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
        r = OptimizeResult(; x = [1.0, 2.0], fun = 3.0)
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
    @test GIVPOptimizer.termination_from_message("converged successfully") ==
          GIVPOptimizer.converged
    @test GIVPOptimizer.termination_from_message("time limit reached") ==
          GIVPOptimizer.time_limit_reached
    @test GIVPOptimizer.termination_from_message("early stop triggered") ==
          GIVPOptimizer.early_stop
    @test GIVPOptimizer.termination_from_message("threshold exceeded") ==
          GIVPOptimizer.early_stop
    @test GIVPOptimizer.termination_from_message("no feasible solution") ==
          GIVPOptimizer.no_feasible
    @test GIVPOptimizer.termination_from_message("no solution found") ==
          GIVPOptimizer.no_feasible
    @test GIVPOptimizer.termination_from_message("max iterations reached") ==
          GIVPOptimizer.max_iterations_reached
    @test GIVPOptimizer.termination_from_message("iteration limit") ==
          GIVPOptimizer.max_iterations_reached
    @test GIVPOptimizer.termination_from_message("something else entirely") ==
          GIVPOptimizer.unknown
    @test GIVPOptimizer.termination_from_message("") == GIVPOptimizer.unknown
end
