# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Static analysis suite — JET (type inference errors) + Aqua (package quality).
# Runs separately from functional tests so failures are clearly isolated.

using JET
using Aqua

@testset "Static analysis" begin
    @testset "Aqua package quality" begin
        Aqua.test_all(
            GIVPOptimizer;
            ambiguities = (broken = false,),
            # Stale deps check: JSON is used in CLI/benchmarks, JET in tests only
            stale_deps = (ignore = [:JSON, :JET, :JuliaFormatter],),
        )
    end

    @testset "JET type inference" begin
        # report_package raises on any type-inference error found in the module
        result = JET.report_package("GIVPOptimizer"; ignored_modules = (Base, Core))
        @test length(JET.get_reports(result)) == 0
    end
end
