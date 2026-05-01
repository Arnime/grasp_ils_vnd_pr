# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Test suite hub — includes one file per module/concern.
# Run with:  julia --project=julia -e 'using Pkg; Pkg.test()'

using Test
using GIVPOptimizer
using Random
using LinearAlgebra
using Aqua
using JET

@testset "GIVPOptimizer.jl" begin
    include("test_exceptions.jl")
    include("test_config.jl")
    include("test_result.jl")
    include("test_helpers.jl")
    include("test_cache.jl")
    include("test_elite.jl")
    include("test_convergence.jl")
    include("test_grasp.jl")
    include("test_vnd.jl")
    include("test_ils.jl")
    include("test_pr.jl")
    include("test_impl.jl")
    include("test_api.jl")
    include("test_properties.jl")
    include("test_static_analysis.jl")
end
