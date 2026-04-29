# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""
GIVPOptimizer — GRASP-ILS-VND with Path Relinking optimizer (Julia port).

Public API:
    givp(func, bounds; direction=:minimize, ...) -> OptimizeResult
    GIVPConfig          (algorithm hyper-parameters)
    OptimizeResult      (result struct)
"""
module GIVPOptimizer

using Random
using Statistics
using LinearAlgebra
using Printf
using Dates

export givp, GIVPConfig, OptimizeResult, TerminationReason, Direction, minimize, maximize
export GivpError,
    InvalidBoundsError,
    InvalidInitialGuessError,
    InvalidConfigError,
    EvaluatorError,
    EmptyPoolError
export validate_config!, to_dict

include("exceptions.jl")
include("config.jl")
include("result.jl")
include("helpers.jl")
include("cache.jl")
include("elite.jl")
include("convergence.jl")
include("grasp.jl")
include("vnd.jl")
include("ils.jl")
include("pr.jl")
include("impl.jl")
include("api.jl")

const __version__ = "0.8.0"

end # module
