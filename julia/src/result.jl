# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Result container returned by the public optimizer API."""

@enum TerminationReason begin
    converged
    max_iterations_reached
    time_limit_reached
    early_stop
    no_feasible
    unknown
end

function termination_from_message(message::String)::TerminationReason
    lower = lowercase(message)
    occursin("converge", lower) && return converged
    occursin("time", lower) && return time_limit_reached
    (occursin("early", lower) || occursin("threshold", lower)) && return early_stop
    (occursin("feasible", lower) || occursin("no solution", lower)) && return no_feasible
    (occursin("iteration", lower) || occursin("max", lower)) &&
        return max_iterations_reached
    return unknown
end

"""
    OptimizeResult

Container for optimization output, modeled after scipy.optimize.OptimizeResult.
"""
mutable struct OptimizeResult
    x::Vector{Float64}
    fun::Float64
    nit::Int
    nfev::Int
    success::Bool
    message::String
    direction::Direction
    meta::Dict{String, Any}
end

function OptimizeResult(;
    x::Vector{Float64} = Float64[],
    fun::Float64 = Inf,
    nit::Int = 0,
    nfev::Int = 0,
    success::Bool = true,
    message::String = "",
    direction::Direction = minimize,
    meta::Dict{String, Any} = Dict{String, Any}(),
)
    OptimizeResult(x, fun, nit, nfev, success, message, direction, meta)
end

function to_dict(r::OptimizeResult)::Dict{String, Any}
    Dict{String, Any}(
        "x" => r.x,
        "fun" => r.fun,
        "nit" => r.nit,
        "nfev" => r.nfev,
        "success" => r.success,
        "termination" => string(termination_from_message(r.message)),
        "direction" => string(r.direction),
    )
end

# Allow tuple unpacking: x, fun = result
Base.iterate(r::OptimizeResult, state = 1) =
    state == 1 ? (r.x, 2) : state == 2 ? (r.fun, 3) : nothing

function Base.length(::OptimizeResult)
    return 2
end
