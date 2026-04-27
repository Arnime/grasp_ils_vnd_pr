# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Diversity-aware elite pool of high-quality solutions."""

mutable struct ElitePool
    max_size::Int
    min_distance::Float64
    pool::Vector{Tuple{Vector{Float64},Float64}}
    _range::Union{Vector{Float64},Nothing}
end

function ElitePool(; max_size::Int=5, min_distance::Float64=0.05,
                   lower::Union{Vector{Float64},Nothing}=nothing,
                   upper::Union{Vector{Float64},Nothing}=nothing)
    rng = if lower !== nothing && upper !== nothing
        max.(upper .- lower, 1e-12)
    else
        nothing
    end
    ElitePool(max_size, min_distance, Tuple{Vector{Float64},Float64}[], rng)
end

function _relative_distance(ep::ElitePool, a::Vector{Float64}, b::Vector{Float64})::Float64
    if ep._range !== nothing
        return mean(abs.(a .- b) ./ ep._range)
    end
    return norm(a .- b)
end

function add!(ep::ElitePool, solution::Vector{Float64}, benefit::Float64)::Bool
    sol = copy(solution)

    for (elite_sol, _) in ep.pool
        distance = _relative_distance(ep, sol, elite_sol)
        distance < ep.min_distance && return false
    end

    if length(ep.pool) < ep.max_size
        push!(ep.pool, (sol, benefit))
        sort!(ep.pool; by=x -> x[2])
        return true
    end

    if benefit < ep.pool[end][2]
        ep.pool[end] = (sol, benefit)
        sort!(ep.pool; by=x -> x[2])
        return true
    end

    return false
end

function get_best(ep::ElitePool)::Tuple{Vector{Float64},Float64}
    isempty(ep.pool) && throw(EmptyPoolError("elite pool is empty; cannot return best solution"))
    return ep.pool[1]
end

function get_all(ep::ElitePool)::Vector{Tuple{Vector{Float64},Float64}}
    return copy(ep.pool)
end

function pool_size(ep::ElitePool)::Int
    return length(ep.pool)
end

function pool_clear!(ep::ElitePool)
    empty!(ep.pool)
end
