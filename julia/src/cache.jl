# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""LRU evaluation cache used by the GRASP/ILS/VND algorithm."""

mutable struct EvaluationCache
    maxsize::Int
    cache::Dict{UInt64,Float64}
    hits::Int
    misses::Int
    insertion_order::Vector{UInt64}
end

function EvaluationCache(; maxsize::Int=10000)
    EvaluationCache(maxsize, Dict{UInt64,Float64}(), 0, 0, UInt64[])
end

function _hash_solution(ec::EvaluationCache, solution::Vector{Float64})::UInt64
    half = get_half(length(solution))
    rounded = copy(solution)
    for i in 1:half
        rounded[i] = round(rounded[i]; digits=3)
    end
    for i in (half+1):length(rounded)
        rounded[i] = round(rounded[i]; digits=0)
    end
    return hash(rounded)
end

function cache_get(ec::EvaluationCache, solution::Vector{Float64})::Union{Float64,Nothing}
    key = _hash_solution(ec, solution)
    if haskey(ec.cache, key)
        ec.hits += 1
        return ec.cache[key]
    end
    ec.misses += 1
    return nothing
end

function cache_put!(ec::EvaluationCache, solution::Vector{Float64}, cost::Float64)
    key = _hash_solution(ec, solution)
    if !haskey(ec.cache, key) && length(ec.cache) >= ec.maxsize
        oldest = popfirst!(ec.insertion_order)
        delete!(ec.cache, oldest)
    end
    if !haskey(ec.cache, key)
        push!(ec.insertion_order, key)
    end
    ec.cache[key] = cost
end

function cache_clear!(ec::EvaluationCache)
    empty!(ec.cache)
    empty!(ec.insertion_order)
    ec.hits = 0
    ec.misses = 0
end

function cache_stats(ec::EvaluationCache)::Dict{String,Any}
    total = ec.hits + ec.misses
    hit_rate = total > 0 ? (ec.hits / total * 100) : 0.0
    Dict{String,Any}(
        "hits" => ec.hits,
        "misses" => ec.misses,
        "hit_rate" => hit_rate,
        "size" => length(ec.cache),
    )
end
