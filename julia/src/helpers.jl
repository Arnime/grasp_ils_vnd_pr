# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Pure-utility helpers shared across the GIVP core submodules."""

# Type alias for the user-supplied objective function
const EvaluatorFn = Function

# --- Per-task configuration (task-local storage) ---

const _INTEGER_SPLIT = Ref{Union{Int,Nothing}}(nothing)
const _GROUP_SIZE = Ref{Union{Int,Nothing}}(nothing)
const _MASTER_RNG = Ref{Union{MersenneTwister,Nothing}}(nothing)

function set_seed!(seed::Union{Int,Nothing})
    if seed === nothing
        _MASTER_RNG[] = nothing
    else
        _MASTER_RNG[] = MersenneTwister(seed)
    end
end

function get_half(n::Int)::Int
    split = _INTEGER_SPLIT[]
    if split !== nothing && 0 <= split <= n
        return split
    end
    return n ÷ 2
end

function set_integer_split!(split::Union{Int,Nothing})
    _INTEGER_SPLIT[] = split
end

function set_group_size!(size::Union{Int,Nothing})
    _GROUP_SIZE[] = size
end

function get_group_size()::Union{Int,Nothing}
    return _GROUP_SIZE[]
end

function new_rng(seed::Union{Int,Nothing}=nothing)::AbstractRNG
    if seed !== nothing
        return MersenneTwister(seed)
    end
    master = _MASTER_RNG[]
    if master !== nothing
        child_seed = rand(master, UInt64)
        return MersenneTwister(child_seed)
    end
    return MersenneTwister()
end

function expired(deadline::Float64)::Bool
    return deadline > 0 && time() >= deadline
end

function safe_evaluate(evaluator::Function, candidate::Vector{Float64})::Float64
    try
        cost = Float64(evaluator(candidate))
        isfinite(cost) || return Inf
        return cost
    catch e
        @warn "evaluator raised an exception; treating candidate as infeasible" exception=(e, catch_backtrace())
        return Inf
    end
end
