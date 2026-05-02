# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""
    helpers.jl — Pure-utility helpers shared across GIVPOptimizer core submodules.

Provides RNG management, per-task configuration, deadline checking, and safe
objective-function evaluation.

All mutable state is stored in Julia's `task_local_storage()` so that concurrent
`givp()` calls from different tasks do not interfere with each other (analogous
to Python's `contextvars.ContextVar`).
"""

# Type alias for the user-supplied objective function
const EvaluatorFn = Function

# ── Task-local storage keys ───────────────────────────────────────────────────
# Using task_local_storage provides isolation: each Task (green thread) gets its
# own copy, preventing data races in concurrent optimization runs.

const _TLS_INTEGER_SPLIT = :givp_integer_split
const _TLS_GROUP_SIZE = :givp_group_size
const _TLS_MASTER_RNG = :givp_master_rng

# ── Seed management ──────────────────────────────────────────────────────────

"""
    set_seed!(seed::Union{Int,Nothing})

Pin the master RNG seed for the current task.  Subsequent calls to [`new_rng`](@ref)
(without an explicit seed) will derive deterministic child RNGs from this master.

Pass `nothing` to restore non-deterministic behaviour.

This function is task-local: concurrent tasks do not interfere with each other.
"""
function set_seed!(seed::Union{Int, Nothing})
    if seed === nothing
        task_local_storage(_TLS_MASTER_RNG, nothing)
    else
        task_local_storage(_TLS_MASTER_RNG, MersenneTwister(seed))
    end
end

# ── Integer split ─────────────────────────────────────────────────────────────

"""
    get_half(n::Int) -> Int

Return the index where integer variables begin.  If an explicit split was set via
[`set_integer_split!`](@ref), that value is used (clamped to `[0, n]`).
Otherwise defaults to `n ÷ 2`.
"""
function get_half(n::Int)::Int
    split = get(task_local_storage(), _TLS_INTEGER_SPLIT, nothing)
    if split !== nothing && 0 <= split <= n
        return split
    end
    return n ÷ 2
end

"""
    set_integer_split!(split::Union{Int,Nothing})

Set the task-local integer split used by [`get_half`](@ref).
Pass `nothing` to reset to the default `n ÷ 2` behaviour.
"""
function set_integer_split!(split::Union{Int, Nothing})
    task_local_storage(_TLS_INTEGER_SPLIT, split)
end

# ── Group size ────────────────────────────────────────────────────────────────

"""
    set_group_size!(size::Union{Int,Nothing})

Set the number of steps per group for the group/block neighbourhoods (task-local).
"""
function set_group_size!(size::Union{Int, Nothing})
    task_local_storage(_TLS_GROUP_SIZE, size)
end

"""
    get_group_size() -> Union{Int,Nothing}

Return the configured group size for the current task, or `nothing` if unset.
"""
function get_group_size()::Union{Int, Nothing}
    return get(task_local_storage(), _TLS_GROUP_SIZE, nothing)
end

# ── RNG factory ───────────────────────────────────────────────────────────────

"""
    new_rng(seed::Union{Int,Nothing}=nothing) -> AbstractRNG

Create an independent `MersenneTwister` RNG.

- If `seed` is given explicitly, seed the RNG with that value.
- If a master seed was previously pinned (via [`set_seed!`](@ref)),
  derive a child seed from it (guaranteeing reproducibility).
- Otherwise, use entropy from the system (non-deterministic).
"""
function new_rng(seed::Union{Int, Nothing} = nothing)::AbstractRNG
    if seed !== nothing
        return MersenneTwister(seed)
    end
    master = get(task_local_storage(), _TLS_MASTER_RNG, nothing)
    if master !== nothing
        child_seed = rand(master, UInt64)
        return MersenneTwister(child_seed)
    end
    return MersenneTwister()
end

# ── Time management ───────────────────────────────────────────────────────────

"""
    expired(deadline::Float64) -> Bool

Return `true` if the wall-clock deadline has passed.  A deadline of `0.0` means
"no limit" and always returns `false`.
"""
function expired(deadline::Float64)::Bool
    return deadline > 0 && time() >= deadline
end

# ── Safe evaluation ───────────────────────────────────────────────────────────

"""
    safe_evaluate(evaluator::Function, candidate::Vector{Float64}) -> Float64

Call `evaluator(candidate)` and return its value.  If the evaluator throws an
exception or returns a non-finite value, return `Inf` (treating the candidate
as infeasible).  This ensures the optimizer never propagates NaN/errors into
its internal state.
"""
function safe_evaluate(evaluator::Function, candidate::Vector{Float64})::Float64
    try
        cost = Float64(evaluator(candidate))
        isfinite(cost) || return Inf
        return cost
    catch e
        @warn "evaluator raised an exception; treating candidate as infeasible" exception =
            (e, catch_backtrace())
        return Inf
    end
end
