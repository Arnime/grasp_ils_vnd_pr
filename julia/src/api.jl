# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Public API for the GIVPOptimizer library."""

"""
    givp(func, bounds; kwargs...) -> OptimizeResult

Minimize (or maximize) a scalar function with GRASP-ILS-VND-PR.

# Arguments
- `func`: Objective callable mapping a `Vector{Float64}` to a scalar.
- `bounds`: Vector of `(low, high)` tuples or a `(lower, upper)` tuple.

# Keyword Arguments
- `num_vars::Union{Int,Nothing}=nothing`: Number of variables (inferred from bounds).
- `direction::Direction=minimize`: Optimization direction.
- `config::Union{GIVPConfig,Nothing}=nothing`: Algorithm hyper-parameters.
- `initial_guess::Union{Vector{Float64},Nothing}=nothing`: Warm-start vector.
- `iteration_callback::Union{Function,Nothing}=nothing`: Called per iteration.
- `seed::Union{Int,Nothing}=nothing`: RNG seed for reproducibility.
- `verbose::Bool=false`: Print progress.
"""
function givp(func::Function, bounds;
              num_vars::Union{Int,Nothing}=nothing,
              direction::Direction=minimize,
              config::Union{GIVPConfig,Nothing}=nothing,
              initial_guess::Union{Vector{Float64},Nothing}=nothing,
              iteration_callback::Union{Function,Nothing}=nothing,
              seed::Union{Int,Nothing}=nothing,
              verbose::Bool=false)::OptimizeResult

    cfg = config !== nothing ? config : GIVPConfig()
    validate_config!(cfg)
    cfg.direction = direction

    # Pin master RNG
    set_seed!(seed)

    # Normalize bounds
    lower, upper, n = _normalize_bounds(bounds, num_vars)

    # Default to fully continuous when integer_split not specified
    if cfg.integer_split === nothing
        cfg.integer_split = n
    end

    # Wrap objective for minimization
    nfev_counter = Ref(0)
    sign = direction == maximize ? -1.0 : 1.0
    function wrapped(x::Vector{Float64})::Float64
        nfev_counter[] += 1
        try
            value = Float64(func(x))
            isfinite(value) || return Inf
            return sign * value
        catch
            return Inf
        end
    end

    sol, core_value = grasp_ils_vnd(wrapped, n, cfg;
        verbose, iteration_callback, lower, upper, initial_guess)

    x = sol
    fun_value = sign * core_value
    success = isfinite(fun_value)

    return OptimizeResult(;
        x, fun=fun_value,
        nit=cfg.max_iterations,
        nfev=nfev_counter[],
        success,
        message=success ? "Optimization completed" : "No feasible solution found",
        direction,
        meta=Dict{String,Any}(),
    )
end

function _normalize_bounds(bounds::Vector{Tuple{Float64,Float64}},
                           num_vars::Union{Int,Nothing})
    lower = [b[1] for b in bounds]
    upper = [b[2] for b in bounds]
    n = length(lower)
    if num_vars !== nothing && n != num_vars
        throw(ArgumentError("bounds length ($n) does not match num_vars ($num_vars)"))
    end
    return lower, upper, n
end

function _normalize_bounds(bounds::Tuple{Vector{Float64},Vector{Float64}},
                           num_vars::Union{Int,Nothing})
    lower, upper = bounds
    n = length(lower)
    length(upper) != n && throw(ArgumentError("lower and upper must have same length"))
    if num_vars !== nothing && n != num_vars
        throw(ArgumentError("bounds length ($n) does not match num_vars ($num_vars)"))
    end
    return lower, upper, n
end
