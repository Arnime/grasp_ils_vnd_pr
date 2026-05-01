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
function givp(
    func::Function,
    bounds;
    num_vars::Union{Int, Nothing} = nothing,
    direction::Direction = minimize,
    config::Union{GIVPConfig, Nothing} = nothing,
    initial_guess::Union{Vector{Float64}, Nothing} = nothing,
    iteration_callback::Union{Function, Nothing} = nothing,
    seed::Union{Int, Nothing} = nothing,
    verbose::Bool = false,
)::OptimizeResult
    # Validate callable — must accept a Vector{Float64}
    applicable(func, Vector{Float64}(undef, 0)) ||
        !hasmethod(func, Tuple{Vector{Float64}}) &&
            @warn "func may not accept Vector{Float64}; ensure it is callable with a vector argument"

    cfg = config !== nothing ? config : GIVPConfig()
    validate_config!(cfg)
    cfg.direction = direction

    # Pin master RNG
    set_seed!(seed)

    # Normalize bounds
    lower, upper, n = _normalize_bounds(bounds, num_vars)
    n > 0 || throw(InvalidBoundsError("bounds must be non-empty (got 0 variables)"))

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
        catch e
            @warn "objective raised an exception; treating as infeasible" exception =
                (e, catch_backtrace())
            return Inf
        end
    end

    sol, core_value, actual_nit, term_msg, meta = grasp_ils_vnd(
        wrapped,
        n,
        cfg;
        verbose,
        iteration_callback,
        lower,
        upper,
        initial_guess,
    )

    x = sol
    fun_value = sign * core_value
    success = isfinite(fun_value)

    return OptimizeResult(;
        x,
        fun = fun_value,
        nit = actual_nit,
        nfev = nfev_counter[],
        success,
        message = isfinite(fun_value) ? term_msg : "no feasible solution found",
        direction,
        meta,
    )
end

"""
    GIVPOptimizerWrapper

Object-oriented wrapper around [`givp`](@ref). Mirrors the Python `GIVPOptimizer` class.

Holds configuration and bounds; exposes a [`run`](@ref) method that returns an
[`OptimizeResult`](@ref). Tracks the best solution across repeated `run()` calls
(useful for multi-start strategies).

# Fields
- `best_x`: Best solution vector found across all `run()` calls, or `nothing`.
- `best_fun`: Best objective value found across all `run()` calls.
- `history`: Vector of `OptimizeResult` from each `run()` call.

# Example
```julia
opt = GIVPOptimizerWrapper(sphere, [(-5.0, 5.0) for _ in 1:10]; seed=42)
result = optimize!(opt)
println(opt.best_fun)
```
"""
mutable struct GIVPOptimizerWrapper{B}
    func::Function
    bounds::B
    num_vars::Union{Int, Nothing}
    direction::Direction
    config::GIVPConfig
    initial_guess::Union{Vector{Float64}, Nothing}
    iteration_callback::Union{Function, Nothing}
    seed::Union{Int, Nothing}
    verbose::Bool
    best_x::Union{Vector{Float64}, Nothing}
    best_fun::Float64
    history::Vector{OptimizeResult}
end

function GIVPOptimizerWrapper(
    func::Function,
    bounds::B;
    num_vars::Union{Int, Nothing} = nothing,
    direction::Direction = minimize,
    config::Union{GIVPConfig, Nothing} = nothing,
    initial_guess::Union{Vector{Float64}, Nothing} = nothing,
    iteration_callback::Union{Function, Nothing} = nothing,
    seed::Union{Int, Nothing} = nothing,
    verbose::Bool = false,
) where {B}
    _best_fun = direction == maximize ? -Inf : Inf
    return GIVPOptimizerWrapper{B}(
        func,
        bounds,
        num_vars,
        direction,
        config !== nothing ? config : GIVPConfig(),
        initial_guess,
        iteration_callback,
        seed,
        verbose,
        nothing,
        _best_fun,
        OptimizeResult[],
    )
end

function _is_better(opt::GIVPOptimizerWrapper, candidate::Float64)::Bool
    return opt.direction == maximize ? candidate > opt.best_fun : candidate < opt.best_fun
end

"""
    optimize!(opt::GIVPOptimizerWrapper) -> OptimizeResult

Execute one optimization round, update `opt.history` and `opt.best_x`/`opt.best_fun`.
"""
function optimize!(opt::GIVPOptimizerWrapper)::OptimizeResult
    result = givp(
        opt.func,
        opt.bounds;
        num_vars = opt.num_vars,
        direction = opt.direction,
        config = opt.config,
        initial_guess = opt.initial_guess,
        iteration_callback = opt.iteration_callback,
        seed = opt.seed,
        verbose = opt.verbose,
    )
    push!(opt.history, result)
    if opt.best_x === nothing || _is_better(opt, result.fun)
        opt.best_x = result.x
        opt.best_fun = result.fun
    end
    return result
end

"""
    _normalize_bounds(bounds, num_vars) -> (lower, upper, n)

Convert various bounds representations into `(lower::Vector{Float64}, upper::Vector{Float64}, n::Int)`.

Accepted formats:
- `Vector{Tuple{Float64,Float64}}`: per-variable `(lo, hi)` pairs.
- `Tuple{Vector{Float64}, Vector{Float64}}`: `(lower_vec, upper_vec)`.
- `Vector{Vector{Float64}}`: `[[lo...], [hi...]]`.

Throws `ArgumentError` if `num_vars` is specified and does not match.
"""
function _normalize_bounds(
    bounds::Vector{Tuple{Float64, Float64}},
    num_vars::Union{Int, Nothing},
)
    lower = [b[1] for b in bounds]
    upper = [b[2] for b in bounds]
    n = length(lower)
    if num_vars !== nothing && n != num_vars
        throw(ArgumentError("bounds length ($n) does not match num_vars ($num_vars)"))
    end
    return lower, upper, n
end

function _normalize_bounds(
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    num_vars::Union{Int, Nothing},
)
    lower, upper = bounds
    n = length(lower)
    length(upper) != n && throw(ArgumentError("lower and upper must have same length"))
    if num_vars !== nothing && n != num_vars
        throw(ArgumentError("bounds length ($n) does not match num_vars ($num_vars)"))
    end
    return lower, upper, n
end

"""Accept `[[lower...], [upper...]]` — two-element vector of vectors."""
function _normalize_bounds(bounds::Vector{Vector{Float64}}, num_vars::Union{Int, Nothing})
    length(bounds) == 2 || throw(
        ArgumentError(
            "Vector{Vector} bounds must be [[lower…], [upper…]] (exactly 2 elements)",
        ),
    )
    lower, upper = bounds[1], bounds[2]
    n = length(lower)
    length(upper) != n && throw(ArgumentError("lower and upper must have same length"))
    if num_vars !== nothing && n != num_vars
        throw(ArgumentError("bounds length ($n) does not match num_vars ($num_vars)"))
    end
    return lower, upper, n
end
