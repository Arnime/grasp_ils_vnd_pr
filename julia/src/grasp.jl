# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""GRASP — Greedy Randomized Adaptive Search Procedure."""

"""
    validate_bounds_and_initial!(lower, upper, initial_guess, num_vars)

Validate that `lower` and `upper` have length `num_vars`, that `upper[i] > lower[i]`
for all variables, and that `initial_guess` (if provided) lies strictly within bounds.

Throws [`InvalidBoundsError`](@ref) or [`InvalidInitialGuessError`](@ref) on failure.
"""
function validate_bounds_and_initial!(
    lower::Vector{Float64},
    upper::Vector{Float64},
    initial_guess::Union{Vector{Float64}, Nothing},
    num_vars::Int,
)
    length(lower) != num_vars && throw(
        InvalidBoundsError(
            "lower (len=$(length(lower))) must have length num_vars=$num_vars",
        ),
    )
    length(upper) != num_vars && throw(
        InvalidBoundsError(
            "upper (len=$(length(upper))) must have length num_vars=$num_vars",
        ),
    )

    if initial_guess !== nothing
        length(initial_guess) != num_vars && throw(
            InvalidInitialGuessError(
                "initial_guess has length $(length(initial_guess)), expected $num_vars",
            ),
        )
        for i in 1:num_vars
            if initial_guess[i] <= lower[i] || initial_guess[i] >= upper[i]
                throw(
                    InvalidInitialGuessError(
                        "initial_guess values must be strictly between lower and upper; violating index: $i",
                    ),
                )
            end
        end
    end
end

function get_current_alpha(iter_idx::Int, config::GIVPConfig)::Float64
    if !config.adaptive_alpha
        return config.alpha
    end
    progress = iter_idx / max(1, config.max_iterations - 1)
    return config.alpha_min + (config.alpha_max - config.alpha_min) * progress
end

function evaluate_with_cache(
    cand::Vector{Float64},
    evaluator::Function,
    cache::Union{EvaluationCache, Nothing},
)::Float64
    if cache !== nothing
        cached_cost = cache_get(cache, cand)
        cached_cost !== nothing && return cached_cost
    end
    cost = safe_evaluate(evaluator, cand)
    if cache !== nothing && isfinite(cost)
        cache_put!(cache, cand, cost)
    end
    return cost
end

function select_from_rcl(
    costs::Vector{Float64},
    alpha::Float64,
    rng::AbstractRNG,
)::Union{Int, Nothing}
    valid_mask = isfinite.(costs)
    !any(valid_mask) && return nothing
    valid_idx = findall(valid_mask)
    valid_costs = costs[valid_idx]
    min_cost = minimum(valid_costs)
    max_cost = maximum(valid_costs)
    threshold = min_cost + alpha * (max_cost - min_cost)
    rcl_local = valid_idx[valid_costs .<= threshold]
    isempty(rcl_local) && (rcl_local = valid_idx)
    return rcl_local[rand(rng, 1:length(rcl_local))]
end

function normalize_integer_tail!(sol::Vector{Float64}, half::Int)
    for idx in (half + 1):length(sol)
        sol[idx] = Float64(round(Int, sol[idx]))
    end
end

function sample_integer_from_bounds(
    lower::Float64,
    upper::Float64,
    rng::AbstractRNG,
)::Float64
    lo = ceil(Int, lower)
    hi = floor(Int, upper)
    hi >= lo && return Float64(rand(rng, lo:hi))
    return Float64(round(Int, (lower + upper) / 2.0))
end

function build_random_candidate(
    num_vars::Int,
    half::Int,
    lower::Vector{Float64},
    upper::Vector{Float64},
    rng::AbstractRNG,
)::Vector{Float64}
    sol = Vector{Float64}(undef, num_vars)
    for i in 1:half
        sol[i] = lower[i] + (upper[i] - lower[i]) * rand(rng)
    end
    for i in (half + 1):num_vars
        sol[i] = sample_integer_from_bounds(lower[i], upper[i], rng)
    end
    return sol
end

function build_heuristic_candidate(
    num_vars::Int,
    half::Int,
    lower::Vector{Float64},
    upper::Vector{Float64},
    rng::AbstractRNG,
)::Vector{Float64}
    sol = Vector{Float64}(undef, num_vars)
    mid = (lower[1:half] .+ upper[1:half]) ./ 2.0
    span = upper[1:half] .- lower[1:half]
    noise = rand(rng, half) .* 0.3 .- 0.15  # uniform(-0.15, 0.15)
    sol[1:half] = clamp.(mid .+ noise .* span, lower[1:half], upper[1:half])

    for idx in (half + 1):num_vars
        lo = ceil(Int, lower[idx])
        hi = floor(Int, upper[idx])
        cont_idx = idx - half
        if hi > lo && cont_idx <= half && span[cont_idx] > 0
            frac = (sol[cont_idx] - lower[cont_idx]) / span[cont_idx]
            target = lo + frac * (hi - lo)
            sol[idx] = Float64(clamp(round(Int, target), lo, hi))
        else
            sol[idx] =
                hi >= lo ? Float64(hi) :
                Float64(round(Int, (lower[idx] + upper[idx]) / 2.0))
        end
    end
    return sol
end

function construct_grasp(
    num_vars::Int,
    lower::Vector{Float64},
    upper::Vector{Float64},
    evaluator::Function,
    initial_guess::Union{Vector{Float64}, Nothing},
    alpha::Float64;
    seed::Union{Int, Nothing} = nothing,
    num_candidates_per_step::Int = 20,
    cache::Union{EvaluationCache, Nothing} = nothing,
    n_workers::Int = 1,
)::Vector{Float64}
    rng = new_rng(seed)
    validate_bounds_and_initial!(lower, upper, initial_guess, num_vars)

    half = get_half(num_vars)
    n_candidates = max(num_candidates_per_step, 5)
    candidates = Vector{Float64}[]

    # Seed from initial guess
    if initial_guess !== nothing
        seed_candidate = copy(initial_guess)
        seed_cost = safe_evaluate(evaluator, seed_candidate)
        if !isfinite(seed_cost)
            seed_candidate = lower .+ (upper .- lower) .* rand(rng, num_vars)
        end
        normalize_integer_tail!(seed_candidate, half)
        push!(candidates, seed_candidate)
    end

    # Heuristic candidate
    heur = build_heuristic_candidate(num_vars, half, lower, upper, rng)
    normalize_integer_tail!(heur, half)
    push!(candidates, heur)

    # Random candidates
    while length(candidates) < n_candidates
        sol = build_random_candidate(num_vars, half, lower, upper, rng)
        normalize_integer_tail!(sol, half)
        push!(candidates, sol)
    end

    # Evaluate: sequential with cache (n_workers=1) or parallel without cache
    costs = Vector{Float64}(undef, length(candidates))
    if n_workers > 1
        # Parallel evaluation: bypass cache (Dict is not thread-safe in Julia)
        Threads.@threads for i in eachindex(candidates)
            costs[i] = safe_evaluate(evaluator, candidates[i])
        end
    else
        for i in eachindex(candidates)
            costs[i] = evaluate_with_cache(candidates[i], evaluator, cache)
        end
    end

    # Select from RCL
    best_idx = select_from_rcl(costs, alpha, rng)
    if best_idx === nothing
        best_idx = argmin(costs)
    end
    return copy(candidates[best_idx])
end
