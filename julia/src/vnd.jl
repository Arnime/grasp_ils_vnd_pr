# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""VND — Variable Neighborhood Descent."""

function create_cached_cost_fn(
    cost_fn::Function,
    cache::Union{EvaluationCache, Nothing},
)::Function
    function cached_cost_fn(sol::Vector{Float64})::Float64
        if cache !== nothing
            cached = cache_get(cache, sol)
            cached !== nothing && return cached
        end
        cost = cost_fn(sol)
        if cache !== nothing
            cache_put!(cache, sol, cost)
        end
        return cost
    end
    return cached_cost_fn
end

# --- Atomic move helpers ---

function try_integer_moves(
    idx::Int,
    sol::Vector{Float64},
    best_benefit::Float64,
    cost_fn::Function,
    lower::Union{Vector{Float64}, Nothing},
    upper::Union{Vector{Float64}, Nothing},
)
    old = sol[idx]
    base = round(Int, old)
    cand_vals = (base - 1, base, base + 1)
    lo = lower !== nothing ? ceil(Int, lower[idx]) : typemin(Int)
    hi = upper !== nothing ? floor(Int, upper[idx]) : typemax(Int)
    for v in cand_vals
        (v < lo || v > hi) && continue
        sol[idx] = Float64(v)
        c = cost_fn(sol)
        if c < best_benefit
            result = copy(sol)
            return result, c, true
        end
    end
    sol[idx] = old
    return sol, best_benefit, false
end

function try_continuous_move(
    idx::Int,
    sol::Vector{Float64},
    best_benefit::Float64,
    cost_fn::Function,
    rng::AbstractRNG,
    lower::Union{Vector{Float64}, Nothing},
    upper::Union{Vector{Float64}, Nothing},
)
    old = sol[idx]
    if lower !== nothing && upper !== nothing
        span = upper[idx] - lower[idx]
        perturb = (rand(rng) - 0.5) * 0.1 * span  # ±5%
    else
        perturb = (rand(rng) - 0.5) * 0.2
    end
    new_val = old + perturb
    if lower !== nothing && upper !== nothing
        new_val = clamp(new_val, lower[idx], upper[idx])
    end
    sol[idx] = new_val
    cost = cost_fn(sol)
    if cost < best_benefit
        return true, cost
    end
    sol[idx] = old
    return false, best_benefit
end

function perturb_index!(
    perturbed::Vector{Float64},
    idx::Int,
    strength::Int,
    rng::AbstractRNG,
    lower::Union{Vector{Float64}, Nothing},
    upper::Union{Vector{Float64}, Nothing},
)
    half = get_half(length(perturbed))
    old = perturbed[idx]
    if idx > half  # integer variable (1-indexed: > half means integer part)
        lo = lower !== nothing ? ceil(Int, lower[idx]) : nothing
        hi = upper !== nothing ? floor(Int, upper[idx]) : nothing
        step = max(1, round(Int, strength / 2))
        delta = rand(rng, (-step):step)
        new_val = round(Int, old) + delta
        if lo !== nothing && hi !== nothing
            new_val = clamp(new_val, lo, hi)
        end
        perturbed[idx] = Float64(new_val)
    else  # continuous variable
        if lower !== nothing && upper !== nothing
            span = upper[idx] - lower[idx]
            delta = (rand(rng) - 0.5) * 0.3 * span  # ±15%
            perturbed[idx] = clamp(old + delta, lower[idx], upper[idx])
        else
            delta = randn(rng) * 0.12 * (abs(old) + 1e-6)
            perturbed[idx] = old + delta
        end
    end
end

# --- Neighbourhood functions ---

function neighborhood_flip(
    cost_fn::Function,
    solution::Vector{Float64},
    current_benefit::Float64,
    num_vars::Int;
    first_improvement::Bool = true,
    seed::Union{Int, Nothing} = nothing,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    sensitivity::Union{Vector{Float64}, Nothing} = nothing,
    deadline::Float64 = 0.0,
)
    rng = new_rng(seed)
    half = get_half(num_vars)

    if sensitivity !== nothing && any(sensitivity .> 0)
        noise = rand(rng, num_vars) .* 0.1 .* maximum(sensitivity)
        priority = sensitivity .+ noise
        indices = sortperm(priority; rev = true)
    else
        indices = randperm(rng, num_vars)
    end

    int_indices = filter(i -> i > half, indices)
    cont_indices = filter(i -> i <= half, indices)

    best_solution = copy(solution)
    best_benefit = current_benefit

    # Integer flip
    for (count, i) in enumerate(int_indices)
        count % 8 == 0 && expired(deadline) && break
        new_sol, new_ben, improved =
            try_integer_moves(i, solution, best_benefit, cost_fn, lower, upper)
        if improved
            best_benefit = new_ben
            best_solution = copy(new_sol)
            first_improvement && return best_solution, best_benefit
        end
    end

    # Continuous flip
    for (count, i) in enumerate(cont_indices)
        count % 8 == 0 && expired(deadline) && break
        old_val = solution[i]
        changed, new_ben =
            try_continuous_move(i, solution, best_benefit, cost_fn, rng, lower, upper)
        if changed
            best_benefit = new_ben
            best_solution = copy(solution)
            first_improvement && return best_solution, best_benefit
        end
        solution[i] = old_val
    end

    return best_solution, best_benefit
end

function neighborhood_swap(
    cost_fn::Function,
    solution::Vector{Float64},
    current_benefit::Float64,
    num_vars::Int;
    first_improvement::Bool = true,
    max_attempts::Int = 50,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    deadline::Float64 = 0.0,
)
    best_solution = copy(solution)
    best_benefit = current_benefit
    rng = new_rng()
    half = get_half(num_vars)
    (half <= 0 || half >= num_vars) && return best_solution, best_benefit
    n_int = num_vars - half

    for _ in 1:max_attempts
        expired(deadline) && break
        cont_idx = rand(rng, 1:half)
        int_idx = half + rand(rng, 1:n_int)

        old_cont = solution[cont_idx]
        old_int = solution[int_idx]

        if lower !== nothing && upper !== nothing
            span_cont = upper[cont_idx] - lower[cont_idx]
            solution[cont_idx] = clamp(
                old_cont + (rand(rng) - 0.5) * 0.16 * span_cont,
                lower[cont_idx],
                upper[cont_idx],
            )
            lo_int = ceil(Int, lower[int_idx])
            hi_int = floor(Int, upper[int_idx])
            new_int = round(Int, old_int) + rand(rng, -1:1)
            solution[int_idx] = Float64(clamp(new_int, lo_int, hi_int))
        else
            solution[cont_idx] = old_cont + (rand(rng) - 0.5) * 0.2
            solution[int_idx] = Float64(round(Int, old_int) + rand(rng, -1:1))
        end

        cost = cost_fn(solution)
        if cost < best_benefit
            best_benefit = cost
            best_solution = copy(solution)
            first_improvement && return best_solution, best_benefit
        end

        solution[cont_idx] = old_cont
        solution[int_idx] = old_int
    end

    return best_solution, best_benefit
end

function neighborhood_multiflip(
    cost_fn::Function,
    solution::Vector{Float64},
    current_benefit::Float64,
    num_vars::Int;
    k::Int = 3,
    max_attempts::Int = 50,
    seed::Union{Int, Nothing} = nothing,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    deadline::Float64 = 0.0,
)
    best_solution = copy(solution)
    best_benefit = current_benefit
    rng = new_rng(seed)
    half = get_half(num_vars)

    for _ in 1:max_attempts
        expired(deadline) && break
        indices = sort(randperm(rng, num_vars)[1:min(k, num_vars)])
        old_vals = solution[indices]

        # Modify integer indices
        for idx in indices
            if idx > half
                base = round(Int, solution[idx])
                delta = rand(rng, -1:1)
                new_val = base + delta
                if lower !== nothing && upper !== nothing
                    lo = ceil(Int, lower[idx])
                    hi = floor(Int, upper[idx])
                    new_val = clamp(new_val, lo, hi)
                end
                solution[idx] = Float64(new_val)
            else
                if lower !== nothing && upper !== nothing
                    span = upper[idx] - lower[idx]
                    perturb = (rand(rng) - 0.5) * 0.1 * span
                else
                    perturb = (rand(rng) - 0.5) * 0.2
                end
                new_val = solution[idx] + perturb
                if lower !== nothing && upper !== nothing
                    new_val = clamp(new_val, lower[idx], upper[idx])
                end
                solution[idx] = new_val
            end
        end

        cost = cost_fn(solution)
        if cost < best_benefit
            best_benefit = cost
            best_solution = copy(solution)
        end
        solution[indices] = old_vals
    end

    return best_solution, best_benefit
end

function try_neighborhoods(
    cached_cost_fn::Function,
    solution::Vector{Float64},
    current_benefit::Float64,
    num_vars::Int;
    first_improvement::Bool = true,
    iteration::Int = 1,
    no_improve_flip_limit::Int = 3,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    sensitivity::Union{Vector{Float64}, Nothing} = nothing,
    deadline::Float64 = 0.0,
)
    expired(deadline) && return solution, current_benefit, false

    new_sol, new_ben = neighborhood_flip(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars;
        first_improvement,
        lower,
        upper,
        sensitivity,
        deadline,
    )
    new_ben < current_benefit && return new_sol, new_ben, true

    expired(deadline) && return solution, current_benefit, false

    new_sol, new_ben = neighborhood_swap(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars;
        first_improvement,
        lower,
        upper,
        deadline,
    )
    new_ben < current_benefit && return new_sol, new_ben, true

    if iteration % no_improve_flip_limit == 0
        expired(deadline) && return solution, current_benefit, false
        new_sol, new_ben = neighborhood_multiflip(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars;
            k = no_improve_flip_limit,
            lower,
            upper,
            deadline,
        )
        new_ben < current_benefit && return new_sol, new_ben, true
    end

    return solution, current_benefit, false
end

function local_search_vnd(
    cost_fn::Function,
    solution::Vector{Float64},
    num_vars::Int;
    max_iter::Int = 300,
    first_improvement::Bool = true,
    no_improve_limit::Int = 5,
    no_improve_flip_limit::Int = 3,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    cache::Union{EvaluationCache, Nothing} = nothing,
    deadline::Float64 = 0.0,
)::Vector{Float64}
    solution = copy(solution)
    cached_cost_fn = create_cached_cost_fn(cost_fn, cache)
    current_benefit = cached_cost_fn(solution)

    sensitivity = zeros(Float64, num_vars)
    iteration = 0
    no_improve_count = 0

    while iteration < max_iter && no_improve_count < no_improve_limit
        expired(deadline) && break
        iteration += 1
        old_benefit = current_benefit
        old_solution = copy(solution)

        solution, current_benefit, improved = try_neighborhoods(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars;
            first_improvement,
            iteration,
            no_improve_flip_limit,
            lower,
            upper,
            sensitivity,
            deadline,
        )

        if improved
            no_improve_count = 0
            # Update sensitivity (P9)
            delta = abs(current_benefit - old_benefit)
            for i in 1:num_vars
                if solution[i] != old_solution[i]
                    sensitivity[i] = sensitivity[i] * 0.9 + delta
                else
                    sensitivity[i] *= 0.9
                end
            end
        else
            no_improve_count += 1
            sensitivity .*= 0.9
        end
    end

    return solution
end
