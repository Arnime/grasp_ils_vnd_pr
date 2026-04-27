# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""ILS — Iterated Local Search."""

function perturb_solution(solution::Vector{Float64}, num_vars::Int;
                          strength::Int=4,
                          seed::Union{Int,Nothing}=nothing,
                          lower::Union{Vector{Float64},Nothing}=nothing,
                          upper::Union{Vector{Float64},Nothing}=nothing)::Vector{Float64}
    perturbed = copy(solution)
    rng = new_rng(seed)
    n_perturb = min(max(strength, num_vars ÷ 5), num_vars)
    indices = randperm(rng, num_vars)[1:n_perturb]
    for idx in indices
        perturb_index!(perturbed, idx, strength, rng, lower, upper)
    end
    return perturbed
end

function ils_search(solution::Vector{Float64}, current_cost::Float64,
                    num_vars::Int, cost_fn::Function, config::GIVPConfig;
                    lower::Union{Vector{Float64},Nothing}=nothing,
                    upper::Union{Vector{Float64},Nothing}=nothing,
                    cache::Union{EvaluationCache,Nothing}=nothing,
                    deadline::Float64=0.0)::Tuple{Vector{Float64},Float64}
    best_solution = copy(solution)
    best_cost = current_cost

    for ils_iter in 0:(config.ils_iterations - 1)
        expired(deadline) && break

        # P12: progressive adaptive strength
        progress = ils_iter / max(1, config.ils_iterations - 1)
        adaptive_strength = max(
            config.perturbation_strength,
            round(Int, config.perturbation_strength * (1.0 + progress)))

        perturbed = perturb_solution(best_solution, num_vars;
            strength=adaptive_strength, lower, upper)

        perturbed = local_search_vnd(cost_fn, perturbed, num_vars;
            max_iter=config.vnd_iterations,
            lower, upper, cache, deadline)

        perturbed_cost = if cache !== nothing
            cached = cache_get(cache, perturbed)
            if cached !== nothing
                cached
            else
                c = cost_fn(perturbed)
                cache_put!(cache, perturbed, c)
                c
            end
        else
            cost_fn(perturbed)
        end

        if perturbed_cost < best_cost
            best_cost = perturbed_cost
            best_solution = copy(perturbed)
        elseif perturbed_cost < best_cost * 1.25  # 25% worse acceptance
            # Accept slightly worse with some probability for diversification
            rng = new_rng()
            if rand(rng) < 0.1
                best_solution = copy(perturbed)
            end
        end
    end

    return best_solution, best_cost
end
