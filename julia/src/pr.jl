# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Path Relinking."""

const MAX_PR_VARS = 25

function path_relinking_forward(
    cost_fn::Function,
    source::Vector{Float64},
    target::Vector{Float64},
    diff_indices::Vector{Int};
    deadline::Float64 = 0.0,
)::Tuple{Vector{Float64}, Float64}
    current = copy(source)
    best_solution = copy(current)
    best_benefit = cost_fn(current)

    for (count, idx) in enumerate(diff_indices)
        count % 5 == 0 && expired(deadline) && break
        current[idx] = target[idx]
        cost = cost_fn(current)
        if cost < best_benefit
            best_benefit = cost
            best_solution = copy(current)
        end
    end
    return best_solution, best_benefit
end

function path_relinking_best(
    cost_fn::Function,
    source::Vector{Float64},
    target::Vector{Float64},
    diff_indices::Vector{Int};
    deadline::Float64 = 0.0,
)::Tuple{Vector{Float64}, Float64}
    current = copy(source)
    best_solution = copy(current)
    best_benefit = cost_fn(current)
    indices = copy(diff_indices)

    while !isempty(indices)
        expired(deadline) && break
        best_move_idx = nothing
        best_move_benefit = best_benefit

        for (count, idx) in enumerate(indices)
            count % 5 == 0 && expired(deadline) && break
            current[idx] == target[idx] && continue
            old_val = current[idx]
            current[idx] = target[idx]
            cost = cost_fn(current)
            if cost < best_move_benefit
                best_move_benefit = cost
                best_move_idx = idx
            end
            current[idx] = old_val
        end

        if best_move_idx !== nothing
            current[best_move_idx] = target[best_move_idx]
            filter!(i -> i != best_move_idx, indices)
            if best_move_benefit < best_benefit
                best_benefit = best_move_benefit
                best_solution = copy(current)
            end
        else
            break
        end
    end

    return best_solution, best_benefit
end

function path_relinking(
    cost_fn::Function,
    source::Vector{Float64},
    target::Vector{Float64};
    strategy::Symbol = :best,
    seed::Union{Int, Nothing} = nothing,
    deadline::Float64 = 0.0,
)::Tuple{Vector{Float64}, Float64}
    src = copy(source)
    tgt = copy(target)
    diff_indices = findall(abs.(src .- tgt) .> 1e-9)

    isempty(diff_indices) && return copy(src), cost_fn(src)

    # Limit to top-K most different variables
    if length(diff_indices) > MAX_PR_VARS
        diffs = abs.(src[diff_indices] .- tgt[diff_indices])
        top_k = sortperm(diffs; rev = true)[1:MAX_PR_VARS]
        diff_indices = diff_indices[top_k]
    end

    rng = new_rng(seed)
    shuffle!(rng, diff_indices)

    if strategy == :best
        return path_relinking_best(cost_fn, src, tgt, diff_indices; deadline)
    end
    return path_relinking_forward(cost_fn, src, tgt, diff_indices; deadline)
end

function bidirectional_path_relinking(
    cost_fn::Function,
    sol1::Vector{Float64},
    sol2::Vector{Float64};
    deadline::Float64 = 0.0,
)::Tuple{Vector{Float64}, Float64}
    best1, cost1 = path_relinking(cost_fn, sol1, sol2; strategy = :forward, deadline)
    expired(deadline) && return best1, cost1
    best2, cost2 = path_relinking(cost_fn, sol2, sol1; strategy = :forward, deadline)
    return cost1 <= cost2 ? (best1, cost1) : (best2, cost2)
end
