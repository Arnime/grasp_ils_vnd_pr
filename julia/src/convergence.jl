# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Convergence monitor that recommends restarts/intensification."""

mutable struct ConvergenceMonitor
    window_size::Int
    restart_threshold::Int
    history::Vector{Float64}
    no_improve_count::Int
    best_ever::Float64
    diversity_scores::Vector{Float64}
end

function ConvergenceMonitor(; window_size::Int=20, restart_threshold::Int=50)
    ConvergenceMonitor(window_size, restart_threshold, Float64[], 0, Inf, Float64[])
end

function update!(cm::ConvergenceMonitor, current_cost::Float64,
                 elite_pool::Union{ElitePool,Nothing}=nothing)::Dict{String,Any}
    push!(cm.history, current_cost)

    if current_cost < cm.best_ever
        cm.best_ever = current_cost
        cm.no_improve_count = 0
    else
        cm.no_improve_count += 1
    end

    diversity = 0.0
    if elite_pool !== nothing && pool_size(elite_pool) >= 2
        solutions = [sol for (sol, _) in get_all(elite_pool)]
        distances = Float64[]
        for i in 1:length(solutions)
            for j in (i+1):length(solutions)
                push!(distances, norm(solutions[i] - solutions[j]))
            end
        end
        diversity = isempty(distances) ? 0.0 : mean(distances)
    end

    push!(cm.diversity_scores, diversity)

    should_restart = cm.no_improve_count >= cm.restart_threshold
    should_intensify = cm.no_improve_count >= cm.restart_threshold ÷ 2 && diversity < 0.5

    Dict{String,Any}(
        "should_restart" => should_restart,
        "should_intensify" => should_intensify,
        "diversity" => diversity,
        "no_improve_count" => cm.no_improve_count,
    )
end
