# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Configuration for the GRASP-ILS-VND-PR algorithm."""

@enum Direction minimize maximize

"""
    GIVPConfig(; kwargs...)

Hyper-parameters for the GRASP-ILS-VND-PR algorithm.
"""
Base.@kwdef mutable struct GIVPConfig
    max_iterations::Int = 100
    alpha::Float64 = 0.12
    vnd_iterations::Int = 200
    ils_iterations::Int = 10
    perturbation_strength::Int = 4
    use_elite_pool::Bool = true
    elite_size::Int = 7
    path_relink_frequency::Int = 8
    adaptive_alpha::Bool = true
    alpha_min::Float64 = 0.08
    alpha_max::Float64 = 0.18
    num_candidates_per_step::Int = 20
    use_cache::Bool = true
    cache_size::Int = 10000
    early_stop_threshold::Int = 80
    use_convergence_monitor::Bool = true
    n_workers::Int = 1
    time_limit::Float64 = 0.0
    direction::Direction = minimize
    integer_split::Union{Int,Nothing} = nothing
    group_size::Union{Int,Nothing} = nothing
end

function validate_config!(cfg::GIVPConfig)
    positive_int_fields = [
        (:max_iterations, cfg.max_iterations),
        (:vnd_iterations, cfg.vnd_iterations),
        (:ils_iterations, cfg.ils_iterations),
        (:elite_size, cfg.elite_size),
        (:path_relink_frequency, cfg.path_relink_frequency),
        (:num_candidates_per_step, cfg.num_candidates_per_step),
        (:cache_size, cfg.cache_size),
        (:early_stop_threshold, cfg.early_stop_threshold),
        (:n_workers, cfg.n_workers),
    ]
    for (name, value) in positive_int_fields
        value < 1 && throw(InvalidConfigError("$name must be a positive integer, got $value"))
    end

    cfg.perturbation_strength < 0 &&
        throw(InvalidConfigError("perturbation_strength must be non-negative, got $(cfg.perturbation_strength)"))

    !(0.0 <= cfg.alpha <= 1.0) &&
        throw(InvalidConfigError("alpha must be in [0, 1], got $(cfg.alpha)"))
    !(0.0 <= cfg.alpha_min <= 1.0) &&
        throw(InvalidConfigError("alpha_min must be in [0, 1], got $(cfg.alpha_min)"))
    !(0.0 <= cfg.alpha_max <= 1.0) &&
        throw(InvalidConfigError("alpha_max must be in [0, 1], got $(cfg.alpha_max)"))
    cfg.alpha_min > cfg.alpha_max &&
        throw(InvalidConfigError("alpha_min ($(cfg.alpha_min)) must be <= alpha_max ($(cfg.alpha_max))"))

    cfg.time_limit < 0 &&
        throw(InvalidConfigError("time_limit must be >= 0, got $(cfg.time_limit)"))

    if cfg.integer_split !== nothing && cfg.integer_split < 0
        throw(InvalidConfigError("integer_split must be >= 0 or nothing, got $(cfg.integer_split)"))
    end

    return cfg
end
