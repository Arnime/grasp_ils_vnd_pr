# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""Experimental utilities for reproducible multi-seed benchmarking."""

"""
    seed_sweep(func, bounds; seeds=30, config=nothing, direction=minimize, verbose=false)

Run the optimizer for multiple independent seeds and collect per-seed metrics.
Returns a `Vector{Dict{String,Any}}` with one entry per seed.
"""
function seed_sweep(
    func::Function,
    bounds;
    seeds::Union{Int, AbstractVector{Int}} = 30,
    config::Union{GIVPConfig, Nothing} = nothing,
    direction::Direction = minimize,
    verbose::Bool = false,
)
    seed_list = seeds isa Int ? collect(0:(seeds - 1)) : collect(seeds)
    cfg = config !== nothing ? deepcopy(config) : GIVPConfig()

    rows = Dict{String, Any}[]
    for s in seed_list
        t0 = time()
        result = givp(func, bounds; direction, config = cfg, seed = s, verbose)
        elapsed = time() - t0
        push!(
            rows,
            Dict{String, Any}(
                "seed" => s,
                "fun" => result.fun,
                "nit" => result.nit,
                "nfev" => result.nfev,
                "time_s" => elapsed,
                "success" => result.success,
                "message" => result.message,
            ),
        )
    end
    return rows
end

"""
    sweep_summary(results)

Aggregate seed-sweep results into `mean/std/min/max` for `fun`, `nit`,
`nfev`, and `time_s`.
"""
function sweep_summary(results::AbstractVector{<:AbstractDict{String, Any}})
    metrics = ("fun", "nit", "nfev", "time_s")
    summary = Dict{String, Dict{String, Float64}}()

    for key in metrics
        values = [Float64(row[key]) for row in results]
        n = length(values)
        n == 0 && continue

        μ = mean(values)
        σ = n > 1 ? std(values) : 0.0
        summary[key] = Dict{String, Float64}(
            "mean" => μ,
            "std" => σ,
            "min" => minimum(values),
            "max" => maximum(values),
        )
    end

    return summary
end
