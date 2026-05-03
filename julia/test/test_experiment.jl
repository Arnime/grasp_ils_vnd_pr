# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

using Test
using GIVPOptimizer

@testset "experiment" begin
    sphere(x) = sum(v * v for v in x)
    bounds = [(-5.12, 5.12) for _ in 1:3]

    rows = seed_sweep(sphere, bounds; seeds = 5, direction = minimize)
    @test length(rows) == 5

    for row in rows
        @test haskey(row, "seed")
        @test haskey(row, "fun")
        @test haskey(row, "nit")
        @test haskey(row, "nfev")
        @test haskey(row, "time_s")
        @test haskey(row, "success")
        @test haskey(row, "message")
    end

    summary = sweep_summary(rows)
    for metric in ("fun", "nit", "nfev", "time_s")
        @test haskey(summary, metric)
        @test haskey(summary[metric], "mean")
        @test haskey(summary[metric], "std")
        @test haskey(summary[metric], "min")
        @test haskey(summary[metric], "max")
        @test summary[metric]["min"] <= summary[metric]["max"]
        @test summary[metric]["std"] >= 0.0
    end

    seeds = collect(0:4)
    cfg = GIVPConfig(; max_iterations = 8, integer_split = 3)
    rows_a = seed_sweep(sphere, bounds; seeds = seeds, config = cfg, direction = minimize)
    rows_b = seed_sweep(sphere, bounds; seeds = seeds, config = cfg, direction = minimize)

    @test length(rows_a) == length(rows_b)
    for (ra, rb) in zip(rows_a, rows_b)
        @test ra["seed"] == rb["seed"]
        @test isapprox(Float64(ra["fun"]), Float64(rb["fun"]); atol = 1e-12, rtol = 0.0)
        @test ra["nfev"] == rb["nfev"]
        @test ra["nit"] == rb["nit"]
    end

    summary_a = sweep_summary(rows_a)
    summary_b = sweep_summary(rows_b)
    @test isapprox(summary_a["fun"]["mean"], summary_b["fun"]["mean"]; atol = 1e-12, rtol = 0.0)
    @test isapprox(summary_a["nfev"]["mean"], summary_b["nfev"]["mean"]; atol = 1e-12, rtol = 0.0)
end
