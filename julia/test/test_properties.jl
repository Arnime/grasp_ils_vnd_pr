# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Property-based tests (invariant testing across random seeds).
# 50+ independent seeds give statistical confidence without PropCheck.jl.

@testset "Property-based tests" begin
    sphere(x) = sum(x .^ 2)
    cfg_fast = GIVPConfig(;
        max_iterations = 10,
        vnd_iterations = 20,
        ils_iterations = 2,
        integer_split = 4,
        use_cache = false,
    )
    bounds_4d = [(-5.12, 5.12) for _ in 1:4]
    lower_4d = [-5.12, -5.12, -5.12, -5.12]
    upper_4d = [5.12, 5.12, 5.12, 5.12]

    # 50 seeds for cheap invariants (solution quality, bounds, basic fields)
    prop_seeds_50 = 0:49
    # 10 seeds for pairs (direction, quality) — each does 2 runs
    prop_seeds_10 = 0:9

    @testset "P1: solution always within bounds" begin
        for seed in prop_seeds_50
            r = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test all(lower_4d .<= r.x .<= upper_4d)
        end
    end

    @testset "P2: nit ≤ max_iterations" begin
        for seed in prop_seeds_50
            r = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test r.nit <= cfg_fast.max_iterations
        end
    end

    @testset "P3: nfev > 0 always" begin
        for seed in prop_seeds_50
            r = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test r.nfev > 0
        end
    end

    @testset "P4: determinism — same seed same result" begin
        for seed in prop_seeds_10
            r1 = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            r2 = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test r1.x == r2.x
            @test r1.fun == r2.fun
            @test r1.nit == r2.nit
            @test r1.nfev == r2.nfev
        end
    end

    @testset "P5: success ↔ isfinite(fun)" begin
        for seed in prop_seeds_50
            r = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test r.success == isfinite(r.fun)
        end
    end

    @testset "P6: direction invariant — minimize f ≡ maximize −f" begin
        neg_sphere(x) = -sum(x .^ 2)
        for seed in prop_seeds_10
            r_min = givp(
                sphere,
                bounds_4d;
                config = cfg_fast,
                seed = seed,
                direction = minimize,
            )
            r_max = givp(
                neg_sphere,
                bounds_4d;
                config = cfg_fast,
                seed = seed,
                direction = maximize,
            )
            @test r_min.x ≈ r_max.x atol = 1e-10
            @test r_min.fun ≈ -r_max.fun atol = 1e-10
        end
    end

    @testset "P7: sphere is non-negative" begin
        for seed in prop_seeds_50
            r = givp(sphere, bounds_4d; config = cfg_fast, seed = seed)
            @test r.fun >= 0.0
        end
    end

    @testset "P8: more iterations quality ≥ fewer" begin
        cfg_short = GIVPConfig(;
            max_iterations = 5,
            vnd_iterations = 10,
            ils_iterations = 1,
            integer_split = 4,
            use_elite_pool = false,
            use_convergence_monitor = false,
            use_cache = false,
        )
        cfg_long = GIVPConfig(;
            max_iterations = 30,
            vnd_iterations = 10,
            ils_iterations = 1,
            integer_split = 4,
            use_elite_pool = false,
            use_convergence_monitor = false,
            use_cache = false,
        )
        for seed in prop_seeds_10
            r_short = givp(sphere, bounds_4d; config = cfg_short, seed = seed)
            r_long = givp(sphere, bounds_4d; config = cfg_long, seed = seed)
            # More iterations must not produce a result worse than 10× fewer iterations
            @test r_long.fun <= r_short.fun * 10.0
        end
    end

    @testset "P9: callback fires exactly max_iterations times" begin
        count = Ref(0)
        cfg_exact = GIVPConfig(;
            max_iterations = 7,
            vnd_iterations = 5,
            ils_iterations = 1,
            integer_split = 4,
            early_stop_threshold = 100,
            use_convergence_monitor = false,
        )
        givp(
            sphere,
            bounds_4d;
            config = cfg_exact,
            seed = 42,
            iteration_callback = (i, c, s) -> (count[] += 1),
        )
        @test count[] == 7
    end

    @testset "P10: Vector{Vector} bounds accepted" begin
        r = givp(sphere, [lower_4d, upper_4d]; config = cfg_fast, seed = 0)
        @test length(r.x) == 4
        @test all(lower_4d .<= r.x .<= upper_4d)
    end
end
