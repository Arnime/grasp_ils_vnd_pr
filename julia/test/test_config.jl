# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "GIVPConfig" begin
    @testset "defaults" begin
        cfg = GIVPConfig()
        @test cfg.max_iterations == 100
        @test cfg.alpha == 0.12
        @test cfg.vnd_iterations == 200
        @test cfg.ils_iterations == 10
        @test cfg.perturbation_strength == 4
        @test cfg.use_elite_pool == true
        @test cfg.elite_size == 7
        @test cfg.path_relink_frequency == 8
        @test cfg.adaptive_alpha == true
        @test cfg.alpha_min == 0.08
        @test cfg.alpha_max == 0.18
        @test cfg.num_candidates_per_step == 20
        @test cfg.use_cache == true
        @test cfg.cache_size == 10000
        @test cfg.early_stop_threshold == 80
        @test cfg.use_convergence_monitor == true
        @test cfg.n_workers == 1
        @test cfg.time_limit == 0.0
        @test cfg.direction == minimize
        @test cfg.integer_split === nothing
        @test cfg.group_size === nothing
        validate_config!(cfg)
    end

    @testset "valid edge values" begin
        cfg = validate_config!(GIVPConfig(alpha = 0.0, alpha_min = 0.0, alpha_max = 0.0))
        @test cfg.alpha == 0.0
        cfg = validate_config!(GIVPConfig(alpha = 1.0, alpha_min = 0.0, alpha_max = 1.0))
        @test cfg.alpha == 1.0
        cfg = validate_config!(GIVPConfig(perturbation_strength = 0))
        @test cfg.perturbation_strength == 0
        cfg = validate_config!(GIVPConfig(time_limit = 0.0))
        @test cfg.time_limit == 0.0
        cfg = validate_config!(GIVPConfig(integer_split = 0))
        @test cfg.integer_split == 0
    end

    @testset "Direction enum" begin
        @test minimize isa Direction
        @test maximize isa Direction
    end
end
