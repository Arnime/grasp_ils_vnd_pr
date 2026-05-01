# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "Exceptions" begin
    @testset "Config validation errors" begin
        @test_throws InvalidConfigError validate_config!(GIVPConfig(max_iterations = 0))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(vnd_iterations = 0))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(ils_iterations = 0))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(elite_size = 0))
        @test_throws InvalidConfigError validate_config!(
            GIVPConfig(path_relink_frequency = 0),
        )
        @test_throws InvalidConfigError validate_config!(
            GIVPConfig(num_candidates_per_step = 0),
        )
        @test_throws InvalidConfigError validate_config!(GIVPConfig(cache_size = 0))
        @test_throws InvalidConfigError validate_config!(
            GIVPConfig(early_stop_threshold = 0),
        )
        @test_throws InvalidConfigError validate_config!(GIVPConfig(n_workers = 0))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha = -0.1))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha = 1.1))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_min = -0.1))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_min = 1.1))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_max = -0.1))
        @test_throws InvalidConfigError validate_config!(GIVPConfig(alpha_max = 1.1))
        @test_throws InvalidConfigError validate_config!(
            GIVPConfig(alpha_min = 0.5, alpha_max = 0.1),
        )
        @test_throws InvalidConfigError validate_config!(GIVPConfig(time_limit = -1.0))
        @test_throws InvalidConfigError validate_config!(
            GIVPConfig(perturbation_strength = -1),
        )
        @test_throws InvalidConfigError validate_config!(GIVPConfig(integer_split = -1))
    end

    @testset "showerror coverage" begin
        buf = IOBuffer()
        Base.showerror(buf, InvalidBoundsError("test bounds"))
        @test occursin("InvalidBoundsError", String(take!(buf)))

        Base.showerror(buf, InvalidInitialGuessError("test guess"))
        @test occursin("InvalidInitialGuessError", String(take!(buf)))

        Base.showerror(buf, InvalidConfigError("test config"))
        @test occursin("InvalidConfigError", String(take!(buf)))

        Base.showerror(buf, EvaluatorError("test eval"))
        @test occursin("EvaluatorError", String(take!(buf)))

        Base.showerror(buf, EmptyPoolError("test pool"))
        @test occursin("EmptyPoolError", String(take!(buf)))
    end

    @testset "Exception hierarchy" begin
        @test InvalidBoundsError <: GivpError
        @test InvalidInitialGuessError <: GivpError
        @test InvalidConfigError <: GivpError
        @test EvaluatorError <: GivpError
        @test EmptyPoolError <: GivpError
        @test GivpError <: Exception
    end
end
