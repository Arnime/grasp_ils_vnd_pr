// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <givp/givp.hpp>

using namespace givp;

// ── Config validation ─────────────────────────────────────────────────────────

TEST_CASE("default config is valid", "[config]")
{
    REQUIRE_NOTHROW(GivpConfig{}.validate());
}

TEST_CASE("max_iterations == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.max_iterations = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("vnd_iterations == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.vnd_iterations = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("ils_iterations == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.ils_iterations = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("elite_size == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.elite_size = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("path_relink_frequency == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.path_relink_frequency = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("n_workers == 0 throws", "[config]")
{
    GivpConfig cfg;
    cfg.n_workers = 0;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("alpha out of range throws", "[config]")
{
    GivpConfig cfg;
    cfg.alpha = -0.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
    cfg.alpha = 1.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("alpha_min > alpha_max throws", "[config]")
{
    GivpConfig cfg;
    cfg.alpha_min = 0.5;
    cfg.alpha_max = 0.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

// ── Bounds validation ─────────────────────────────────────────────────────────

TEST_CASE("empty bounds throws InvalidBounds", "[config]")
{
    REQUIRE_THROWS_AS(givp::givp([](const std::vector<double> &)
                                 { return 0.0; },
                                 {}, GivpConfig{}),
                      InvalidBounds);
}

TEST_CASE("inverted bounds throws InvalidBounds", "[config]")
{
    std::vector<std::pair<double, double>> bad{{5.0, -1.0}};
    REQUIRE_THROWS_AS(givp::givp([](const std::vector<double> &)
                                 { return 0.0; },
                                 bad, GivpConfig{}),
                      InvalidBounds);
}

TEST_CASE("non-finite bounds throws InvalidBounds", "[config]")
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::vector<std::pair<double, double>> bad{{-inf, 1.0}};
    REQUIRE_THROWS_AS(givp::givp([](const std::vector<double> &)
                                 { return 0.0; },
                                 bad, GivpConfig{}),
                      InvalidBounds);
}

TEST_CASE("initial_guess wrong size throws InvalidInitialGuess", "[config]")
{
    std::vector<std::pair<double, double>> bounds(3, {-1.0, 1.0});
    GivpConfig cfg;
    cfg.initial_guess = std::vector<double>{0.0, 0.0}; // only 2 values for 3 vars
    REQUIRE_THROWS_AS(
        givp::givp([](const std::vector<double> &)
                   { return 0.0; }, bounds, cfg),
        InvalidInitialGuess);
}

TEST_CASE("initial_guess out of bounds throws InvalidInitialGuess", "[config]")
{
    std::vector<std::pair<double, double>> bounds(2, {-1.0, 1.0});
    GivpConfig cfg;
    cfg.initial_guess = std::vector<double>{0.0, 5.0}; // 5.0 > upper bound
    REQUIRE_THROWS_AS(
        givp::givp([](const std::vector<double> &)
                   { return 0.0; }, bounds, cfg),
        InvalidInitialGuess);
}

// ── TerminationReason parsing ─────────────────────────────────────────────────

TEST_CASE("termination_from_message parses known strings", "[config]")
{
    REQUIRE(termination_from_message("max iterations reached") ==
            TerminationReason::MaxIterationsReached);
    REQUIRE(termination_from_message("time limit reached") ==
            TerminationReason::TimeLimitReached);
    REQUIRE(termination_from_message("early stop due to stagnation") ==
            TerminationReason::EarlyStop);
    REQUIRE(termination_from_message("converged") ==
            TerminationReason::Converged);
    REQUIRE(termination_from_message("no feasible solution found") ==
            TerminationReason::NoFeasible);
    REQUIRE(termination_from_message("") ==
            TerminationReason::Unknown);
}
