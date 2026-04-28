// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
//
// Tests that specifically exercise code paths not reached by the basic suite:
//   - Mixed-integer variables (integer_split < num_vars)
//   - Stagnation-restart block in impl_core
//   - Convergence-monitor restart block in impl_core
//   - Path-relinking diff-variable cap (>MAX_PR_VARS dimensions)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include <givp/givp.hpp>

using namespace givp;

// ── Shared objective functions ────────────────────────────────────────────────

static double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (auto v : x) s += v * v;
    return s;
}

static double rastrigin(const std::vector<double>& x) {
    constexpr double pi = 3.14159265358979323846;
    double n = static_cast<double>(x.size());
    double s = 10.0 * n;
    for (auto v : x) s += v * v - 10.0 * std::cos(2.0 * pi * v);
    return s;
}

// ── Mixed-integer (exercises try_integer_moves + perturb_index integer branch) ─

TEST_CASE("mixed integer: 3 continuous + 2 integer variables", "[mixed]") {
    // half = integer_split = 3 → vars 3 and 4 are integer-rounded
    std::vector<std::pair<double, double>> bounds = {
        {-5.0, 5.0}, {-5.0, 5.0}, {-5.0, 5.0},   // continuous
        {0.0,  4.0}, {0.0,  4.0}                   // integer
    };
    GivpConfig cfg;
    cfg.seed           = 42;
    cfg.max_iterations = 30;
    cfg.integer_split  = 3;  // half = 3

    auto result = givp::givp(sphere, bounds, cfg);

    REQUIRE(result.success);
    REQUIRE(result.x.size() == 5);
    // Integer tail must be whole numbers
    REQUIRE(result.x[3] == Catch::Approx(std::round(result.x[3])).margin(1e-9));
    REQUIRE(result.x[4] == Catch::Approx(std::round(result.x[4])).margin(1e-9));
}

TEST_CASE("all-integer problem (integer_split = 0)", "[mixed]") {
    std::vector<std::pair<double, double>> bounds(4, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed           = 7;
    cfg.max_iterations = 20;
    cfg.integer_split  = 0;  // all integer

    auto result = givp::givp(sphere, bounds, cfg);
    REQUIRE(result.success);
    for (std::size_t i = 0; i < result.x.size(); ++i)
        REQUIRE(result.x[i] ==
                Catch::Approx(std::round(result.x[i])).margin(1e-9));
}

TEST_CASE("mixed integer with initial guess in integer region", "[mixed]") {
    std::vector<std::pair<double, double>> bounds = {
        {-5.0, 5.0}, {-5.0, 5.0},   // continuous
        {0.0,  3.0}, {0.0,  3.0}    // integer
    };
    GivpConfig cfg;
    cfg.seed          = 99;
    cfg.max_iterations = 20;
    cfg.integer_split  = 2;
    cfg.initial_guess  = std::vector<double>{0.0, 0.0, 1.0, 2.0};

    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}

// ── Stagnation restart (flat objective → stagnation > max_iter/4) ─────────────

TEST_CASE("flat objective triggers stagnation restart", "[advanced]") {
    // All evaluations return the same cost: stagnation accumulates every
    // iteration.  With max_iterations=20, restart fires at stagnation > 5.
    auto flat = [](const std::vector<double>&) -> double { return 1.0; };
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed                    = 42;
    cfg.max_iterations          = 20;
    cfg.integer_split           = 5;
    cfg.use_convergence_monitor = false;  // isolate the stagnation path
    cfg.early_stop_threshold    = 1000;

    REQUIRE_NOTHROW(givp::givp(flat, bounds, cfg));
}

// ── Convergence-monitor restart (should_restart branch + cache->clear()) ──────

TEST_CASE("convergence monitor fires restart on plateau", "[advanced]") {
    // Flat objective → no_improve_count reaches restart_threshold (50 updates).
    // Two conv_monitor->update calls per iteration: fires around iteration 25.
    // early_stop_threshold=1000 prevents early exit before the restart.
    auto flat = [](const std::vector<double>&) -> double { return 1.0; };
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed                    = 42;
    cfg.max_iterations          = 30;
    cfg.integer_split           = 5;
    cfg.use_convergence_monitor = true;
    cfg.use_cache               = true;   // exercises cache->clear() in restart
    cfg.early_stop_threshold    = 1000;

    REQUIRE_NOTHROW(givp::givp(flat, bounds, cfg));
}

// ── Path relinking diff-variable cap (diff_indices.size() > MAX_PR_VARS = 25) ─

TEST_CASE("path relinking 30D problem exercises diff-variable cap", "[advanced]") {
    // 30D problem: two distinct elite solutions almost certainly differ in all
    // 30 dimensions, which exceeds MAX_PR_VARS=25 and exercises the trim branch.
    std::vector<std::pair<double, double>> bounds(30, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed                  = 42;
    cfg.max_iterations        = 15;
    cfg.integer_split         = 30;
    cfg.path_relink_frequency = 2;  // fire PR at iterations 2, 4, 6, ...
    cfg.use_elite_pool        = true;
    cfg.elite_size            = 5;

    REQUIRE_NOTHROW(givp::givp(rastrigin, bounds, cfg));
}

// ── bidirectional_path_relinking with identical solutions (empty diff) ─────────

TEST_CASE("path relinking with identical elite solutions is a no-op", "[advanced]") {
    // Force identical solutions into the elite pool, then trigger PR.
    // When all diff_indices are empty, bidirectional_path_relinking returns the
    // current cost immediately without iterating — exercises the empty-diff branch.
    //
    // Strategy: use a 1D all-flat objective so every candidate gets the same
    // cost and the pool accumulates near-identical solutions.
    auto flat1d = [](const std::vector<double>&) -> double { return 0.0; };

    std::vector<std::pair<double, double>> bounds(1, {0.0, 1.0});
    GivpConfig cfg;
    cfg.seed                  = 1;
    cfg.max_iterations        = 12;
    cfg.integer_split         = 1;
    cfg.path_relink_frequency = 2;
    cfg.use_elite_pool        = true;
    cfg.elite_size            = 3;
    cfg.use_convergence_monitor = false;
    cfg.early_stop_threshold  = 1000;

    REQUIRE_NOTHROW(givp::givp(flat1d, bounds, cfg));
}
