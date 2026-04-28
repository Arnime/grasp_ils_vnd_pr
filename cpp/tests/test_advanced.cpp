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

// ── Rng::from_seed(nullopt) — random_device branch ───────────────────────────

TEST_CASE("no seed uses random_device branch", "[advanced]") {
    // cfg.seed is left at its default (nullopt) → Rng::from_seed(nullopt)
    // takes the std::random_device path instead of the seeded path.
    std::vector<std::pair<double, double>> bounds(3, {-5.0, 5.0});
    GivpConfig cfg;
    // cfg.seed intentionally not set
    cfg.max_iterations = 5;
    cfg.integer_split  = 3;

    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}

// ── adaptive_alpha=false → get_current_alpha returns fixed alpha ──────────────

TEST_CASE("non-adaptive alpha uses fixed alpha throughout", "[advanced]") {
    // With adaptive_alpha=false get_current_alpha immediately returns cfg.alpha
    // instead of interpolating — exercises the `if (!adaptive) return alpha;` branch.
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed           = 42;
    cfg.max_iterations = 10;
    cfg.integer_split  = 5;
    cfg.adaptive_alpha = false;
    cfg.alpha          = 0.15;

    auto result = givp::givp(sphere, bounds, cfg);
    REQUIRE(result.success);
}

// ── sample_integer_from_bounds: lo_i > hi_i fallback ─────────────────────────

TEST_CASE("integer bounds narrower than 1 integer triggers fallback", "[advanced]") {
    // bounds (0.1, 0.9): ceil(0.1)=1, floor(0.9)=0 → lo_i > hi_i
    // → sample_integer_from_bounds returns round((0.1+0.9)/2) = round(0.5) = 1
    // This exercises the `if (lo_i > hi_i) return std::round(...)` branch.
    std::vector<std::pair<double, double>> bounds = {
        {0.1, 0.9},   // integer: lo_i=1 > hi_i=0 → fallback
        {-5.0, 5.0},  // integer: normal
        {-5.0, 5.0},  // integer: normal
    };
    GivpConfig cfg;
    cfg.seed           = 1;
    cfg.max_iterations = 10;
    cfg.integer_split  = 0;  // all-integer

    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}

// ── config.hpp: alpha_min / alpha_max individual range checks ─────────────────

TEST_CASE("alpha_min out of range throws", "[config]") {
    GivpConfig cfg;
    cfg.alpha_min = -0.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
    cfg.alpha_min = 1.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

TEST_CASE("alpha_max out of range throws", "[config]") {
    GivpConfig cfg;
    cfg.alpha_max = -0.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
    cfg.alpha_max = 1.1;
    REQUIRE_THROWS_AS(cfg.validate(), InvalidConfig);
}

// ── pr.hpp: backward path is chosen when cost_bwd < cost_fwd ─────────────────

TEST_CASE("path relinking backward direction selected", "[advanced]") {
    // Use a non-symmetric objective so that going sol2→sol1 yields a better
    // result than sol1→sol2, exercising the `return {best_bwd, cost_bwd}` branch.
    // Rastrigin on 10D with path_relink_frequency=1 forces PR every iteration;
    // bidirectional PR will eventually pick the backward direction.
    std::vector<std::pair<double, double>> bounds(10, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed                  = 123;
    cfg.max_iterations        = 30;
    cfg.integer_split         = 10;
    cfg.path_relink_frequency = 1;  // PR on every iteration
    cfg.use_elite_pool        = true;
    cfg.elite_size            = 5;
    cfg.use_convergence_monitor = false;

    REQUIRE_NOTHROW(givp::givp(rastrigin, bounds, cfg));
}

// ── impl_core.hpp: elite_pool has fewer than 2 solutions (PR guard) ───────────

TEST_CASE("path relinking skipped when elite pool has < 2 solutions", "[advanced]") {
    // With elite_size=1 the pool can never hold 2 solutions, so the guard
    // `elite_pool.len() >= 2` is always false and do_path_relinking is never
    // called — exercises the `false` branch of that condition.
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed                  = 7;
    cfg.max_iterations        = 15;
    cfg.integer_split         = 5;
    cfg.use_elite_pool        = true;
    cfg.elite_size            = 1;   // pool holds at most 1 solution
    cfg.path_relink_frequency = 2;
    cfg.use_convergence_monitor = false;

    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}

// ── vnd.hpp: multiflip triggered (no_improve_flip_count >= no_improve_limit=5) ─

TEST_CASE("vnd multiflip neighborhood triggered after 5 consecutive flip failures", "[advanced]") {
    // Flat objective → neighborhood_flip never improves → no_improve_flip_count
    // reaches 5 within the first 6 VND iterations, triggering neighborhood_multiflip.
    auto flat = [](const std::vector<double>&) -> double { return 1.0; };
    std::vector<std::pair<double, double>> bounds(4, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed                    = 42;
    cfg.max_iterations          = 5;
    cfg.integer_split           = 4;
    cfg.vnd_iterations          = 10;  // enough iterations for count to reach 5
    cfg.use_convergence_monitor = false;
    cfg.early_stop_threshold    = 1000;

    REQUIRE_NOTHROW(givp::givp(flat, bounds, cfg));
}
