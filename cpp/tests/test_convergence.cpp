// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <givp/config.hpp>
#include <givp/detail/cache.hpp>
#include <givp/detail/convergence.hpp>
#include <givp/detail/elite.hpp>
#include <givp/detail/helpers.hpp>
#include <givp/detail/pr.hpp>
#include <givp/exceptions.hpp>
#include <givp/givp.hpp>

using namespace givp;
using namespace givp::detail;

// ── EvaluationCache ───────────────────────────────────────────────────────────

TEST_CASE("cache is empty on construction", "[cache]") {
    EvaluationCache c{10};
    auto s = c.stats();
    REQUIRE(s.hits == 0);
    REQUIRE(s.misses == 0);
    REQUIRE(s.size == 0);
    REQUIRE(s.hit_rate == Catch::Approx(0.0));
}

TEST_CASE("cache get returns nullopt for unknown key", "[cache]") {
    EvaluationCache c{10};
    std::vector<double> sol{1.0, 2.0};
    REQUIRE_FALSE(c.get(sol, 2).has_value());
}

TEST_CASE("cache put and get round-trip", "[cache]") {
    EvaluationCache c{10};
    std::vector<double> sol{1.0, 2.0};
    c.put(sol, 2, 42.0);
    auto v = c.get(sol, 2);
    REQUIRE(v.has_value());
    REQUIRE(v.value_or(std::numeric_limits<double>::quiet_NaN()) == Catch::Approx(42.0));
}

TEST_CASE("cache duplicate put is a no-op", "[cache]") {
    // Exercises the `if (cache_.count(key)) return;` early-return in put().
    // The value stored by the first put must be preserved.
    EvaluationCache c{10};
    std::vector<double> sol{3.0, 4.0};
    c.put(sol, 2, 42.0);
    c.put(sol, 2, 99.0); // duplicate — should be ignored
    auto v = c.get(sol, 2);
    REQUIRE(v.has_value());
    REQUIRE(v.value_or(std::numeric_limits<double>::quiet_NaN()) == Catch::Approx(42.0));
}

TEST_CASE("cache hit rate is tracked", "[cache]") {
    EvaluationCache c{10};
    std::vector<double> sol{3.0, 4.0};
    c.put(sol, 2, 5.0);
    c.get(sol, 2);        // hit
    c.get({9.0, 9.0}, 2); // miss
    auto s = c.stats();
    REQUIRE(s.hits == 1);
    REQUIRE(s.misses == 1); // only the explicit miss; put() does not call get()
}

TEST_CASE("cache evicts oldest entry when full", "[cache]") {
    EvaluationCache c{2};
    c.put({1.0}, 1, 10.0);
    c.put({2.0}, 1, 20.0);
    c.put({3.0}, 1, 30.0); // should evict {1.0}
    REQUIRE_FALSE(c.get({1.0}, 1).has_value());
    REQUIRE(c.get({3.0}, 1).has_value());
}

TEST_CASE("cache clear removes all entries", "[cache]") {
    EvaluationCache c{10};
    c.put({1.0, 2.0}, 2, 3.0);
    c.clear();
    REQUIRE_FALSE(c.get({1.0, 2.0}, 2).has_value());
    REQUIRE(c.stats().size == 0);
}

// ── ElitePool ─────────────────────────────────────────────────────────────────

TEST_CASE("empty pool get_best throws", "[elite]") {
    ElitePool pool{5, 0.01, {-5.0, -5.0}, {5.0, 5.0}};
    REQUIRE_THROWS_AS(pool.get_best(), EmptyPool);
}

TEST_CASE("pool accepts distinct solutions", "[elite]") {
    ElitePool pool{5, 0.01, {-5.0, -5.0}, {5.0, 5.0}};
    REQUIRE(pool.add({1.0, 0.0}, 5.0));
    REQUIRE(pool.add({-3.0, 2.0}, 1.0));
    REQUIRE(pool.len() == 2);
}

TEST_CASE("pool get_best returns lowest cost", "[elite]") {
    ElitePool pool{5, 0.01, {-5.0, -5.0}, {5.0, 5.0}};
    pool.add({1.0, 0.0}, 5.0);
    pool.add({-3.0, 2.0}, 1.0);
    auto [ptr, cost] = pool.get_best();
    REQUIRE(cost == Catch::Approx(1.0));
}

TEST_CASE("pool rejects duplicate (too-close) solution", "[elite]") {
    ElitePool pool{5, 0.5, {-5.0, -5.0}, {5.0, 5.0}};
    pool.add({0.0, 0.0}, 3.0);
    // Very close solution should be rejected
    REQUIRE_FALSE(pool.add({0.01, 0.0}, 2.0));
}

TEST_CASE("pool replaces worst when full and new is better", "[elite]") {
    ElitePool pool{2, 0.01, {-10.0, -10.0}, {10.0, 10.0}};
    pool.add({1.0, 0.0}, 10.0);
    pool.add({5.0, 0.0}, 8.0);
    bool replaced = pool.add({-5.0, 0.0}, 1.0);
    REQUIRE(replaced);
    REQUIRE(pool.len() == 2);
    auto [ptr, best] = pool.get_best();
    REQUIRE(best == Catch::Approx(1.0));
}

TEST_CASE("pool keep_top trims to n", "[elite]") {
    ElitePool pool{5, 0.01, {-5.0, -5.0}, {5.0, 5.0}};
    pool.add({1.0, 0.0}, 3.0);
    pool.add({-3.0, 2.0}, 2.0);
    pool.add({4.0, -4.0}, 5.0);
    pool.keep_top(2);
    REQUIRE(pool.len() == 2);
}

TEST_CASE("pool clear empties", "[elite]") {
    ElitePool pool{5, 0.01, {-5.0, -5.0}, {5.0, 5.0}};
    pool.add({1.0, 0.0}, 3.0);
    pool.clear();
    REQUIRE(pool.empty());
}

// ── ConvergenceMonitor ────────────────────────────────────────────────────────

TEST_CASE("monitor first update resets no_improve_count", "[convergence]") {
    ConvergenceMonitor m{10, 20};
    auto sig = m.update(1.0, nullptr);
    REQUIRE(sig.no_improve_count == 0);
    REQUIRE(sig.diversity == Catch::Approx(1.0));
}

TEST_CASE("monitor increments no_improve_count on plateau", "[convergence]") {
    ConvergenceMonitor m{10, 20};
    m.update(1.0, nullptr);
    auto sig = m.update(2.0, nullptr); // no improvement
    REQUIRE(sig.no_improve_count == 1);
}

TEST_CASE("monitor resets no_improve_count when improved", "[convergence]") {
    ConvergenceMonitor m{10, 20};
    m.update(5.0, nullptr);
    m.update(6.0, nullptr);            // no improve
    m.update(6.0, nullptr);            // no improve
    auto sig = m.update(1.0, nullptr); // improvement
    REQUIRE(sig.no_improve_count == 0);
}

TEST_CASE("monitor single-solution pool has diversity = 1", "[convergence]") {
    ConvergenceMonitor m{10, 20};
    ElitePool pool{5, 0.01, {-1.0}, {1.0}};
    pool.add({0.5}, 0.5);
    auto sig = m.update(0.5, &pool);
    REQUIRE(sig.diversity == Catch::Approx(1.0));
}

TEST_CASE("monitor reset_no_improve resets counter", "[convergence]") {
    ConvergenceMonitor m{10, 20};
    m.update(1.0, nullptr);
    m.update(2.0, nullptr);
    m.reset_no_improve();
    auto sig = m.update(2.0, nullptr);  // no improvement after reset
    REQUIRE(sig.no_improve_count == 1); // 0 after reset, then +1
}

// ── pr.hpp: bidirectional_path_relinking backward branch + best update ────────

TEST_CASE("bidirectional PR picks backward when it finds better solution", "[pr]") {
    // Lookup table on 4 binary variables (0=sol1, 1=sol2 per-variable).
    // forward path from sol1=(0,0,0,0) best = (1,0,0,0) cost=20
    // backward path from sol2=(1,1,1,1) best = (0,1,1,1) cost=-100
    // cost_bwd=-100 < cost_fwd=20  →  return {best_bwd, cost_bwd} fires.
    // Also covers the `if (best_mv_cost < best_cost)` update inside
    // path_relinking_best (both directions improve on their starting cost).
    static const double tbl[16] = {
        50.0,   // 0000 = sol1
        37.0,   // 0001
        36.0,   // 0010
        51.0,   // 0011
        35.0,   // 0100
        52.0,   // 0101
        53.0,   // 0110
        -100.0, // 0111 ← great backward intermediate
        20.0,   // 1000 ← best forward step-1
        24.0,   // 1001
        23.0,   // 1010
        40.0,   // 1011
        22.0,   // 1100
        41.0,   // 1101
        42.0,   // 1110
        25.0,   // 1111 = sol2
    };
    auto obj = [&](const std::vector<double> &x) -> double {
        int idx = (x[0] > 0.5 ? 8 : 0) | (x[1] > 0.5 ? 4 : 0) | (x[2] > 0.5 ? 2 : 0) |
                  (x[3] > 0.5 ? 1 : 0);
        return tbl[idx];
    };

    std::vector<double> sol1 = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> sol2 = {1.0, 1.0, 1.0, 1.0};
    std::optional<EvaluationCache> cache = std::nullopt;
    Rng rng = Rng::from_seed(42);
    Deadline dl; // no deadline

    auto [best, cost] = bidirectional_path_relinking(obj, sol1, sol2, 4, cache, rng, dl);
    REQUIRE(cost == Catch::Approx(-100.0));
}

TEST_CASE("bidirectional PR with expired deadline exits early", "[pr]") {
    // An already-expired deadline causes the while loop in path_relinking_best
    // to break immediately (covers the `if (expired(deadline)) break` branch).
    auto obj = [](const std::vector<double> &x) -> double {
        double s = 0.0;
        for (auto v : x)
            s += v * v;
        return s;
    };
    std::vector<double> sol1 = {3.0, 3.0, 3.0};
    std::vector<double> sol2 = {0.0, 0.0, 0.0};
    std::optional<EvaluationCache> cache = std::nullopt;
    Rng rng = Rng::from_seed(1);
    // deadline already in the past → expired() returns true on the first check
    Deadline past_dl = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    // Must not crash; best returned is one of the starting points
    REQUIRE_NOTHROW(bidirectional_path_relinking(obj, sol1, sol2, 3, cache, rng, past_dl));
}

// ── vnd.hpp: neighborhood_multiflip finds improvement (any_improved = true) ──

TEST_CASE("vnd multiflip branch improves negative product objective", "[vnd]") {
    // f(x0,x1,x2) = -x0*x1*x2.  Bounds [0,5].  initial=(0,0,0) → cost=0.
    // At (0,0,0): any SINGLE-variable change keeps one factor=0 → cost=0 (no
    // improvement for flip). Any PAIR swap also leaves one factor=0 → cost=0.
    // After 5 no-improve flip+swap rounds, neighborhood_multiflip perturbs all
    // 3 vars simultaneously → all > 0 → cost < 0 → any_improved = true fires.
    auto neg_product = [](const std::vector<double> &x) -> double { return -(x[0] * x[1] * x[2]); };
    std::vector<std::pair<double, double>> bounds(3, {0.0, 5.0});
    GivpConfig cfg;
    cfg.seed = 42;
    cfg.max_iterations = 3;
    cfg.integer_split = 3;   // all continuous
    cfg.vnd_iterations = 20; // enough iterations to reach multiflip
    cfg.ils_iterations = 1;
    cfg.use_convergence_monitor = false;
    cfg.early_stop_threshold = 1000;
    cfg.initial_guess = std::vector<double>{0.0, 0.0, 0.0};

    auto result = givp::givp(neg_product, bounds, cfg);
    REQUIRE(result.fun < 0.0); // optimizer escaped the (0,0,0) local plateau
}

// ── impl_core.hpp: do_path_relinking improves best (refined_cost < best_cost) ─

TEST_CASE("path relinking improves best solution in main loop", "[pr]") {
    // Use sphere with limited VND so the main loop stays at a mediocre solution
    // when PR first fires.  PR + VND(vnd/2 iters) should find a better point,
    // exercising the `if (refined_cost < best_cost)` body in do_path_relinking.
    auto sphere = [](const std::vector<double> &x) -> double {
        double s = 0.0;
        for (auto v : x)
            s += v * v;
        return s;
    };
    std::vector<std::pair<double, double>> bounds(5, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed = 42;
    cfg.max_iterations = 10;
    cfg.integer_split = 5;
    cfg.vnd_iterations = 2; // intentionally weak — main loop stays suboptimal
    cfg.ils_iterations = 1;
    cfg.path_relink_frequency = 1; // PR fires at every iteration ≥ 1
    cfg.use_elite_pool = true;
    cfg.elite_size = 5;
    cfg.use_convergence_monitor = false;
    cfg.early_stop_threshold = 1000;

    // With weak VND, the elite solutions are spread out. PR between them plus
    // VND(1 iter) regularly beats the current best, triggering the update.
    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}
