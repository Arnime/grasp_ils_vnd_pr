// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <givp/givp.hpp>
#include <givp/detail/cache.hpp>
#include <givp/detail/convergence.hpp>
#include <givp/detail/elite.hpp>

using namespace givp;
using namespace givp::detail;

// ── EvaluationCache ───────────────────────────────────────────────────────────

TEST_CASE("cache is empty on construction", "[cache]") {
    EvaluationCache c{10};
    auto s = c.stats();
    REQUIRE(s.hits   == 0);
    REQUIRE(s.misses == 0);
    REQUIRE(s.size   == 0);
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
    REQUIRE(*v == Catch::Approx(42.0));
}

TEST_CASE("cache hit rate is tracked", "[cache]") {
    EvaluationCache c{10};
    std::vector<double> sol{3.0, 4.0};
    c.put(sol, 2, 5.0);
    c.get(sol, 2); // hit
    c.get({9.0, 9.0}, 2); // miss
    auto s = c.stats();
    REQUIRE(s.hits   == 1);
    REQUIRE(s.misses == 2); // one from put+get above, one explicit miss
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
    m.update(6.0, nullptr); // no improve
    m.update(6.0, nullptr); // no improve
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
    auto sig = m.update(2.0, nullptr); // no improvement after reset
    REQUIRE(sig.no_improve_count == 1); // 0 after reset, then +1
}
