// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <givp/givp.hpp>

using namespace givp;

// ── Objective functions ───────────────────────────────────────────────────────

static double sphere(const std::vector<double> &x)
{
    double s = 0.0;
    for (auto v : x)
        s += v * v;
    return s;
}

static double rosenbrock(const std::vector<double> &x)
{
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i)
        s += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) +
             (1.0 - x[i]) * (1.0 - x[i]);
    return s;
}

static double rastrigin(const std::vector<double> &x)
{
    constexpr double pi = 3.14159265358979323846;
    double n = static_cast<double>(x.size());
    double s = 10.0 * n;
    for (auto xi : x)
        s += xi * xi - 10.0 * std::cos(2.0 * pi * xi);
    return s;
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

TEST_CASE("sphere 5D finds near-zero minimum", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(5, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed = 42;
    cfg.max_iterations = 50;
    cfg.integer_split = 5; // all continuous

    auto result = givp::givp(sphere, bounds, cfg);

    REQUIRE(result.success);
    REQUIRE(result.fun < 1.0);
    REQUIRE(result.x.size() == 5);
    REQUIRE(result.nfev > 0);
}

TEST_CASE("rosenbrock 5D converges", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 10.0});
    GivpConfig cfg;
    cfg.seed = 7;
    cfg.max_iterations = 80;
    cfg.integer_split = 5;

    auto result = givp::givp(rosenbrock, bounds, cfg);

    REQUIRE(result.success);
    REQUIRE(result.fun < 500.0);
}

TEST_CASE("rastrigin 3D does not crash", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(3, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed = 99;
    cfg.max_iterations = 30;
    cfg.integer_split = 3;

    auto result = givp::givp(rastrigin, bounds, cfg);

    REQUIRE(result.success);
    REQUIRE(result.x.size() == 3);
}

TEST_CASE("maximize direction negates correctly", "[basic]")
{
    // Maximizing sphere means driving x toward bounds, fun > 0
    std::vector<std::pair<double, double>> bounds(3, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed = 1;
    cfg.max_iterations = 30;
    cfg.integer_split = 3;
    cfg.direction = Direction::Maximize;

    auto result = givp::givp(sphere, bounds, cfg);

    REQUIRE(result.success);
    REQUIRE(result.fun > 0.0); // maximum of sphere on bounds > 0
    REQUIRE(result.direction == Direction::Maximize);
}

TEST_CASE("initial guess is accepted", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(3, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed = 5;
    cfg.max_iterations = 30;
    cfg.integer_split = 3;
    cfg.initial_guess = std::vector<double>{0.1, 0.2, 0.3};

    REQUIRE_NOTHROW(givp::givp(sphere, bounds, cfg));
}

TEST_CASE("time limit stops the run early", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(10, {-5.12, 5.12});
    GivpConfig cfg;
    cfg.seed = 3;
    cfg.max_iterations = 10'000; // huge — time limit must fire first
    cfg.integer_split = 10;
    cfg.time_limit = 0.1; // 100 ms

    auto result = givp::givp(sphere, bounds, cfg);
    REQUIRE(result.success);
    // message should mention time
    REQUIRE(result.message.find("time") != std::string::npos);
    REQUIRE(result.termination == TerminationReason::TimeLimitReached);
}

TEST_CASE("result nfev matches evaluations roughly", "[basic]")
{
    std::vector<std::pair<double, double>> bounds(2, {-1.0, 1.0});
    GivpConfig cfg;
    cfg.seed = 0;
    cfg.max_iterations = 5;
    cfg.integer_split = 2;
    cfg.use_cache = false;

    auto result = givp::givp(sphere, bounds, cfg);
    REQUIRE(result.nfev > 0);
    REQUIRE(result.nit >= 1);
    REQUIRE(result.nit <= cfg.max_iterations);
}

TEST_CASE("objective returning infinity is handled", "[basic]")
{
    auto bad_func = [](const std::vector<double> &x) -> double
    {
        if (x[0] > 0)
            return std::numeric_limits<double>::infinity();
        return x[0] * x[0];
    };
    std::vector<std::pair<double, double>> bounds(1, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed = 2;
    cfg.max_iterations = 20;
    cfg.integer_split = 1;

    REQUIRE_NOTHROW(givp::givp(bad_func, bounds, cfg));
}

TEST_CASE("objective throwing exception is handled", "[basic]")
{
    auto throwing_func = [](const std::vector<double> &x) -> double
    {
        if (x[0] > 3.0)
            throw std::runtime_error("deliberate");
        return x[0] * x[0];
    };
    std::vector<std::pair<double, double>> bounds(1, {-5.0, 5.0});
    GivpConfig cfg;
    cfg.seed = 11;
    cfg.max_iterations = 20;
    cfg.integer_split = 1;

    REQUIRE_NOTHROW(givp::givp(throwing_func, bounds, cfg));
}
