// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

#define ANKERL_NANOBENCH_IMPLEMENT

#if __has_include(<nanobench.h>)
#include <nanobench.h>
#elif __has_include("../../build/_deps/nanobench-src/src/include/nanobench.h")
#include "../../build/_deps/nanobench-src/src/include/nanobench.h"
#else
#error "nanobench.h not found. Configure CMake to fetch nanobench first."
#endif

#if __has_include(<givp/givp.hpp>)
#include <givp/givp.hpp>
#elif __has_include("../include/givp/givp.hpp")
#include "../include/givp/givp.hpp"
#else
#error "givp/givp.hpp not found. Open the project through CMake."
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// ── Objective functions
// ───────────────────────────────────────────────────────

static double sphere(const std::vector<double> &x) {
  double s = 0.0;
  for (auto v : x)
    s += v * v;
  return s;
}

static double rosenbrock(const std::vector<double> &x) {
  double s = 0.0;
  for (std::size_t i = 0; i + 1 < x.size(); ++i)
    s += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) +
         (1.0 - x[i]) * (1.0 - x[i]);
  return s;
}

static double rastrigin(const std::vector<double> &x) {
  constexpr double pi = 3.14159265358979323846;
  double n = static_cast<double>(x.size());
  double s = 10.0 * n;
  for (auto xi : x)
    s += xi * xi - 10.0 * std::cos(2.0 * pi * xi);
  return s;
}

// ── Helpers
// ───────────────────────────────────────────────────────────────────

// Deliberately lean config for CI smoke benchmarks.
// These values prioritise speed, not solution quality.
static givp::GivpConfig fast_config(std::uint64_t seed = 42) {
  givp::GivpConfig cfg;
  cfg.seed = seed;
  cfg.max_iterations = 8;
  cfg.vnd_iterations = 15;
  cfg.ils_iterations = 2;
  cfg.use_convergence_monitor = false; // skip overhead
  cfg.path_relink_frequency = 4;
  cfg.integer_split = std::nullopt; // set per call
  return cfg;
}

// ── Benchmarks
// ────────────────────────────────────────────────────────────────

int main() {
  ankerl::nanobench::Bench bench;
  // CI smoke run: 1 warm-up + at most 3 measured iterations per benchmark.
  // Each optimizer call finishes in ~50-150 ms with the lean config above,
  // so the entire benchmark binary completes in well under 30 seconds.
  bench.timeUnit(std::chrono::milliseconds{1}, "ms")
      .warmup(1)
      .minEpochIterations(3)
      .maxEpochTime(std::chrono::milliseconds{5'000});

  // sphere 5D
  {
    std::vector<std::pair<double, double>> bounds(5, {-5.12, 5.12});
    bench.run("sphere_5d", [&] {
      auto cfg = fast_config(42);
      cfg.integer_split = 5; // all continuous
      auto r = givp::givp(sphere, bounds, cfg);
      ankerl::nanobench::doNotOptimizeAway(r.fun);
    });
  }

  // rosenbrock 5D
  {
    std::vector<std::pair<double, double>> bounds(5, {-5.0, 10.0});
    bench.run("rosenbrock_5d", [&] {
      auto cfg = fast_config(42);
      cfg.integer_split = 5;
      auto r = givp::givp(rosenbrock, bounds, cfg);
      ankerl::nanobench::doNotOptimizeAway(r.fun);
    });
  }

  // rastrigin 10D
  {
    std::vector<std::pair<double, double>> bounds(10, {-5.12, 5.12});
    bench.run("rastrigin_10d", [&] {
      auto cfg = fast_config(42);
      cfg.integer_split = 10;
      auto r = givp::givp(rastrigin, bounds, cfg);
      ankerl::nanobench::doNotOptimizeAway(r.fun);
    });
  }

  // rastrigin 30D (heavier — matches Python/Julia/Rust comparison)
  {
    std::vector<std::pair<double, double>> bounds(30, {-5.12, 5.12});
    bench.run("rastrigin_30d", [&] {
      auto cfg = fast_config(42);
      cfg.integer_split = 30;
      auto r = givp::givp(rastrigin, bounds, cfg);
      ankerl::nanobench::doNotOptimizeAway(r.fun);
    });
  }
}
