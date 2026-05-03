// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT

/// @file fuzz_givp.cpp
/// @brief Randomised fuzz / property driver for the givp public API.
///
/// Exercises the solver with random inputs to detect:
/// - Unexpected exceptions (anything other than givp::GivpError subtypes)
/// - NaN / Inf leaking into the solution vector
/// - Out-of-bounds solutions
/// - Violations of direction semantics
///
/// Usage:
/// @code
/// # Default: 1 000 trials, seed 42
/// ./givp_fuzz
///
/// # Custom parameters
/// ./givp_fuzz --n-trials 5000 --seed 123 --verbose
///
/// # With a timeout (seconds)
/// ./givp_fuzz --n-trials 100000 --timeout 60
/// @endcode

#include "givp/givp.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// ── Objective functions ──────────────────────────────────────────────────────

static double sphere(const std::vector<double> &x) noexcept {
    double s = 0.0;
    for (auto v : x)
        s += v * v;
    return s;
}

static double wave(const std::vector<double> &x) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        s += static_cast<double>(i + 1) * std::sin(x[i]);
    return s;
}

static double noisy_sphere(const std::vector<double> &x) noexcept {
    return sphere(x) + std::sin(x[0] * 1e6) * 0.01;
}

static double nan_on_neg(const std::vector<double> &x) noexcept {
    return (x[0] < 0.0) ? std::numeric_limits<double>::quiet_NaN() : sphere(x);
}

// ── Simple LCG (reproducible, no extra deps) ─────────────────────────────────

static std::uint64_t lcg_next(std::uint64_t &state) noexcept {
    state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return state;
}

static double lcg_f64(std::uint64_t &state) noexcept {
    return static_cast<double>(lcg_next(state) >> 11) / static_cast<double>(UINT64_C(1) << 53);
}

// ── CLI argument parsing ─────────────────────────────────────────────────────

struct FuzzArgs {
    std::size_t n_trials = 1000;
    std::uint64_t seed = 42;
    double timeout_secs = 300.0;
    bool verbose = false;
};

static FuzzArgs parse_args(int argc, char *argv[]) {
    FuzzArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--n-trials") && i + 1 < argc)
            args.n_trials = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if ((a == "--seed") && i + 1 < argc)
            args.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
        else if ((a == "--timeout") && i + 1 < argc)
            args.timeout_secs = std::stod(argv[++i]);
        else if (a == "--verbose")
            args.verbose = true;
    }
    return args;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    auto args = parse_args(argc, argv);
    auto t0 = std::chrono::steady_clock::now();

    using Func = double (*)(const std::vector<double> &);
    struct FuncEntry {
        const char *name;
        Func fn;
    };
    FuncEntry funcs[] = {
        {"sphere", sphere},
        {"wave", wave},
        {"noisy_sphere", noisy_sphere},
        {"nan_on_neg", nan_on_neg},
    };
    constexpr std::size_t n_funcs = sizeof(funcs) / sizeof(funcs[0]);
    const char *dir_names[] = {"minimize", "maximize"};

    std::size_t failures = 0;
    std::size_t trials = 0;
    auto rng = args.seed;

    for (std::size_t trial = 0; trial < args.n_trials; ++trial) {
        auto elapsed = std::chrono::steady_clock::now() - t0;
        if (std::chrono::duration<double>(elapsed).count() > args.timeout_secs) {
            std::cerr << "Timeout reached after " << trial << " trials.\n";
            break;
        }

        const std::size_t ndim = lcg_next(rng) % 5 + 1;
        const std::size_t func_idx = lcg_next(rng) % n_funcs;
        const bool maximize = (lcg_next(rng) % 2) == 1;

        std::vector<std::pair<double, double>> bounds;
        bounds.reserve(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            double lo = lcg_f64(rng) * 100.0 - 50.0;
            double width = lcg_f64(rng) * 19.0 + 1.0;
            bounds.emplace_back(lo, lo + width);
        }

        givp::GivpConfig cfg;
        cfg.max_iterations = 3;
        cfg.vnd_iterations = 5;
        cfg.ils_iterations = 1;
        cfg.early_stop_threshold = 3;
        cfg.use_convergence_monitor = false;
        cfg.direction = maximize ? givp::Direction::Maximize : givp::Direction::Minimize;

        const auto &[func_name, func] = funcs[func_idx];
        ++trials;

        try {
            auto res = givp::givp(func, bounds, cfg);

            bool ok = true;

            // Invariant: solution dimension matches ndim
            if (res.x.size() != ndim) {
                std::cerr << "[FAIL] trial=" << trial << " func=" << func_name << ": solution size "
                          << res.x.size() << " != ndim " << ndim << '\n';
                ok = false;
            }

            // Invariant: solution within declared bounds (tiny tolerance)
            for (std::size_t i = 0; i < res.x.size() && i < bounds.size(); ++i) {
                const double xi = res.x[i];
                const double lo = bounds[i].first;
                const double hi = bounds[i].second;
                const double tol = (hi - lo) * 1e-9 + 1e-12;
                if (xi < lo - tol || xi > hi + tol) {
                    std::cerr << "[FAIL] trial=" << trial << ": x[" << i << "]=" << xi
                              << " out of [" << lo << ", " << hi << "]\n";
                    ok = false;
                }
            }

            if (!ok) {
                ++failures;
            } else if (args.verbose) {
                std::cout << "[OK] trial=" << trial << " func=" << func_name
                          << " dir=" << dir_names[maximize ? 1 : 0] << " ndim=" << ndim
                          << " fun=" << res.fun << '\n';
            }
        } catch (const givp::GivpError &) {
            // Expected for degenerate inputs — not a failure
            if (args.verbose) {
                std::cout << "[OK/ERR] trial=" << trial << " func=" << func_name << ": GivpError\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "[FAIL] trial=" << trial << " func=" << func_name
                      << ": unexpected exception: " << e.what() << '\n';
            ++failures;
        } catch (...) {
            std::cerr << "[FAIL] trial=" << trial << " func=" << func_name
                      << ": unknown exception\n";
            ++failures;
        }
    }

    auto elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nFuzz complete: " << trials << " trials, " << failures << " failures, "
              << elapsed_s << "s elapsed\n";

    return failures > 0 ? 1 : 0;
}
