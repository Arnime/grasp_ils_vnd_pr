// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace givp::detail {

// ── Deadline type ─────────────────────────────────────────────────────────────
using Deadline = std::optional<std::chrono::steady_clock::time_point>;

// ── RNG wrapper (MT19937-64) ──────────────────────────────────────────────────
class Rng {
    std::mt19937_64 engine_;

  public:
    explicit Rng(std::uint64_t seed) : engine_(seed) {}

    static Rng from_seed(std::optional<std::uint64_t> seed) {
        if (seed)
            return Rng(*seed);
        std::random_device rd;
        std::uint64_t s =
            (static_cast<std::uint64_t>(rd()) << 32) | static_cast<std::uint64_t>(rd());
        return Rng(s);
    }

    /// Spawn an independent child RNG seeded from this parent.
    Rng child() {
        std::uniform_int_distribution<std::uint64_t> d;
        return Rng(d(engine_));
    }

    double uniform(double lo, double hi) {
        std::uniform_real_distribution<double> d(lo, hi);
        return d(engine_);
    }

    std::int64_t uniform_int(std::int64_t lo, std::int64_t hi) {
        std::uniform_int_distribution<std::int64_t> d(lo, hi);
        return d(engine_);
    }

    std::size_t uniform_index(std::size_t lo, std::size_t hi) {
        std::uniform_int_distribution<std::size_t> d(lo, hi);
        return d(engine_);
    }

    double random_double() { return uniform(0.0, 1.0); }
};

// ── Pure helpers ──────────────────────────────────────────────────────────────

inline std::size_t get_half(std::size_t num_vars, std::optional<std::size_t> integer_split) {
    return integer_split.value_or(num_vars / 2);
}

inline bool expired(const Deadline &deadline) {
    if (!deadline)
        return false;
    return std::chrono::steady_clock::now() >= *deadline;
}

template <typename F> double safe_evaluate(const F &func, const std::vector<double> &candidate) {
    try {
        double v = func(candidate);
        return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    } catch (...) {
        return std::numeric_limits<double>::infinity();
    }
}

inline void normalize_integer_tail(std::vector<double> &solution, std::size_t half) {
    for (std::size_t i = half; i < solution.size(); ++i)
        solution[i] = std::round(solution[i]);
}

inline double clamp_val(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

} // namespace givp::detail
