// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "exceptions.hpp"

namespace givp {

/// Optimization direction.
enum class Direction {
    Minimize,
    Maximize,
};

/// Algorithm hyper-parameters.  Defaults mirror the Rust implementation.
struct GivpConfig {
    std::size_t max_iterations         = 100;
    double      alpha                  = 0.12;
    std::size_t vnd_iterations         = 200;
    std::size_t ils_iterations         = 10;
    std::size_t perturbation_strength  = 4;
    bool        use_elite_pool         = true;
    std::size_t elite_size             = 7;
    std::size_t path_relink_frequency  = 8;
    bool        adaptive_alpha         = true;
    double      alpha_min              = 0.08;
    double      alpha_max              = 0.18;
    std::size_t num_candidates_per_step = 20;
    bool        use_cache              = true;
    std::size_t cache_size             = 10'000;
    std::size_t early_stop_threshold   = 80;
    bool        use_convergence_monitor = true;
    std::size_t n_workers              = 1;
    double      time_limit             = 0.0;
    Direction   direction              = Direction::Minimize;

    std::optional<std::size_t>            integer_split;
    std::optional<std::size_t>            group_size;
    std::optional<std::vector<double>>    initial_guess;
    std::optional<std::uint64_t>          seed;
    bool verbose = false;

    /// Validate all numeric ranges; throws InvalidConfig on failure.
    void validate() const {
        if (max_iterations == 0)
            throw InvalidConfig("max_iterations must be > 0");
        if (vnd_iterations == 0)
            throw InvalidConfig("vnd_iterations must be > 0");
        if (ils_iterations == 0)
            throw InvalidConfig("ils_iterations must be > 0");
        if (elite_size == 0)
            throw InvalidConfig("elite_size must be > 0");
        if (path_relink_frequency == 0)
            throw InvalidConfig("path_relink_frequency must be > 0");
        if (alpha < 0.0 || alpha > 1.0)
            throw InvalidConfig("alpha must be in [0.0, 1.0]");
        if (alpha_min < 0.0 || alpha_min > 1.0)
            throw InvalidConfig("alpha_min must be in [0.0, 1.0]");
        if (alpha_max < 0.0 || alpha_max > 1.0)
            throw InvalidConfig("alpha_max must be in [0.0, 1.0]");
        if (alpha_min > alpha_max)
            throw InvalidConfig("alpha_min must be <= alpha_max");
    }
};

} // namespace givp
