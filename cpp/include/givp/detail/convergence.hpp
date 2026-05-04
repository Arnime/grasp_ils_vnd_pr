// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "elite.hpp"

namespace givp::detail {

struct ConvergenceSignal {
    bool should_restart;
    bool should_intensify;
    double diversity;
    std::size_t no_improve_count;
};

/// Tracks convergence and triggers restarts / intensification.
class ConvergenceMonitor {
    std::size_t window_size_;
    std::size_t restart_threshold_;
    std::vector<double> history_;
    std::size_t no_improve_count_ = 0;
    double best_ever_ = std::numeric_limits<double>::infinity();
    std::vector<double> diversity_scores_;

    static double compute_diversity(const ElitePool &pool) {
        const auto &sols = pool.get_all();
        if (sols.size() < 2)
            return 1.0;
        double total = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < sols.size(); ++i) {
            for (std::size_t j = i + 1; j < sols.size(); ++j) {
                double dist = 0.0;
                for (std::size_t k = 0; k < sols[i].first.size(); ++k) {
                    double d = sols[i].first[k] - sols[j].first[k];
                    dist += d * d;
                }
                total += std::sqrt(dist);
                ++count;
            }
        }
        return total / static_cast<double>(count);
    }

  public:
    ConvergenceMonitor(std::size_t window_size, std::size_t restart_threshold)
        : window_size_(window_size), restart_threshold_(restart_threshold) {}

    ConvergenceSignal update(double current_cost, const ElitePool *elite_pool = nullptr) {
        if (current_cost < best_ever_) {
            best_ever_ = current_cost;
            no_improve_count_ = 0;
        } else {
            ++no_improve_count_;
        }

        history_.push_back(current_cost);
        if (history_.size() > window_size_)
            history_.erase(history_.begin());

        double diversity = elite_pool ? compute_diversity(*elite_pool) : 1.0;
        diversity_scores_.push_back(diversity);

        bool should_restart = no_improve_count_ >= restart_threshold_;
        bool should_intensify = no_improve_count_ >= restart_threshold_ / 2 && diversity < 0.5;

        return {should_restart, should_intensify, diversity, no_improve_count_};
    }

    void reset_no_improve() { no_improve_count_ = 0; }
};

} // namespace givp::detail
