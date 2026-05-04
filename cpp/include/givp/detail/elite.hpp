// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../exceptions.hpp"

namespace givp::detail {

/// Diversity-aware elite pool of best solutions.
class ElitePool {
    std::size_t max_size_;
    double min_distance_;
    std::vector<std::pair<std::vector<double>, double>> pool_;
    std::vector<double> range_;

    double relative_distance(const std::vector<double> &a, const std::vector<double> &b) const {
        double n = static_cast<double>(a.size());
        double total = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            total += std::abs(a[i] - b[i]) / range_[i];
        return total / n;
    }

  public:
    ElitePool(std::size_t max_size, double min_distance, const std::vector<double> &lower,
              const std::vector<double> &upper)
        : max_size_(max_size), min_distance_(min_distance) {
        range_.resize(lower.size());
        for (std::size_t i = 0; i < lower.size(); ++i)
            range_[i] = std::max(upper[i] - lower[i], 1e-12);
        pool_.reserve(max_size + 1);
    }

    bool add(std::vector<double> solution, double cost) {
        for (const auto &[existing, _] : pool_) {
            if (relative_distance(solution, existing) < min_distance_)
                return false;
        }

        if (pool_.size() < max_size_) {
            pool_.emplace_back(std::move(solution), cost);
            std::sort(pool_.begin(), pool_.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });
            return true;
        }

        // Replace worst if new is better
        if (cost < pool_.back().second) {
            pool_.pop_back();
            pool_.emplace_back(std::move(solution), cost);
            std::sort(pool_.begin(), pool_.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });
            return true;
        }
        return false;
    }

    std::pair<const std::vector<double> *, double> get_best() const {
        if (pool_.empty())
            throw EmptyPool("elite pool is empty");
        return {&pool_.front().first, pool_.front().second};
    }

    const std::vector<std::pair<std::vector<double>, double>> &get_all() const { return pool_; }

    std::size_t len() const { return pool_.size(); }
    bool empty() const { return pool_.empty(); }
    void clear() { pool_.clear(); }

    void keep_top(std::size_t n) {
        if (pool_.size() > n)
            pool_.resize(n);
    }
};

} // namespace givp::detail
