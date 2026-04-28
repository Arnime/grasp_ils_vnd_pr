// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "cache.hpp"
#include "grasp.hpp"
#include "helpers.hpp"

namespace givp::detail {

static constexpr std::size_t MAX_PR_VARS = 25;

/// Greedy (best-move) path relinking from source toward target.
template <typename F>
static std::pair<std::vector<double>, double> path_relinking_best(
    const F& func,
    const std::vector<double>& source, const std::vector<double>& target,
    std::vector<std::size_t> diff_indices,
    std::optional<EvaluationCache>& cache, std::size_t half,
    const Deadline& deadline) {

    std::vector<double> current = source;
    std::vector<double> best    = current;
    double best_cost = evaluate_with_cache(current, func, cache, half);

    while (!diff_indices.empty()) {
        if (expired(deadline)) break;

        std::size_t best_idx_pos  = 0;
        double      best_mv_cost  = std::numeric_limits<double>::infinity();
        double      best_mv_val   = 0.0;

        for (std::size_t pos = 0; pos < diff_indices.size(); ++pos) {
            std::size_t idx = diff_indices[pos];
            double old      = current[idx];
            current[idx]    = target[idx];
            double cost     = evaluate_with_cache(current, func, cache, half);
            if (cost < best_mv_cost) {
                best_mv_cost  = cost;
                best_idx_pos  = pos;
                best_mv_val   = target[idx];
            }
            current[idx] = old;
        }

        std::size_t chosen_idx = diff_indices[best_idx_pos];
        diff_indices.erase(diff_indices.begin() +
                           static_cast<std::ptrdiff_t>(best_idx_pos));
        current[chosen_idx] = best_mv_val;

        if (best_mv_cost < best_cost) {
            best_cost = best_mv_cost;
            best      = current;
        }
    }
    return {std::move(best), best_cost};
}

/// Bidirectional path relinking between two solutions.
template <typename F, typename RngT>
std::pair<std::vector<double>, double> bidirectional_path_relinking(
    const F& func,
    const std::vector<double>& sol1, const std::vector<double>& sol2,
    std::size_t half,
    std::optional<EvaluationCache>& cache, RngT& rng,
    const Deadline& deadline) {

    std::size_t n = sol1.size();
    std::vector<std::size_t> diff_indices;
    for (std::size_t i = 0; i < n; ++i)
        if (std::abs(sol1[i] - sol2[i]) > 1e-12)
            diff_indices.push_back(i);

    if (diff_indices.empty()) {
        double cost = evaluate_with_cache(sol1, func, cache, half);
        return {sol1, cost};
    }

    // Limit to the MAX_PR_VARS most different variables
    if (diff_indices.size() > MAX_PR_VARS) {
        std::sort(diff_indices.begin(), diff_indices.end(),
                  [&](std::size_t a, std::size_t b) {
                      return std::abs(sol1[b] - sol2[b]) <
                             std::abs(sol1[a] - sol2[a]);
                  });
        diff_indices.resize(MAX_PR_VARS);
    }

    // Shuffle order (Fisher-Yates)
    for (std::size_t i = diff_indices.size() - 1; i > 0; --i) {
        std::size_t j = rng.uniform_index(0, i);
        std::swap(diff_indices[i], diff_indices[j]);
    }

    auto [best_fwd, cost_fwd] =
        path_relinking_best(func, sol1, sol2, diff_indices, cache, half, deadline);
    auto [best_bwd, cost_bwd] =
        path_relinking_best(func, sol2, sol1, diff_indices, cache, half, deadline);

    return (cost_fwd <= cost_bwd)
               ? std::make_pair(std::move(best_fwd), cost_fwd)
               : std::make_pair(std::move(best_bwd), cost_bwd);
}

} // namespace givp::detail
