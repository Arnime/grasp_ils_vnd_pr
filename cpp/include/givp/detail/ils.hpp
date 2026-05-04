// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "cache.hpp"
#include "grasp.hpp"
#include "helpers.hpp"
#include "vnd.hpp"

namespace givp::detail {

template <typename RngT>
static std::vector<double> perturb_solution(const std::vector<double> &solution, std::size_t half,
                                            std::size_t strength, const std::vector<double> &lower,
                                            const std::vector<double> &upper, RngT &rng) {

    std::size_t n = solution.size();
    std::size_t num_perturb = std::max(std::size_t{1}, std::min(strength, n / 5));
    std::vector<double> perturbed = solution;
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), std::size_t{0});

    for (std::size_t i = 0; i < num_perturb; ++i) {
        std::size_t j = rng.uniform_index(i, n - 1);
        std::swap(indices[i], indices[j]);
    }

    for (std::size_t k = 0; k < num_perturb; ++k) {
        std::size_t idx = indices[k];
        if (idx >= half) {
            double step = std::max(static_cast<double>(strength) / 2.0, 1.0);
            double delta = rng.uniform(-step, step);
            perturbed[idx] = std::round(clamp_val(perturbed[idx] + delta, lower[idx], upper[idx]));
        } else {
            double span = upper[idx] - lower[idx];
            double delta = rng.uniform(-0.15, 0.15) * span;
            perturbed[idx] = clamp_val(perturbed[idx] + delta, lower[idx], upper[idx]);
        }
    }

    normalize_integer_tail(perturbed, half);
    return perturbed;
}

/// Iterated Local Search.
template <typename F, typename RngT>
double ils_search(const F &func, std::vector<double> &solution, double current_cost,
                  std::size_t half, const std::vector<double> &lower,
                  const std::vector<double> &upper, std::size_t ils_iterations,
                  std::size_t vnd_iterations, std::size_t perturbation_strength,
                  std::optional<EvaluationCache> &cache, RngT &rng, const Deadline &deadline) {

    double best_cost = current_cost;
    std::vector<double> best_sol = solution;

    for (std::size_t i = 0; i < ils_iterations; ++i) {
        if (expired(deadline))
            break;

        // Progressive adaptive strength
        double progress =
            static_cast<double>(i) / static_cast<double>(std::max(ils_iterations, std::size_t{1}));
        std::size_t effective_strength =
            std::max(perturbation_strength,
                     static_cast<std::size_t>(static_cast<double>(perturbation_strength) *
                                              (1.0 + progress)));

        auto candidate = perturb_solution(best_sol, half, effective_strength, lower, upper, rng);
        double perturbed_cost = evaluate_with_cache(candidate, func, cache, half);
        double vnd_cost = local_search_vnd(func, candidate, perturbed_cost, half, lower, upper,
                                           vnd_iterations, cache, rng, deadline);

        if (vnd_cost < best_cost) {
            best_cost = vnd_cost;
            best_sol = candidate;
        } else if (vnd_cost < best_cost * 1.25 && rng.random_double() < 0.1) {
            // Accept slightly worse with 10% probability (diversification)
            best_sol = candidate;
            best_cost = vnd_cost;
        }
    }

    solution = std::move(best_sol);
    return best_cost;
}

} // namespace givp::detail
