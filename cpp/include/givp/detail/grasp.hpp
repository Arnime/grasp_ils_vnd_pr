// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "cache.hpp"
#include "helpers.hpp"

namespace givp::detail {

// ── Adaptive alpha ────────────────────────────────────────────────────────────

inline double get_current_alpha(std::size_t iter_idx,
                                  std::size_t max_iterations,
                                  double alpha_min, double alpha_max,
                                  bool adaptive, double alpha) {
    if (!adaptive) return alpha;
    double progress =
        static_cast<double>(iter_idx) /
        static_cast<double>(std::max(max_iterations, std::size_t{1}));
    return alpha_min + (alpha_max - alpha_min) * progress;
}

// ── Cached evaluation ─────────────────────────────────────────────────────────

template <typename F>
double evaluate_with_cache(const std::vector<double>& candidate,
                            const F& func,
                            std::optional<EvaluationCache>& cache,
                            std::size_t half) {
    if (cache) {
        auto cached = cache->get(candidate, half);
        if (cached) return *cached;
        double cost = safe_evaluate(func, candidate);
        cache->put(candidate, half, cost);
        return cost;
    }
    return safe_evaluate(func, candidate);
}

// ── RCL selection ─────────────────────────────────────────────────────────────

template <typename RngT>
static std::size_t select_from_rcl(const std::vector<double>& costs,
                                    double alpha, RngT& rng) {
    double min_cost = *std::min_element(costs.begin(), costs.end());
    double max_cost = *std::max_element(costs.begin(), costs.end());
    double threshold = min_cost + alpha * (max_cost - min_cost);

    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < costs.size(); ++i)
        if (costs[i] <= threshold) candidates.push_back(i);

    if (candidates.empty()) return 0;
    return candidates[rng.uniform_index(0, candidates.size() - 1)];
}

// ── Candidate builders ────────────────────────────────────────────────────────

template <typename RngT>
static double sample_integer_from_bounds(double lo, double hi, RngT& rng) {
    std::int64_t lo_i = static_cast<std::int64_t>(std::ceil(lo));
    std::int64_t hi_i = static_cast<std::int64_t>(std::floor(hi));
    if (lo_i > hi_i) return std::round((lo + hi) / 2.0);
    return static_cast<double>(rng.uniform_int(lo_i, hi_i));
}

template <typename RngT>
static std::vector<double> build_random_candidate(
    std::size_t num_vars, std::size_t half,
    const std::vector<double>& lower, const std::vector<double>& upper,
    RngT& rng) {
    std::vector<double> sol(num_vars);
    for (std::size_t i = 0; i < half; ++i)
        sol[i] = rng.uniform(lower[i], upper[i]);
    for (std::size_t i = half; i < num_vars; ++i)
        sol[i] = sample_integer_from_bounds(lower[i], upper[i], rng);
    return sol;
}

template <typename RngT>
static std::vector<double> build_heuristic_candidate(
    std::size_t num_vars, std::size_t half,
    const std::vector<double>& lower, const std::vector<double>& upper,
    RngT& rng) {
    std::vector<double> sol(num_vars);
    for (std::size_t i = 0; i < half; ++i) {
        double mid   = (lower[i] + upper[i]) / 2.0;
        double span  = upper[i] - lower[i];
        double noise = rng.uniform(-0.15, 0.15) * span;
        sol[i] = clamp_val(mid + noise, lower[i], upper[i]);
    }
    for (std::size_t i = half; i < num_vars; ++i)
        sol[i] = sample_integer_from_bounds(lower[i], upper[i], rng);
    return sol;
}

// ── GRASP construction ────────────────────────────────────────────────────────

template <typename F, typename RngT>
std::pair<std::vector<double>, double> construct_grasp(
    std::size_t num_vars,
    const std::vector<double>& lower, const std::vector<double>& upper,
    const F& func,
    const std::vector<double>* initial_guess,
    double alpha, std::size_t half, std::size_t num_candidates,
    std::optional<EvaluationCache>& cache, RngT& rng,
    const Deadline& deadline) {

    std::vector<std::vector<double>> candidates;
    std::vector<double> costs;
    candidates.reserve(num_candidates);
    costs.reserve(num_candidates);

    // Optional initial guess as first candidate
    if (initial_guess) {
        auto sol = *initial_guess;
        normalize_integer_tail(sol, half);
        double cost = evaluate_with_cache(sol, func, cache, half);
        candidates.push_back(std::move(sol));
        costs.push_back(cost);
    }

    // One heuristic candidate
    if (candidates.size() < num_candidates) {
        auto sol = build_heuristic_candidate(num_vars, half, lower, upper, rng);
        normalize_integer_tail(sol, half);
        double cost = evaluate_with_cache(sol, func, cache, half);
        candidates.push_back(std::move(sol));
        costs.push_back(cost);
    }

    // Fill rest with random candidates
    while (candidates.size() < num_candidates) {
        if (expired(deadline)) break;
        auto sol = build_random_candidate(num_vars, half, lower, upper, rng);
        normalize_integer_tail(sol, half);
        double cost = evaluate_with_cache(sol, func, cache, half);
        candidates.push_back(std::move(sol));
        costs.push_back(cost);
    }

    std::size_t idx = select_from_rcl(costs, alpha, rng);
    double selected_cost = costs[idx];
    auto selected_sol    = std::move(candidates[idx]);
    return {std::move(selected_sol), selected_cost};
}

} // namespace givp::detail
