// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "../config.hpp"
#include "../exceptions.hpp"
#include "../result.hpp"
#include "cache.hpp"
#include "convergence.hpp"
#include "elite.hpp"
#include "grasp.hpp"
#include "helpers.hpp"
#include "ils.hpp"
#include "pr.hpp"
#include "vnd.hpp"

namespace givp::detail
{

    inline std::pair<std::vector<double>, std::vector<double>> validate_bounds(
        const std::vector<std::pair<double, double>> &bounds,
        const std::optional<std::vector<double>> &initial_guess)
    {

        if (bounds.empty())
            throw InvalidBounds("bounds cannot be empty");

        std::vector<double> lower, upper;
        lower.reserve(bounds.size());
        upper.reserve(bounds.size());

        for (std::size_t i = 0; i < bounds.size(); ++i)
        {
            double lo = bounds[i].first, hi = bounds[i].second;
            if (lo >= hi)
                throw InvalidBounds("lower >= upper at index " + std::to_string(i) +
                                    ": " + std::to_string(lo) + " >= " +
                                    std::to_string(hi));
            if (!std::isfinite(lo) || !std::isfinite(hi))
                throw InvalidBounds("non-finite bound at index " + std::to_string(i));
            lower.push_back(lo);
            upper.push_back(hi);
        }

        if (initial_guess)
        {
            if (initial_guess->size() != bounds.size())
                throw InvalidInitialGuess(
                    "expected " + std::to_string(bounds.size()) + " values, got " +
                    std::to_string(initial_guess->size()));
            for (std::size_t i = 0; i < initial_guess->size(); ++i)
            {
                double v = (*initial_guess)[i];
                if (v < bounds[i].first || v > bounds[i].second)
                    throw InvalidInitialGuess(
                        "value " + std::to_string(v) + " out of bounds [" +
                        std::to_string(bounds[i].first) + ", " +
                        std::to_string(bounds[i].second) + "] at index " +
                        std::to_string(i));
            }
        }
        return {std::move(lower), std::move(upper)};
    }

    template <typename F>
    static void do_path_relinking(
        const F &func,
        const ElitePool &elite_pool,
        std::vector<double> &best_solution, double &best_cost,
        std::size_t half,
        const std::vector<double> &lower, const std::vector<double> &upper,
        std::size_t vnd_iterations,
        std::optional<EvaluationCache> &cache, Rng &rng,
        const Deadline &deadline)
    {

        const auto &all = elite_pool.get_all();
        std::size_t max_pairs = std::min(std::size_t{3}, all.size());

        for (std::size_t i = 0; i < max_pairs; ++i)
        {
            for (std::size_t j = i + 1; j < std::min(all.size(), i + 4); ++j)
            {
                if (expired(deadline))
                    return;

                auto [pr_sol, pr_cost] = bidirectional_path_relinking(
                    func, all[i].first, all[j].first, half, cache, rng, deadline);

                double refined_cost =
                    local_search_vnd(func, pr_sol, pr_cost, half, lower, upper,
                                     vnd_iterations / 2, cache, rng, deadline);

                if (refined_cost < best_cost)
                {
                    best_cost = refined_cost;
                    best_solution = std::move(pr_sol);
                }
            }
        }
    }

    /// Main optimizer loop.
    template <typename F>
    OptimizeResult run(F &&func,
                       const std::vector<std::pair<double, double>> &bounds,
                       GivpConfig config)
    {
        config.validate();

        // C++17: structured bindings cannot be captured by lambdas; use
        // regular named variables so the multi-worker lambda can capture them.
        auto bounds_vecs = validate_bounds(bounds, config.initial_guess);
        auto lower = std::move(bounds_vecs.first);
        auto upper = std::move(bounds_vecs.second);
        std::size_t num_vars = bounds.size();

        // When integer_split is not set, treat all variables as continuous
        // (half == num_vars → no integer rounding applied).
        std::size_t half = get_half(
            num_vars,
            config.integer_split.has_value() ? config.integer_split
                                             : std::optional<std::size_t>{num_vars});

        bool is_maximize = (config.direction == Direction::Maximize);

        // Atomic counter — wrapped lambda is called from a single thread only,
        // but std::atomic makes the intent explicit.
        std::atomic<std::size_t> nfev{0};
        auto wrapped = [&](const std::vector<double> &x) -> double
        {
            nfev.fetch_add(1, std::memory_order_relaxed);
            double v = func(x);
            return is_maximize ? -v : v;
        };

        auto rng = Rng::from_seed(config.seed);
        std::optional<EvaluationCache> cache;
        if (config.use_cache)
            cache.emplace(config.cache_size);

        ElitePool elite_pool{config.elite_size, 0.05, lower, upper};
        std::optional<ConvergenceMonitor> conv_monitor;
        if (config.use_convergence_monitor)
            conv_monitor.emplace(20, 50);

        Deadline deadline;
        if (config.time_limit > 0.0)
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(config.time_limit));

        // ── Initialise best solution ─────────────────────────────────────────────
        std::vector<double> best_solution;
        double best_cost;

        if (config.initial_guess)
        {
            best_solution = *config.initial_guess;
            normalize_integer_tail(best_solution, half);
            best_cost = evaluate_with_cache(best_solution, wrapped, cache, half);
        }
        else
        {
            auto child = rng.child();
            auto [sol, cost] = construct_grasp(num_vars, lower, upper, wrapped,
                                               nullptr, config.alpha, half,
                                               config.num_candidates_per_step,
                                               cache, child, deadline);
            best_solution = std::move(sol);
            best_cost = cost;
        }

        if (config.use_elite_pool)
            elite_pool.add(best_solution, best_cost);

        std::size_t stagnation = 0;
        std::size_t iterations_executed = 0;
        std::string message;

        // ── Main loop ─────────────────────────────────────────────────────────────
        for (std::size_t iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            if (expired(deadline))
            {
                message = "time limit reached";
                break;
            }
            iterations_executed = iteration + 1;

            double alpha = get_current_alpha(
                iteration, config.max_iterations,
                config.alpha_min, config.alpha_max,
                config.adaptive_alpha, config.alpha);

            auto child = rng.child();
            const std::vector<double> *ig =
                (iteration == 0 && config.initial_guess)
                    ? &(*config.initial_guess)
                    : nullptr;

            std::vector<double> candidate;
            double ils_cost = std::numeric_limits<double>::infinity();

            if (config.n_workers <= 1)
            {
                // Single-worker path keeps cache behavior identical to prior releases.
                auto grasp_result = construct_grasp(
                    num_vars, lower, upper, wrapped, ig, alpha, half,
                    config.num_candidates_per_step, cache, child, deadline);
                candidate = std::move(grasp_result.first);

                double grasp_eval =
                    evaluate_with_cache(candidate, wrapped, cache, half);
                double vnd_cost =
                    local_search_vnd(wrapped, candidate, grasp_eval, half,
                                     lower, upper, config.vnd_iterations,
                                     cache, child, deadline);

                ils_cost =
                    ils_search(wrapped, candidate, vnd_cost, half,
                               lower, upper, config.ils_iterations,
                               config.vnd_iterations, config.perturbation_strength,
                               cache, child, deadline);
            }
            else
            {
                struct WorkerResult
                {
                    std::vector<double> candidate;
                    double cost;
                };

                std::vector<std::future<WorkerResult>> futures;
                futures.reserve(config.n_workers);

                for (std::size_t worker = 0; worker < config.n_workers; ++worker)
                {
                    auto worker_rng = rng.child();
                    const std::vector<double> *worker_ig = (worker == 0) ? ig : nullptr;

                    futures.push_back(std::async(
                        std::launch::async,
                        [&, worker_rng = std::move(worker_rng), worker_ig]() mutable -> WorkerResult
                        {
                            // Keep per-worker cache local to avoid shared mutable state.
                            std::optional<EvaluationCache> local_cache;

                            auto grasp_result = construct_grasp(
                                num_vars, lower, upper, wrapped, worker_ig, alpha, half,
                                config.num_candidates_per_step, local_cache, worker_rng, deadline);
                            std::vector<double> local_candidate = std::move(grasp_result.first);

                            double grasp_eval =
                                evaluate_with_cache(local_candidate, wrapped, local_cache, half);
                            double vnd_cost =
                                local_search_vnd(wrapped, local_candidate, grasp_eval, half,
                                                 lower, upper, config.vnd_iterations,
                                                 local_cache, worker_rng, deadline);
                            double local_cost =
                                ils_search(wrapped, local_candidate, vnd_cost, half,
                                           lower, upper, config.ils_iterations,
                                           config.vnd_iterations, config.perturbation_strength,
                                           local_cache, worker_rng, deadline);

                            return WorkerResult{std::move(local_candidate), local_cost};
                        }));
                }

                for (auto &f : futures)
                {
                    auto wr = f.get();
                    if (wr.cost < ils_cost)
                    {
                        ils_cost = wr.cost;
                        candidate = std::move(wr.candidate);
                    }
                }
            }

            // Update best
            if (ils_cost < best_cost)
            {
                best_cost = ils_cost;
                best_solution = candidate;
                stagnation = 0;
            }
            else
            {
                ++stagnation;
            }

            if (config.use_elite_pool)
                elite_pool.add(candidate, ils_cost);

            // Convergence monitor — single update per iteration
            std::optional<std::size_t> no_improve_count;
            if (conv_monitor)
            {
                auto sig = conv_monitor->update(best_cost, &elite_pool);
                no_improve_count = sig.no_improve_count;
                if (sig.should_restart)
                {
                    elite_pool.keep_top(2);
                    conv_monitor->reset_no_improve();
                    no_improve_count = 0;
                    stagnation = 0;
                    if (cache)
                        cache->clear();
                }
            }

            // Path relinking
            if (config.use_elite_pool && iteration > 0 &&
                iteration % config.path_relink_frequency == 0 &&
                elite_pool.len() >= 2)
            {
                do_path_relinking(wrapped, elite_pool, best_solution, best_cost,
                                  half, lower, upper, config.vnd_iterations,
                                  cache, child, deadline);
            }

            // Stagnation restart
            if (stagnation > config.max_iterations / 4)
            {
                auto child2 = rng.child();
                auto [rsol, rcost0] = construct_grasp(
                    num_vars, lower, upper, wrapped, nullptr, alpha, half,
                    config.num_candidates_per_step, cache, child2, deadline);
                double rcost =
                    local_search_vnd(wrapped, rsol, rcost0, half,
                                     lower, upper, config.vnd_iterations,
                                     cache, child2, deadline);
                rcost = ils_search(wrapped, rsol, rcost, half,
                                   lower, upper, config.ils_iterations,
                                   config.vnd_iterations, config.perturbation_strength,
                                   cache, child2, deadline);
                if (rcost < best_cost)
                {
                    best_cost = rcost;
                    best_solution = std::move(rsol);
                }
                stagnation = 0;
            }

            // Early stop — reuse the same convergence signal from this iteration.
            if (no_improve_count.has_value() &&
                *no_improve_count >= config.early_stop_threshold)
            {
                message = "early stop due to stagnation";
                break;
            }

            if (iteration == config.max_iterations - 1)
                message = "max iterations reached";
        }

        // ── Build result ─────────────────────────────────────────────────────────
        double final_cost = is_maximize ? -best_cost : best_cost;

        OptimizeResult result;
        result.x = std::move(best_solution);
        result.fun = final_cost;
        result.nit = iterations_executed;
        result.nfev = nfev.load(std::memory_order_relaxed);
        result.success = std::isfinite(final_cost);
        result.message = message;
        result.direction = config.direction;
        result.termination = termination_from_message(message);
        return result;
    }

} // namespace givp::detail
