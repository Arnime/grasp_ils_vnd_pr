// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <cctype>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"

namespace givp {

/// Reason the optimizer stopped.
enum class TerminationReason {
    Converged,
    MaxIterationsReached,
    TimeLimitReached,
    EarlyStop,
    NoFeasible,
    Unknown,
};

inline TerminationReason termination_from_message(const std::string &msg) {
    std::string lo(msg.size(), '\0');
    for (std::size_t i = 0; i < msg.size(); ++i)
        lo[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(msg[i])));

    if (lo.find("converged") != std::string::npos)
        return TerminationReason::Converged;
    if (lo.find("max") != std::string::npos && lo.find("iteration") != std::string::npos)
        return TerminationReason::MaxIterationsReached;
    if (lo.find("time") != std::string::npos)
        return TerminationReason::TimeLimitReached;
    if (lo.find("early") != std::string::npos || lo.find("stagnation") != std::string::npos)
        return TerminationReason::EarlyStop;
    if (lo.find("no feasible") != std::string::npos || lo.find("no_feasible") != std::string::npos)
        return TerminationReason::NoFeasible;
    return TerminationReason::Unknown;
}

/// Result of a single optimization run.
struct OptimizeResult {
    /// Best solution vector found.
    std::vector<double> x;
    /// Best objective value (in the original direction).
    double fun = std::numeric_limits<double>::infinity();
    /// Number of main-loop iterations executed.
    std::size_t nit = 0;
    /// Total number of objective-function evaluations.
    std::size_t nfev = 0;
    /// True when `fun` is finite.
    bool success = false;
    /// Human-readable termination message.
    std::string message;
    /// Direction used for this run.
    Direction direction = Direction::Minimize;
    /// Structured termination reason.
    TerminationReason termination = TerminationReason::Unknown;
    /// Extra metadata (string key → string value).
    std::unordered_map<std::string, std::string> meta;
};

} // namespace givp
