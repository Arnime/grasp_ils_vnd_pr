// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
//
// givp/givp.hpp — single include for the GIVP C++17 optimizer.
//
// Quick start:
//
//   #include <givp/givp.hpp>
//
//   auto sphere = [](const std::vector<double>& x) {
//       double s = 0; for (auto v : x) s += v*v; return s;
//   };
//   std::vector<std::pair<double,double>> bounds(5, {-5.12, 5.12});
//   givp::OptimizeResult r = givp::givp(sphere, bounds);
//   std::cout << "best: " << r.fun << "\n";
//
#pragma once

#include "config.hpp"
#include "detail/impl_core.hpp"
#include "result.hpp"


namespace givp {

/// Run the GRASP-ILS-VND with Path Relinking optimizer.
///
/// @tparam F  Callable with signature `double(const std::vector<double>&)`.
/// @param func     Objective function to minimize (or maximize).
/// @param bounds   Variable bounds as a vector of (lower, upper) pairs.
/// @param config   Algorithm configuration (optional, defaults are reasonable).
/// @return         OptimizeResult with best solution, objective value, and
/// stats.
/// @throws InvalidBounds, InvalidInitialGuess, InvalidConfig on bad input.
template <typename F>
OptimizeResult givp(F &&func,
                    const std::vector<std::pair<double, double>> &bounds,
                    GivpConfig config = {}) {
  return detail::run(std::forward<F>(func), bounds, std::move(config));
}

} // namespace givp
