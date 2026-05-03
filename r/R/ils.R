# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' ILS perturbation and re-search phase
#' @keywords internal
ils_search <- function(func, x0, value0, bounds, config, direction, cache, state) {
  x_best <- x0
  v_best <- value0
  n <- length(x_best)
  k <- min(as.integer(config$perturbation_strength), n)

  for (iter in seq_len(as.integer(config$ils_iterations))) {
    idx <- sample.int(n, size = max(1L, k), replace = FALSE)
    x_try <- x_best
    x_try[idx] <- x_try[idx] + stats::rnorm(length(idx), sd = 0.1 * (bounds[idx, 2] - bounds[idx, 1]))
    x_try <- normalize_integer_tail(x_try, config$integer_split)
    x_try <- clamp_to_bounds(x_try, bounds)

    v_try <- evaluate_candidate(func, x_try, cache, state)
    refined <- vnd_search(func, x_try, v_try, bounds, config, direction, cache, state)

    if (is.finite(refined$value) && is_improvement(refined$value, v_best, direction)) {
      x_best <- refined$x
      v_best <- refined$value
    }
  }

  list(x = x_best, value = v_best)
}