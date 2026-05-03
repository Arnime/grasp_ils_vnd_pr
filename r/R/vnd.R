# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' VND local search phase
#' @keywords internal
vnd_search <- function(
  func,
  x0,
  value0,
  bounds,
  config,
  direction,
  cache,
  state
) {
  x_best <- x0
  v_best <- value0
  n <- length(x_best)
  step <- 0.05 * (bounds[, 2] - bounds[, 1])

  for (iter in seq_len(as.integer(config$vnd_iterations))) {
    idx <- ((iter - 1L) %% n) + 1L
    x_try <- x_best
    delta <- sample(c(-1, 1), 1L) * step[idx]
    x_try[idx] <- x_try[idx] + delta
    x_try <- normalize_integer_tail(x_try, config$integer_split)
    x_try <- clamp_to_bounds(x_try, bounds)
    v_try <- evaluate_candidate(func, x_try, cache, state)

    if (is.finite(v_try) && is_improvement(v_try, v_best, direction)) {
      x_best <- x_try
      v_best <- v_try
    }
  }

  list(x = x_best, value = v_best)
}
