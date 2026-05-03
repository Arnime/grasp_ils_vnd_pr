# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Path relinking between two elite solutions
#' @keywords internal
path_relink <- function(func, xa, xb, bounds, config, direction, cache, state) {
  steps <- 5L
  best_x <- xa
  best_v <- evaluate_candidate(func, xa, cache, state)

  for (s in seq_len(steps)) {
    t <- s / steps
    x <- (1 - t) * xa + t * xb
    x <- normalize_integer_tail(x, config$integer_split)
    x <- clamp_to_bounds(x, bounds)
    v <- evaluate_candidate(func, x, cache, state)
    if (is.finite(v) && is_improvement(v, best_v, direction)) {
      best_x <- x
      best_v <- v
    }
  }

  list(x = best_x, value = best_v)
}