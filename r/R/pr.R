# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Path relinking between two elite solutions
#' @keywords internal
path_relink_forward <- function(
  func,
  xa,
  xb,
  bounds,
  config,
  direction,
  cache,
  state
) {
  steps <- max(2L, as.integer(config$path_relink_frequency))
  best_x <- xa
  best_v <- evaluate_candidate(func, xa, cache, state) # nolint: object_usage_linter

  for (s in seq_len(steps)) {
    t <- s / steps
    x <- (1 - t) * xa + t * xb
    x <- normalize_integer_tail(x, config$integer_split) # nolint: object_usage_linter
    x <- clamp_to_bounds(x, bounds) # nolint: object_usage_linter
    v <- evaluate_candidate(func, x, cache, state) # nolint: object_usage_linter
    if (is.finite(v) &&
      is_improvement(v, best_v, direction)) { # nolint: object_usage_linter
      best_x <- x
      best_v <- v
    }
  }

  list(x = best_x, value = best_v)
}

#' Path relinking from xb to xa
#' @keywords internal
path_relink_backward <- function(
  func,
  xa,
  xb,
  bounds,
  config,
  direction,
  cache,
  state
) {
  path_relink_forward(func, xb, xa, bounds, config, direction, cache, state)
}

#' Bidirectional path relinking combining forward and backward paths
#' @keywords internal
path_relink_bidirectional <- function(
  func,
  xa,
  xb,
  bounds,
  config,
  direction,
  cache,
  state
) {
  forward <- path_relink_forward(
    func,
    xa,
    xb,
    bounds,
    config,
    direction,
    cache,
    state
  )
  backward <- path_relink_backward(
    func,
    xa,
    xb,
    bounds,
    config,
    direction,
    cache,
    state
  )

  if (is.finite(backward$value) &&
    is_improvement(backward$value, forward$value, direction)) { # nolint: object_usage_linter
    return(backward)
  }
  forward
}

#' Path relinking between two elite solutions
#' @keywords internal
path_relink <- function(func, xa, xb, bounds, config, direction, cache, state) {
  path_relink_bidirectional(
    func,
    xa,
    xb,
    bounds,
    config,
    direction,
    cache,
    state
  )
}
