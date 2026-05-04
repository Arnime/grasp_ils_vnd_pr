# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Normalize bounds representation
#' @keywords internal
normalize_bounds <- function(bounds, num_vars = NULL) {
  if (is.null(bounds) || length(bounds) == 0L) {
    abort_invalid_bounds("bounds must be non-empty") # nolint: object_usage_linter
  }

  if (is.list(bounds) && all(vapply(bounds, length, integer(1)) == 2L)) {
    b <- do.call(rbind, bounds)
  } else if (is.matrix(bounds) && ncol(bounds) == 2L) {
    b <- bounds
  } else {
    abort_invalid_bounds( # nolint: object_usage_linter
      "bounds must be a list of (lo, hi) pairs or a two-column matrix"
    )
  }

  if (!is.null(num_vars) && nrow(b) != as.integer(num_vars)) {
    abort_invalid_bounds("num_vars does not match bounds length") # nolint: object_usage_linter
  }

  if (any(!is.finite(b))) {
    abort_invalid_bounds("bounds must be finite") # nolint: object_usage_linter
  }
  if (any(b[, 2] <= b[, 1])) {
    abort_invalid_bounds( # nolint: object_usage_linter
      "upper bounds must be greater than lower bounds"
    )
  }

  b
}

#' Clamp a numeric vector to bounds
#' @keywords internal
clamp_to_bounds <- function(x, bounds) {
  pmax(bounds[, 1], pmin(bounds[, 2], x))
}

#' Normalize integer tail according to integer_split
#' @keywords internal
normalize_integer_tail <- function(x, integer_split = NULL) {
  if (is.null(integer_split)) {
    return(x)
  }
  n <- length(x)
  half <- as.integer(integer_split)
  half <- max(0L, min(half, n))
  if (half < n) {
    x[(half + 1L):n] <- round(x[(half + 1L):n])
  }
  x
}

#' Evaluate objective with finite fallback
#' @keywords internal
safe_eval <- function(func, x) {
  out <- tryCatch(as.numeric(func(x)), error = function(e) Inf)
  if (!is.finite(out)) Inf else out
}

#' Determine whether value improves current best
#' @keywords internal
is_improvement <- function(value, best_value, direction) {
  if (identical(direction, "maximize")) {
    value > best_value
  } else {
    value < best_value
  }
}

#' Better of two objective values
#' @keywords internal
better_value <- function(a, b, direction) {
  if (identical(direction, "maximize")) max(a, b) else min(a, b)
}

#' Set RNG seed from explicit or config seed
#' @keywords internal
set_seed_if_needed <- function(seed = NULL, config_seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed) # NOSONAR: reproducibility seed, not a security use case
  } else if (!is.null(config_seed)) {
    set.seed(config_seed) # NOSONAR: reproducibility seed, not a security use case
  }
}
