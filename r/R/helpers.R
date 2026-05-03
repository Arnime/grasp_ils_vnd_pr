# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Normalize bounds representation
#' @keywords internal
normalize_bounds <- function(bounds, num_vars = NULL) {
  if (is.null(bounds) || length(bounds) == 0L) {
    abort_invalid_bounds("bounds must be non-empty")
  }

  if (is.list(bounds) && all(vapply(bounds, length, integer(1)) == 2L)) {
    b <- do.call(rbind, bounds)
  } else if (is.matrix(bounds) && ncol(bounds) == 2L) {
    b <- bounds
  } else {
    abort_invalid_bounds("bounds must be a list of (lo, hi) pairs or a two-column matrix")
  }

  if (!is.null(num_vars) && nrow(b) != as.integer(num_vars)) {
    abort_invalid_bounds("num_vars does not match bounds length")
  }

  if (any(!is.finite(b))) {
    abort_invalid_bounds("bounds must be finite")
  }
  if (any(b[, 2] <= b[, 1])) {
    abort_invalid_bounds("upper bounds must be greater than lower bounds")
  }

  b
}
