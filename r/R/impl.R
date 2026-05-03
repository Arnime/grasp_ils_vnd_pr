# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Native fallback implementation for R port
#' @keywords internal
run_givp_native <- function(func, bounds, config, direction, seed = NULL) {
  b <- normalize_bounds(bounds)
  n <- nrow(b)

  if (!is.null(seed)) {
    set.seed(seed)
  } else if (!is.null(config$seed)) {
    set.seed(config$seed)
  }

  best_x <- rep(NA_real_, n)
  best_value <- if (identical(direction, "maximize")) -Inf else Inf
  nfev <- 0L

  for (i in seq_len(config$max_iterations)) {
    x <- stats::runif(n, min = b[, 1], max = b[, 2])
    value <- tryCatch(as.numeric(func(x)), error = function(e) Inf)
    nfev <- nfev + 1L

    if (!is.finite(value)) {
      next
    }

    improved <- if (identical(direction, "maximize")) value > best_value else value < best_value
    if (improved) {
      best_value <- value
      best_x <- x
    }
  }

  success <- all(is.finite(best_x)) && is.finite(best_value)
  msg <- if (success) "max iterations reached" else "no feasible solution found"
  make_result(best_x, best_value, config$max_iterations, nfev, success, msg, direction)
}
