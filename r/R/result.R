# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Build an optimization result object
#' @keywords internal
make_result <- function(x, fun, nit, nfev, success, message, direction) {
  structure(
    list(
      x = x,
      fun = fun,
      nit = nit,
      nfev = nfev,
      success = success,
      message = message,
      direction = direction,
      termination = infer_termination_reason(message)
    ),
    class = "givp_result"
  )
}

#' Infer termination reason from message
#' @keywords internal
infer_termination_reason <- function(message) {
  m <- tolower(message)
  if (grepl("converg", m)) return("converged")
  if (grepl("time", m)) return("time_limit_reached")
  if (grepl("early|stagn", m)) return("early_stop")
  if (grepl("feasible", m)) return("no_feasible")
  if (grepl("iter|max", m)) return("max_iterations_reached")
  "unknown"
}

#' Print optimization result
#' @param x A `givp_result` object.
#' @param ... Unused.
#' @export
print.givp_result <- function(x, ...) {
  cat("<givp_result>\n")
  cat("  success   :", x$success, "\n")
  cat("  fun       :", format(x$fun, digits = 8), "\n")
  cat("  nit       :", x$nit, "\n")
  cat("  nfev      :", x$nfev, "\n")
  cat("  direction :", x$direction, "\n")
  cat("  message   :", x$message, "\n")
  invisible(x)
}
