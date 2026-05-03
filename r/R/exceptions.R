# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Raise a GIVP condition
#' @keywords internal
givp_abort <- function(message, subclass = "givp_error") {
  rlang::abort(message, .subclass = subclass)
}

#' Raise invalid bounds error
#' @keywords internal
abort_invalid_bounds <- function(message) {
  givp_abort(message, "givp_error_invalid_bounds")
}

#' Raise invalid config error
#' @keywords internal
abort_invalid_config <- function(message) {
  givp_abort(message, "givp_error_invalid_config")
}

#' Raise invalid initial guess error
#' @keywords internal
abort_invalid_initial_guess <- function(message) {
  givp_abort(message, "givp_error_invalid_initial_guess")
}
