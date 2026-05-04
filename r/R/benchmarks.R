# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Sphere benchmark function
#' @param x Numeric vector.
#' @return Scalar objective value.
#' @export
sphere <- function(x) {
  sum(x * x)
}

#' Rosenbrock benchmark function
#' @param x Numeric vector.
#' @return Scalar objective value.
#' @export
rosenbrock <- function(x) {
  if (length(x) < 2L) {
    return(0)
  }
  sum(100 * (x[-1L] - x[-length(x)]^2)^2 + (1 - x[-length(x)])^2)
}

#' Rastrigin benchmark function
#' @param x Numeric vector.
#' @return Scalar objective value.
#' @export
rastrigin <- function(x) {
  n <- length(x)
  10 * n + sum(x^2 - 10 * cos(2 * pi * x))
}
