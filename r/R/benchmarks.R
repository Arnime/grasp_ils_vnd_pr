# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Sphere benchmark function
#' @param x Numeric vector.
#' @return Scalar objective value.
#' @export
sphere <- function(x) {
  sum(x * x)
}
