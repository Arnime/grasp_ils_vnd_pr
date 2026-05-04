# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Build elite pool
#' @keywords internal
make_elite_pool <- function(max_size = 7L, direction = "minimize") {
  structure(
    list(
      max_size = as.integer(max_size),
      direction = direction,
      items = list()
    ),
    class = "givp_elite_pool"
  )
}

#' Add solution to elite pool
#' @keywords internal
elite_add <- function(pool, x, value) {
  pool$items[[length(pool$items) + 1L]] <- list(x = x, value = value)

  ord <- order(
    vapply(pool$items, function(it) it$value, numeric(1L)),
    decreasing = identical(pool$direction, "maximize")
  )
  pool$items <- pool$items[ord]
  if (length(pool$items) > pool$max_size) {
    pool$items <- pool$items[seq_len(pool$max_size)]
  }
  pool
}

#' Get best from elite pool
#' @keywords internal
elite_best <- function(pool) {
  if (length(pool$items) == 0L) {
    return(NULL)
  }
  pool$items[[1L]]
}
