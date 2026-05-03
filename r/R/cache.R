# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Build an evaluation cache
#' @keywords internal
make_eval_cache <- function(max_size = 10000L) {
  env <- new.env(parent = emptyenv())
  env$store <- new.env(hash = TRUE, parent = emptyenv())
  env$order <- character()
  env$max_size <- as.integer(max_size)
  env
}

#' Build cache key from numeric vector
#' @keywords internal
cache_key <- function(x) {
  paste(formatC(x, digits = 12, format = "fg", flag = "#"), collapse = "|")
}

#' Get cached value
#' @keywords internal
cache_get <- function(cache, x) {
  if (is.null(cache)) return(NULL)
  key <- cache_key(x)
  if (exists(key, envir = cache$store, inherits = FALSE)) {
    value <- get(key, envir = cache$store, inherits = FALSE)
    cache$order <- c(cache$order[cache$order != key], key)
    return(value)
  }
  NULL
}

#' Set cache value with LRU eviction
#' @keywords internal
cache_set <- function(cache, x, value) {
  if (is.null(cache)) return(invisible(NULL))
  key <- cache_key(x)
  assign(key, value, envir = cache$store)
  cache$order <- c(cache$order[cache$order != key], key)

  while (length(cache$order) > cache$max_size) {
    oldest <- cache$order[1]
    rm(list = oldest, envir = cache$store)
    cache$order <- cache$order[-1]
  }
  invisible(NULL)
}