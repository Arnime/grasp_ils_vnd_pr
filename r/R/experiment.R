# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Multi-seed sweep for reproducible experiments
#'
#' @param func Objective function.
#' @param bounds Variable bounds.
#' @param n_runs Number of seeds/runs.
#' @param config Optional `GIVPConfig`.
#' @param num_vars Optional number of variables.
#' @param direction Optional optimization direction.
#' @param verbose Verbose mode.
#' @return Named list of `givp_result` indexed by seed.
#' @export
seed_sweep <- function(func,
                       bounds,
                       n_runs = 30L,
                       config = NULL,
                       num_vars = NULL,
                       direction = NULL,
                       verbose = FALSE) {
  if (is.null(config)) {
    config <- givp_config()
  }
  n_runs <- as.integer(n_runs)
  if (n_runs < 1L) {
    abort_invalid_config("n_runs must be >= 1")
  }

  out <- vector("list", n_runs)
  names(out) <- as.character(seq(0L, n_runs - 1L))
  for (seed in seq(0L, n_runs - 1L)) {
    out[[seed + 1L]] <- givp(
      func = func,
      bounds = bounds,
      num_vars = num_vars,
      direction = direction,
      config = config$clone(deep = TRUE),
      seed = seed,
      verbose = verbose
    )
  }
  out
}

#' Summarize a seed sweep result set
#'
#' @param sweep_results Result list from [seed_sweep()].
#' @return List with aggregate statistics.
#' @export
sweep_summary <- function(sweep_results) {
  if (!is.list(sweep_results) || length(sweep_results) == 0L) {
    abort_invalid_config("sweep_results must be a non-empty list")
  }

  fun <- vapply(sweep_results, function(x) as.numeric(x$fun), numeric(1L))
  nfev <- vapply(sweep_results, function(x) as.numeric(x$nfev), numeric(1L))
  nit <- vapply(sweep_results, function(x) as.numeric(x$nit), numeric(1L))

  list(
    fun_mean = mean(fun),
    fun_std = stats::sd(fun),
    fun_min = min(fun),
    fun_max = max(fun),
    nfev_mean = mean(nfev),
    nit_mean = mean(nit)
  )
}
