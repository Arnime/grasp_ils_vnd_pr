# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Evaluate objective with optional cache and nfev tracking
#' @keywords internal
evaluate_candidate <- function(func, x, cache, state) {
  cached <- cache_get(cache, x)
  if (!is.null(cached)) {
    return(cached)
  }

  value <- safe_eval(func, x)
  state$nfev <- state$nfev + 1L
  cache_set(cache, x, value)
  value
}

#' GRASP construction phase
#' @keywords internal
grasp_construct <- function(
  func,
  bounds,
  config,
  direction,
  cache,
  state,
  alpha = NULL
) {
  n <- nrow(bounds)
  n_candidates <- as.integer(config$num_candidates_per_step)
  alpha_value <- if (is.null(alpha)) config$alpha else alpha

  # Pre-generate all candidate points
  candidates <- matrix(NA_real_, nrow = n_candidates, ncol = n)
  for (k in seq_len(n_candidates)) {
    x <- stats::runif(n, min = bounds[, 1], max = bounds[, 2])
    x <- normalize_integer_tail(x, config$integer_split)
    x <- clamp_to_bounds(x, bounds)
    candidates[k, ] <- x
  }

  # Evaluate candidates: parallel when n_workers > 1 on Unix, sequential
  # otherwise. Parallel evaluation bypasses the cache because forked R
  # processes cannot share mutable environments.
  use_parallel <- isTRUE(config$n_workers > 1L) && .Platform$OS.type == "unix"
  if (use_parallel) {
    cand_list <- lapply(seq_len(n_candidates), function(k) candidates[k, ])
    vals_list <- parallel::mclapply(
      cand_list,
      function(x) safe_eval(func, x),
      mc.cores = config$n_workers
    )
    values <- vapply(vals_list, function(v) {
      if (is.numeric(v) && length(v) == 1L) v else Inf
    }, numeric(1))
    state$nfev <- state$nfev + n_candidates
  } else {
    values <- rep(NA_real_, n_candidates)
    for (k in seq_len(n_candidates)) {
      values[k] <- evaluate_candidate(func, candidates[k, ], cache, state)
    }
  }

  finite_idx <- which(is.finite(values))
  if (length(finite_idx) == 0L) {
    x <- stats::runif(n, min = bounds[, 1], max = bounds[, 2])
    x <- normalize_integer_tail(x, config$integer_split)
    x <- clamp_to_bounds(x, bounds)
    return(list(x = x, value = evaluate_candidate(func, x, cache, state)))
  }

  vals <- values[finite_idx]
  lo <- min(vals)
  hi <- max(vals)
  if (identical(direction, "maximize")) {
    cutoff <- hi - alpha_value * (hi - lo)
    rcl_rel <- which(vals >= cutoff)
  } else {
    cutoff <- lo + alpha_value * (hi - lo)
    rcl_rel <- which(vals <= cutoff)
  }
  rcl_rel <- if (length(rcl_rel) == 0L) {
    which(vals == if (identical(direction, "maximize")) hi else lo)
  } else {
    rcl_rel
  }

  pick <- sample(finite_idx[rcl_rel], 1L)
  list(x = as.numeric(candidates[pick, ]), value = values[pick])
}
