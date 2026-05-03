# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

#' Native fallback implementation for R port
#' @keywords internal
run_givp_native <- function(func, bounds, config, direction, seed = NULL) {
  b <- normalize_bounds(bounds) # nolint: object_usage_linter
  n <- nrow(b)
  set_seed_if_needed(seed, config$seed)

  state <- new.env(parent = emptyenv())
  state$nfev <- 0L
  state$nit <- 0L

  cache <- if (isTRUE(config$use_cache)) {
    make_eval_cache(config$cache_size)
  } else {
    NULL
  }
  elite <- if (isTRUE(config$use_elite_pool)) {
    make_elite_pool(config$elite_size, direction)
  } else {
    NULL
  }
  conv <- if (isTRUE(config$use_convergence_monitor)) {
    make_convergence_monitor()
  } else {
    NULL
  }

  if (!is.null(config$initial_guess)) {
    best_x <- normalize_integer_tail(
      as.numeric(config$initial_guess),
      config$integer_split
    )
    best_x <- clamp_to_bounds(best_x, b)
    best_value <- evaluate_candidate(func, best_x, cache, state)
  } else {
    init <- grasp_construct(func, b, config, direction, cache, state)
    best_x <- init$x
    best_value <- init$value
  }

  if (!is.null(elite) && is.finite(best_value)) {
    elite <- elite_add(elite, best_x, best_value)
  }

  stagnation <- 0L
  message <- "max iterations reached"
  t0 <- proc.time()[[3L]]

  for (iter in seq_len(config$max_iterations)) {
    state$nit <- iter

    reached_time_limit <- config$time_limit > 0 &&
      (proc.time()[[3L]] - t0) >= config$time_limit
    if (reached_time_limit) {
      message <- "time limit reached"
      break
    }

    alpha <- if (isTRUE(config$adaptive_alpha)) {
      frac <- (iter - 1L) / max(1L, config$max_iterations - 1L)
      config$alpha_min + frac * (config$alpha_max - config$alpha_min)
    } else {
      config$alpha
    }

    candidate <- grasp_construct(
      func,
      b,
      config,
      direction,
      cache,
      state,
      alpha
    )
    candidate <- vnd_search(
      func,
      candidate$x,
      candidate$value,
      b,
      config,
      direction,
      cache,
      state
    )
    candidate <- ils_search(
      func,
      candidate$x,
      candidate$value,
      b,
      config,
      direction,
      cache,
      state
    )

    candidate_improved <- is.finite(candidate$value) &&
      is_improvement(candidate$value, best_value, direction)
    if (candidate_improved) {
      best_x <- candidate$x
      best_value <- candidate$value
      stagnation <- 0L
    } else {
      stagnation <- stagnation + 1L
    }

    if (!is.null(elite) && is.finite(candidate$value)) {
      elite <- elite_add(elite, candidate$x, candidate$value)
    }

    should_relink <- !is.null(elite) && length(elite$items) >= 2L &&
      (iter %% as.integer(config$path_relink_frequency) == 0L)
    if (should_relink) {
      pr <- path_relink(
        func,
        elite$items[[1L]]$x,
        elite$items[[2L]]$x,
        b,
        config,
        direction,
        cache,
        state
      )
      pr_improved <- is.finite(pr$value) &&
        is_improvement(pr$value, best_value, direction)
      if (pr_improved) {
        best_x <- pr$x
        best_value <- pr$value
        stagnation <- 0L
      }
    }

    if (!is.null(conv)) {
      conv <- convergence_update(conv, best_value, direction)
      if (conv$no_improve >= as.integer(config$early_stop_threshold)) {
        message <- "early stop due to stagnation"
        break
      }
    }

    if (stagnation > max(5L, as.integer(config$max_iterations %/% 4L))) {
      restart <- grasp_construct(func, b, config, direction, cache, state)
      restart_improved <- is.finite(restart$value) &&
        is_improvement(restart$value, best_value, direction)
      if (restart_improved) {
        best_x <- restart$x
        best_value <- restart$value
      }
      stagnation <- 0L
    }
  }

  success <- all(is.finite(best_x)) && is.finite(best_value)
  if (!success) {
    message <- "no feasible solution found"
    best_x <- if (exists("best_x", inherits = FALSE)) {
      best_x
    } else {
      rep(NA_real_, n)
    }
  }

  make_result( # nolint: object_usage_linter
    x = best_x,
    fun = best_value,
    nit = as.integer(state$nit),
    nfev = as.integer(state$nfev),
    success = success,
    message = message,
    direction = direction
  )
}
