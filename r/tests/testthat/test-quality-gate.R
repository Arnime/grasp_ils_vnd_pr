# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

random_baseline_min <- function(fn, bounds, n_samples = 80L, seed = 1L) {
  set.seed(seed)
  mat <- vapply(
    bounds,
    function(b) runif(n_samples, min = b[1], max = b[2]),
    FUN.VALUE = numeric(n_samples)
  )
  vals <- apply(mat, 1L, fn)
  min(vals)
}

test_that("givp on sphere beats random baseline in median across seeds", {
  bounds <- list(c(-5, 5), c(-5, 5), c(-5, 5), c(-5, 5))
  seeds <- 1:7
  cfg <- givp_config(
    max_iterations = 12L,
    vnd_iterations = 20L,
    ils_iterations = 3L,
    num_candidates_per_step = 10L,
    path_relink_frequency = 4L
  )

  givp_vals <- vapply(
    seeds,
    function(s) {
      res <- givp(sphere, bounds = bounds, seed = s, config = cfg)
      expect_true(is.finite(res$fun))
      res$fun
    },
    FUN.VALUE = numeric(1)
  )

  baseline_vals <- vapply(
    seeds,
    function(s) random_baseline_min(sphere, bounds, n_samples = 80L, seed = s),
    FUN.VALUE = numeric(1)
  )

  expect_lt(stats::median(givp_vals), stats::median(baseline_vals))
})

test_that("givp on rosenbrock stays finite and beats random baseline on most seeds", {
  bounds <- list(c(-2, 2), c(-2, 2), c(-2, 2), c(-2, 2))
  seeds <- 1:9
  cfg <- givp_config(
    max_iterations = 16L,
    vnd_iterations = 25L,
    ils_iterations = 4L,
    num_candidates_per_step = 12L,
    path_relink_frequency = 4L
  )

  givp_vals <- vapply(
    seeds,
    function(s) {
      res <- givp(rosenbrock, bounds = bounds, seed = s, config = cfg)
      expect_true(is.finite(res$fun))
      res$fun
    },
    FUN.VALUE = numeric(1)
  )

  baseline_vals <- vapply(
    seeds,
    function(s) random_baseline_min(rosenbrock, bounds, n_samples = 120L, seed = s),
    FUN.VALUE = numeric(1)
  )

  wins <- sum(givp_vals < baseline_vals)
  expect_gte(wins, floor(length(seeds) / 2) + 1L)
})
