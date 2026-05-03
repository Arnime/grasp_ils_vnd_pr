# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

test_that("givp improves from random baseline on sphere", {
  b <- list(c(-5, 5), c(-5, 5), c(-5, 5))
  base_x <- c(2, -3, 1)
  base_val <- sphere(base_x)
  cfg <- givp_config(
    max_iterations = 8L,
    vnd_iterations = 20L,
    ils_iterations = 3L,
    num_candidates_per_step = 10L
  )
  res <- givp(sphere, bounds = b, seed = 42, config = cfg)

  expect_true(res$success)
  expect_true(is.finite(res$fun))
  expect_lte(res$fun, base_val)
})

test_that("integer tail is rounded when integer_split is set", {
  cfg <- givp_config(max_iterations = 15L, integer_split = 1L, seed = 1)
  res <- givp(sphere, bounds = list(c(-5, 5), c(0, 4), c(0, 4)), config = cfg)

  expect_true(res$success)
  expect_equal(res$x[2], round(res$x[2]))
  expect_equal(res$x[3], round(res$x[3]))
})

test_that("seed_sweep runs n_runs experiments", {
  cfg <- givp_config(
    max_iterations = 5L,
    vnd_iterations = 15L,
    ils_iterations = 2L,
    num_candidates_per_step = 8L
  )
  runs <- seed_sweep(sphere, bounds = list(c(-3, 3), c(-3, 3)), n_runs = 5L, config = cfg)
  expect_length(runs, 5L)
  expect_true(all(vapply(runs, inherits, logical(1), "givp_result")))
})

test_that("sweep_summary returns expected keys", {
  cfg <- givp_config(
    max_iterations = 5L,
    vnd_iterations = 15L,
    ils_iterations = 2L,
    num_candidates_per_step = 8L
  )
  runs <- seed_sweep(sphere, bounds = list(c(-3, 3), c(-3, 3)), n_runs = 4L, config = cfg)
  sm <- sweep_summary(runs)
  expect_true(all(c("fun_mean", "fun_std", "fun_min", "fun_max", "nfev_mean", "nit_mean") %in% names(sm)))
  expect_true(is.finite(sm$fun_mean))
})