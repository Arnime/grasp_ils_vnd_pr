# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

test_that("givp returns structured result", {
  f <- function(x) sum(x * x)
  res <- givp(f, bounds = list(c(-5, 5), c(-5, 5)), seed = 42)

  expect_s3_class(res, "givp_result")
  expect_equal(length(res$x), 2)
  expect_true(is.numeric(res$fun))
  expect_true(res$nfev > 0)
  expect_true(is.character(res$message))
})

test_that("givp handles maximize direction", {
  f <- function(x) -sum(x * x)
  res <- givp(f, bounds = list(c(-3, 3), c(-3, 3)), direction = "maximize", seed = 1)
  expect_equal(res$direction, "maximize")
  expect_true(is.numeric(res$fun))
})

test_that("givp accepts minimize flag", {
  f <- function(x) sum(x * x)
  res <- givp(f, bounds = list(c(-3, 3)), minimize = TRUE)
  expect_equal(res$direction, "minimize")
})

test_that("givp validates objective", {
  expect_error(givp(1, bounds = list(c(-1, 1))), class = "givp_error_invalid_objective")
})

test_that("givp validates bounds empty", {
  expect_error(givp(function(x) sum(x), bounds = list()), class = "givp_error_invalid_bounds")
})

test_that("givp validates non-finite bounds", {
  expect_error(
    givp(function(x) sum(x), bounds = list(c(-Inf, 1))),
    class = "givp_error_invalid_bounds"
  )
})

test_that("givp validates inverted bounds", {
  expect_error(
    givp(function(x) sum(x), bounds = list(c(2, -2))),
    class = "givp_error_invalid_bounds"
  )
})

test_that("givp validates initial guess length", {
  expect_error(
    givp(function(x) sum(x * x), bounds = list(c(-1, 1), c(-1, 1)), initial_guess = c(0)),
    class = "givp_error_invalid_initial_guess"
  )
})

test_that("givp validates initial guess inside bounds", {
  expect_error(
    givp(function(x) sum(x * x), bounds = list(c(-1, 1)), initial_guess = c(2)),
    class = "givp_error_invalid_initial_guess"
  )
})

test_that("GIVPOptimizer wrapper runs", {
  opt <- GIVPOptimizer$new(
    func = function(x) sum(x * x),
    bounds = list(c(-2, 2), c(-2, 2)),
    seed = 123
  )
  res <- opt$optimize()
  expect_s3_class(res, "givp_result")
})
