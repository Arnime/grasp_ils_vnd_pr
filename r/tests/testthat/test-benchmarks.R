# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

test_that("sphere benchmark returns zero at origin", {
  expect_equal(sphere(c(0, 0, 0)), 0)
})

test_that("sphere benchmark is non-negative", {
  vals <- c(
    sphere(c(1, 2, 3)),
    sphere(c(-1, -2, -3)),
    sphere(c(0.5, -0.2, 1.1))
  )
  expect_true(all(vals >= 0))
})

test_that("givp on sphere produces finite objective", {
  res <- givp(sphere, bounds = list(c(-5, 5), c(-5, 5), c(-5, 5)), seed = 10)
  expect_true(res$success)
  expect_true(is.finite(res$fun))
  expect_lte(res$fun, 25)
})
