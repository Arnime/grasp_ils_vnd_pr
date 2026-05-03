# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

test_that("givp_config returns GIVPConfig", {
  cfg <- givp_config()
  expect_true(inherits(cfg, "GIVPConfig"))
})

test_that("config validates max_iterations", {
  expect_error(givp_config(max_iterations = 0L), class = "givp_error_invalid_config")
})

test_that("config validates alpha range", {
  expect_error(givp_config(alpha = -0.1), class = "givp_error_invalid_config")
  expect_error(givp_config(alpha = 1.1), class = "givp_error_invalid_config")
})

test_that("config validates alpha interval ordering", {
  expect_error(givp_config(alpha_min = 0.5, alpha_max = 0.4), class = "givp_error_invalid_config")
})

test_that("config validates direction", {
  expect_error(givp_config(direction = "sideways"), class = "givp_error_invalid_config")
})

test_that("config can be customized", {
  cfg <- givp_config(max_iterations = 5L, direction = "maximize")
  expect_equal(cfg$max_iterations, 5L)
  expect_equal(cfg$direction, "maximize")
  lst <- cfg$as_list()
  expect_true("max_iterations" %in% names(lst))
  expect_true("direction" %in% names(lst))
})
