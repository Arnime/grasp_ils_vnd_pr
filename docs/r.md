<!-- SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior -->
<!-- SPDX-License-Identifier: MIT -->

# R

`givp` is available as an R package with a functional API (`givp`) and an
object-oriented API (`GIVPOptimizer`) based on R6.

## Installation

From a local clone:

```bash
R CMD INSTALL r
```

## Quick start

```r
library(givp)

sphere <- function(x) sum(x * x)
res <- givp(
  func = sphere,
  bounds = list(c(-5.12, 5.12), c(-5.12, 5.12)),
  seed = 42
)

print(res)
```

## Functional API

```r
givp(
  func,
  bounds,
  num_vars = NULL,
  minimize = NULL,
  direction = NULL,
  config = NULL,
  initial_guess = NULL,
  iteration_callback = NULL,
  seed = NULL,
  verbose = FALSE
)
```

## Configuration

```r
cfg <- givp_config(max_iterations = 50L, direction = "minimize")
res <- givp(sphere, bounds = list(c(-5, 5), c(-5, 5)), config = cfg)
```

## Object-oriented API

```r
opt <- GIVPOptimizer$new(
  func = sphere,
  bounds = list(c(-5, 5), c(-5, 5)),
  seed = 42
)
res <- opt$optimize()
```

## Error model

Errors follow an `rlang::abort()` hierarchy rooted at `givp_error`, with
specialized subclasses such as:

- `givp_error_invalid_bounds`
- `givp_error_invalid_config`
- `givp_error_invalid_initial_guess`
- `givp_error_invalid_objective`

## Testing

```bash
Rscript -e "testthat::test_dir('r/tests/testthat')"
```
