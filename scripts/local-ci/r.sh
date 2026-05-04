#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

echo "[r] Install dependencies"
Rscript -e "deps <- c('R6','rlang','testthat'); missing <- deps[!vapply(deps, requireNamespace, logical(1), quietly = TRUE)]; if (length(missing)) install.packages(missing, repos='https://cloud.r-project.org', dependencies=NA)"
Rscript -e "if (!requireNamespace('testthat', quietly=TRUE)) stop('testthat installation failed')"

echo "[r] Install package"
R CMD INSTALL r

echo "[r] Tests"
Rscript -e "testthat::test_dir('r/tests/testthat')"

echo "[r] Build package"
R CMD build r

echo "[r] Completed"
