#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

echo "[r] Install dependencies"
Rscript -e "install.packages(c('R6','rlang','testthat'), repos='https://cloud.r-project.org')"

echo "[r] Install package"
R CMD INSTALL r

echo "[r] Tests"
Rscript -e "testthat::test_dir('r/tests/testthat')"

echo "[r] Build package"
R CMD build r

echo "[r] Completed"
