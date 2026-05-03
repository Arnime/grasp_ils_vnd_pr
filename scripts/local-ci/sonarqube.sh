#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

MISSING_VARS=()
[ -z "${SONAR_TOKEN:-}" ] && MISSING_VARS+=("SONAR_TOKEN")
[ -z "${SONAR_HOST_URL:-}" ] && MISSING_VARS+=("SONAR_HOST_URL")

if [ "${#MISSING_VARS[@]}" -gt 0 ]; then
  echo "Missing required environment variables for SonarQube scan."
  echo "Missing: ${MISSING_VARS[*]}"
  echo "Set these keys in .env (repo root) or export them in the shell."
  exit 1
fi

EXTRA_ARGS=()
if command -v cmake >/dev/null 2>&1; then
  echo "[sonarqube] Preparing C++ compile database"
  cmake -S cpp -B build-sonar \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGIVP_BUILD_TESTS=ON \
    -DGIVP_BUILD_BENCHMARKS=OFF
  if [ ! -f build-sonar/compile_commands.json ]; then
    echo "[sonarqube] compile_commands.json not generated; skipping C++ sources in this scan"
    EXTRA_ARGS+=("-Dsonar.sources=python/src,julia/src,rust/src,r/R")
    EXTRA_ARGS+=("-Dsonar.tests=python/tests,julia/test,rust/tests,r/tests/testthat")
  fi
elif [ -f build-sonar/compile_commands.json ]; then
  echo "[sonarqube] Using pre-generated C++ compile database at build-sonar/compile_commands.json"
else
  echo "[sonarqube] cmake not found in scanner image; skipping C++ compile database preparation"
  # Skip C++ in this environment because compile_commands.json cannot be generated.
  EXTRA_ARGS+=("-Dsonar.sources=python/src,julia/src,rust/src,r/R")
  EXTRA_ARGS+=("-Dsonar.tests=python/tests,julia/test,rust/tests,r/tests/testthat")
fi

if [ -f julia-lcov.info ]; then
  echo "[sonarqube] Julia coverage artifact found: julia-lcov.info"
else
  echo "[sonarqube] Julia coverage artifact not found; run julia target before sonarqube for coverage"
fi

if [ -d r/R ] && [ -d r/tests/testthat ]; then
  echo "[sonarqube] R sources/tests found and will be analyzed"
fi

echo "[sonarqube] Running scan"
sonar-scanner \
  -Dsonar.projectKey=Arnime_grasp_ils_vnd_pr \
  -Dsonar.projectName=givp \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout=300 \
  -Dsonar.working.directory=/tmp/sonarwork \
  "${EXTRA_ARGS[@]}"

echo "[sonarqube] Completed"
