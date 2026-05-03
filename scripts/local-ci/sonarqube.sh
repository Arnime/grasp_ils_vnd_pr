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
    -DGIVP_BUILD_TESTS=OFF \
    -DGIVP_BUILD_BENCHMARKS=OFF
else
  echo "[sonarqube] cmake not found in scanner image; skipping C++ compile database preparation"
  # Skip C++ in this environment because compile_commands.json cannot be generated.
  EXTRA_ARGS+=("-Dsonar.sources=python/src,julia/src,rust/src,r/R")
  EXTRA_ARGS+=("-Dsonar.tests=python/tests,julia/test,rust/tests,r/tests/testthat")
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
