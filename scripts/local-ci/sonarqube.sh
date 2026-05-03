#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

if [ -z "${SONAR_TOKEN:-}" ] || [ -z "${SONAR_HOST_URL:-}" ]; then
  echo "Missing required environment variables for SonarQube scan."
  echo "Set SONAR_TOKEN and SONAR_HOST_URL in .env (repo root) or shell env."
  exit 1
fi

if command -v cmake >/dev/null 2>&1; then
  echo "[sonarqube] Preparing C++ compile database"
  cmake -S cpp -B build-sonar \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGIVP_BUILD_TESTS=OFF \
    -DGIVP_BUILD_BENCHMARKS=OFF
else
  echo "[sonarqube] cmake not found in scanner image; skipping C++ compile database preparation"
fi

echo "[sonarqube] Running scan"
sonar-scanner \
  -Dsonar.projectKey=Arnime_grasp_ils_vnd_pr \
  -Dsonar.projectName=givp \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout=300

echo "[sonarqube] Completed"
