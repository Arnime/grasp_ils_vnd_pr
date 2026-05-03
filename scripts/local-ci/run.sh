#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

COMPOSE_FILE="docker/local-ci/docker-compose.yml"
DEFAULT_TARGETS=(workflow-lint python rust julia cpp r sonarqube)
CONTINUE_ON_ERROR=false
ENV_FILE=".env"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/local-ci/run.sh [target ...] [--continue-on-error]

Targets:
  workflow-lint  Validate GitHub Actions workflows with actionlint
  python         Run CI-equivalent Python jobs
  rust           Run CI-equivalent Rust jobs
  julia          Run CI-equivalent Julia jobs
  cpp            Run CI-equivalent C++ jobs
  r              Run CI-equivalent R jobs
  sonarqube      Run local SonarQube scan (requires SONAR_TOKEN and SONAR_HOST_URL)
  all            Run all targets above (default)

Examples:
  bash scripts/local-ci/run.sh
  bash scripts/local-ci/run.sh workflow-lint python
  RUN_MUTATION=true bash scripts/local-ci/run.sh python
EOF
}

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Missing compose file: $COMPOSE_FILE"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is required."
  exit 1
fi

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      ;;
    all)
      TARGETS=("${DEFAULT_TARGETS[@]}")
      ;;
    workflow-lint|python|rust|julia|cpp|r|sonarqube)
      TARGETS+=("$arg")
      ;;
    *)
      echo "Unknown target: $arg"
      usage
      exit 2
      ;;
  esac
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

FAILED=()
for target in "${TARGETS[@]}"; do
  echo "==== Running target: ${target} ===="
  compose_cmd=(docker compose -f "$COMPOSE_FILE")
  if [ -f "$ENV_FILE" ]; then
    compose_cmd+=(--env-file "$ENV_FILE")
  fi
  if ! "${compose_cmd[@]}" run --rm "$target"; then
    FAILED+=("$target")
    if [ "$CONTINUE_ON_ERROR" = false ]; then
      echo "Target failed: $target"
      exit 1
    fi
  fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed targets: ${FAILED[*]}"
  exit 1
fi

echo "All selected local CI targets passed."
