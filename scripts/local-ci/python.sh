#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

echo "[python] Installing dependencies"
python -m pip install --upgrade pip
pip install --require-hashes -r python/requirements/ci.txt
pip install --require-hashes -r python/requirements/benchmarks.txt
bash .github/scripts/install-package.sh

echo "[python] Lint"
ruff check python/src python/tests

echo "[python] Type-check"
mypy

echo "[python] Full test suite"
pytest

echo "[python] Coverage gate (>=95%)"
pytest --cov=givp --cov-report=xml:coverage-python.xml --cov-fail-under=95

echo "[python] Property-based tests"
pytest python/tests/test_properties.py -v \
  --no-cov \
  --override-ini="addopts=" \
  -p no:randomly

echo "[python] Algorithm quality gate"
pytest -m quality_gate python/tests/test_algorithm_quality.py \
  -v \
  --no-cov \
  --override-ini="addopts=" \
  --tb=short

echo "[python] Benchmark smoke tests"
pytest -m slow python/tests/test_benchmark_scripts.py -v \
  --override-ini="addopts="

echo "[python] Benchmark regression"
mkdir -p python/benchmarks/.results
pytest python/benchmarks/test_benchmarks.py \
  --benchmark-only \
  --benchmark-autosave \
  --benchmark-storage=python/benchmarks/.results \
  --override-ini="addopts="

if [ "${RUN_MUTATION:-false}" = "true" ]; then
  echo "[python] Mutation testing"
  pip install mutmut==2.4.4
  BROKEN_LINKS=$(find . -xtype l | wc -l)
  if [ "$BROKEN_LINKS" -gt 0 ]; then
    find . -xtype l -print -delete
  fi

  MUTATE_PATHS="python/src/givp/api.py"
  TEST_RUNNER="python -m pytest python/tests/test_api.py -q --no-cov -x"
  mutmut run \
    --paths-to-mutate "$MUTATE_PATHS" \
    --runner "$TEST_RUNNER" \
    --max-children 8 \
    --no-progress

  KILLED=$(mutmut results 2>/dev/null | grep -cE "^[0-9]+ - [^-]+ - killed" || true)
  TOTAL=$(mutmut results 2>/dev/null | grep -cE "^[0-9]+ - [^-]+ - (killed|survived|timeout)" || true)
  echo "Killed: ${KILLED}  Total: ${TOTAL}"
  if [ "${TOTAL}" -eq 0 ]; then
    echo "No mutants generated."
    exit 1
  fi
  KILL_RATE=$(( KILLED * 100 / TOTAL ))
  echo "Kill rate: ${KILL_RATE}%"
  if [ "${KILL_RATE}" -lt 65 ]; then
    echo "ERROR: kill rate ${KILL_RATE}% is below 65%."
    exit 1
  fi
fi

echo "[python] Completed"
