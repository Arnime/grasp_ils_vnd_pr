#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

BUILD_DIR=build_local_ci_cpp
COV_DIR=build_local_ci_cpp_cov
BENCH_DIR=build_local_ci_cpp_bench

echo "[cpp] Configure + build + test"
cmake -S cpp -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_TESTS=ON -DGIVP_BUILD_BENCHMARKS=OFF
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "[cpp] Coverage"
cmake -S cpp -B "$COV_DIR" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage" -DCMAKE_EXE_LINKER_FLAGS="--coverage" -DGIVP_BUILD_TESTS=ON -DGIVP_BUILD_BENCHMARKS=OFF
cmake --build "$COV_DIR" --parallel
ctest --test-dir "$COV_DIR" --output-on-failure

lcov --capture \
  --directory "$COV_DIR" \
  --output-file coverage_raw.info \
  --gcov-tool gcov \
  --ignore-errors source,gcov,negative \
  --rc geninfo_unexecuted_blocks=1
lcov --remove coverage_raw.info \
  "*/${COV_DIR}/_deps/*" \
  '/usr/*' \
  --output-file coverage.info \
  --ignore-errors source,negative
lcov --list coverage.info

hit=$(grep '^LH:' coverage.info | awk -F: '{s+=$2} END {print s+0}')
total=$(grep '^LF:' coverage.info | awk -F: '{s+=$2} END {print s+0}')
if [ "$total" -eq 0 ]; then
  echo "No coverage data found in coverage.info"
  exit 1
fi
pct=$(awk "BEGIN {printf \"%.1f\", $hit / $total * 100}")
echo "C++ coverage: ${pct}% (${hit} / ${total} lines)"
awk "BEGIN { if ($hit / $total * 100 < 80.0) { print \"Coverage below 80% threshold (got ${pct}%)\"; exit 1 } }"

echo "[cpp] Benchmark smoke"
cmake -S cpp -B "$BENCH_DIR" -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_TESTS=OFF -DGIVP_BUILD_BENCHMARKS=ON
cmake --build "$BENCH_DIR" --parallel
./"$BENCH_DIR"/benchmarks/givp_benchmarks

echo "[cpp] Completed"
