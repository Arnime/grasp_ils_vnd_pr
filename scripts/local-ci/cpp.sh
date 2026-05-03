#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

echo "[cpp] Configure + build + test"
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_TESTS=ON -DGIVP_BUILD_BENCHMARKS=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure

echo "[cpp] Coverage"
cmake -S cpp -B build_cov -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage" -DCMAKE_EXE_LINKER_FLAGS="--coverage" -DGIVP_BUILD_TESTS=ON -DGIVP_BUILD_BENCHMARKS=OFF
cmake --build build_cov --parallel
ctest --test-dir build_cov --output-on-failure

lcov --capture \
  --directory build_cov \
  --output-file coverage_raw.info \
  --gcov-tool gcov \
  --ignore-errors source,gcov,negative \
  --rc geninfo_unexecuted_blocks=1
lcov --remove coverage_raw.info \
  '*/build_cov/_deps/*' \
  '/usr/*' \
  --output-file coverage.info \
  --ignore-errors source
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
cmake -S cpp -B build_bench -DCMAKE_BUILD_TYPE=Release -DGIVP_BUILD_TESTS=OFF -DGIVP_BUILD_BENCHMARKS=ON
cmake --build build_bench --parallel
./build_bench/benchmarks/givp_benchmarks

echo "[cpp] Completed"
