#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace/rust

echo "[rust] Toolchain components"
rustup component add clippy rustfmt llvm-tools-preview
cargo install cargo-llvm-cov --locked

echo "[rust] Build"
cargo build --verbose

echo "[rust] Test"
cargo test --verbose

echo "[rust] Clippy"
cargo clippy -- -D warnings

echo "[rust] fmt check"
cargo fmt --all -- --check

echo "[rust] Coverage gate (>=90%)"
cargo llvm-cov --all-features --lcov --output-path lcov.info
hit=$(grep '^LH:' lcov.info | awk -F: '{s+=$2} END {print s+0}')
total=$(grep '^LF:' lcov.info | awk -F: '{s+=$2} END {print s+0}')
if [ "$total" -eq 0 ]; then
  echo "No coverage data found in lcov.info"
  exit 1
fi
pct=$(awk "BEGIN {printf \"%.1f\", $hit / $total * 100}")
echo "Rust coverage: ${pct}% (${hit} / ${total} lines)"
awk "BEGIN { if ($hit / $total * 100 < 90.0) { print \"Coverage below 90% threshold (got ${pct}%)\"; exit 1 } }"

echo "[rust] Benchmarks (smoke)"
cargo bench -- --test

echo "[rust] Completed"
