# Changelog

<!-- markdownlint-disable MD024 -->

## [1.0.0](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.5.4...v1.0.0) (2026-04-29)

### Features

* promote Python implementation to stable v1.0.0 — full production release
  (95% test coverage, mypy strict, CLI, PyPI, fuzzing, property-based tests)
* Julia implementation reaches v0.8.0 (80% production-ready): full feature
  parity, 1100+ line test suite, coverage measured, multi-version CI, JuliaHub
* Rust implementation reaches v0.7.0 (70% production-ready): all modules ported,
  multi-toolchain CI, published on crates.io
* C++ header-only library updated to v1.0.0 in sync with Python release

## [0.5.2](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.5.1...v0.5.2) (2026-04-27)

### Features

* add Julia port with full feature parity (GRASP, ILS, VND, Path Relinking,
  elite pool, convergence monitor, LRU cache, adaptive α, time budget)
* add Julia CI workflow (`ci-julia.yml`) with multi-OS/version matrix
* add Julia benchmarks (`BenchmarkTools.jl`) with regression tracking
* update documentation, badges, and metadata to cover both Python and Julia

## [0.5.1](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.5.0...v0.5.1) (2026-04-27)

### Bug Fixes

* fix badge and repository links in PyPI README — replace stale `Arnime/givp`
  references with `Arnime/grasp_ils_vnd_pr` for CI, Codecov and OpenSSF Scorecard

## [0.5.0](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.3.0...v0.5.0) (2026-04-26)

### Features

* rename `construct_solution_numpy` to `construct_grasp` for a cleaner public API
* add individual component documentation (GRASP, ILS, VND, Path Relinking)
  to examples page

### Bug Fixes

* fix shell injection in `backfill-provenance.yml` — move
  `${{ inputs.tag_name }}` to `env:` block (semgrep `run-shell-injection`)
* fix mypy `no-any-return` error in `get_current_alpha` (`grasp.py` line 413)
* fix broken `api/givp.md` nav/link references — corrected to
  `api/grasp_ils_vnd_pr.md`
* fix all stale `Arnime/givp` repository URLs replaced with
  `Arnime/grasp_ils_vnd_pr`

### Documentation

* fix MD046 code-block style in `docs/examples/benchmarks.md`
  (indented → fenced)

### Tests

* add branch coverage for `bidirectional_path_relinking` —
  `cost1 <= cost2` and `cost2 < cost1` paths

## [0.3.0](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.2.2...v0.3.0) (2026-04-23)

### Features

* add Python 3.14 support; extend `requires-python` to `<3.15`

## [0.2.0](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.1.2...v0.2.0) (2026-04-22)

### Features

* add hypothesis as cross-platform fuzz backend (Windows/macOS/Linux) ([e58e2a4](https://github.com/Arnime/grasp_ils_vnd_pr/commit/e58e2a4c8a79ed281f892b577f004f25230d9318))

## [0.1.2](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.1.1...v0.1.2) (2026-04-22)

### Documentation

* remove duplicate README section, disable MD013 tables and MD060 lint rules ([3b88654](https://github.com/Arnime/grasp_ils_vnd_pr/commit/3b88654ea18be9ea16dab4e2a50f52125527a1f7))
* replace external-domain badges with static shields.io badges ([7cb6a22](https://github.com/Arnime/grasp_ils_vnd_pr/commit/7cb6a22ce15b8813b5e18ac744b5157f02b7cf07))

## [0.1.1](https://github.com/Arnime/grasp_ils_vnd_pr/compare/v0.1.0...v0.1.1) (2026-04-22)

### Bug Fixes

* correct indentation in testpypi workflow and add markdownlint configuration ([f83e771](https://github.com/Arnime/grasp_ils_vnd_pr/commit/f83e7719b37b9fa0e09aed67806260d81bc69f83))
