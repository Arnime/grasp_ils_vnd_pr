#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

set -euo pipefail

cd /workspace

echo "[julia] Install dependencies"
julia --project=julia -e '
  using Pkg
  Pkg.Registry.update()
  rm(joinpath("julia", "Manifest.toml"); force=true)
  Pkg.instantiate()
'

echo "[julia] Run tests"
JULIA_NUM_THREADS=auto julia --project=julia -e 'using Pkg; Pkg.test()'

echo "[julia] Coverage"
JULIA_NUM_THREADS=2 julia --project=julia -e 'using Pkg; Pkg.test(; coverage=true)'

julia -e '
  using Pkg
  Pkg.add("CoverageTools")
  using CoverageTools
  coverage = process_folder("julia/src")
  LCOV.writefile("julia-lcov.info", coverage)
'

julia -e '
  using Pkg
  Pkg.add("CoverageTools")
  using CoverageTools
  cov = process_folder("julia/src")
  let
    hit = 0
    total = 0
    for s in cov
      for c in s.coverage
        c === nothing && continue
        total += 1
        c > 0 && (hit += 1)
      end
    end
    pct = total > 0 ? hit / total * 100.0 : 0.0
    println("Julia coverage: $(round(pct; digits=1))%  ($hit / $total lines)")
    pct >= 95.0 || (println("Coverage below 95% threshold (got $(round(pct; digits=1))%)"); exit(1))
  end
'

echo "[julia] Formatting check"
julia -e '
  using Pkg; Pkg.add("JuliaFormatter")
  using JuliaFormatter
  formatted = format("julia/"; overwrite=false)
  if !formatted
    println("Julia code is not formatted.")
    exit(1)
  end
'

echo "[julia] Fuzz trials"
julia --project=julia julia/fuzz/fuzz_givp.jl \
  --n-trials 100 \
  --seed 1337 \
  --timeout 90

echo "[julia] Aqua lint"
julia --project=julia -e '
  using Pkg
  Pkg.add(["Aqua"])
  using GIVPOptimizer, Aqua
  Aqua.test_all(
    GIVPOptimizer;
    ambiguities = (broken = false,),
    stale_deps = (ignore = [:JSON, :Aqua],),
  )
'

echo "[julia] JET"
julia --project=julia -e '
  using Pkg
  Pkg.add(["JET"])
  using GIVPOptimizer, JET
  result = JET.report_package("GIVPOptimizer"; ignored_modules=(Base, Core))
  reports = JET.get_reports(result)
  if !isempty(reports)
    for r in reports
      println("JET warning: ", r)
    end
  end
'

echo "[julia] Audit checks"
julia -e '
  using Pkg
  Pkg.activate("julia")
  Pkg.Registry.update()
  Pkg.instantiate()
  Pkg.status()
'

if grep -rn --include="*.jl" -E '\\bccall\\b|\\b@ccall\\b|\\bunsafe_' julia/src/; then
  echo "Unsafe FFI call (ccall/@ccall/unsafe_*) found in julia/src/."
  exit 1
fi

echo "[julia] Completed"
