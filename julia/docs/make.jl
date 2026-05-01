# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Documenter.jl build script for GIVPOptimizer.jl
# Build locally:  julia --project=julia/docs julia/docs/make.jl
# Preview:        julia --project=julia/docs julia/docs/make.jl && open julia/docs/build/index.html

using Pkg

# Ensure the local GIVPOptimizer package is available
Pkg.develop(; path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using GIVPOptimizer

makedocs(;
    sitename = "GIVPOptimizer.jl",
    authors = "Arnaldo Mendes Pires Junior",
    modules = [GIVPOptimizer],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://arnime.github.io/grasp_ils_vnd_pr",
        assets = String[],
        repolink = "https://github.com/Arnime/grasp_ils_vnd_pr",
    ),
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Algorithm" => "algorithm.md",
        "API Reference" => "api.md",
        "Julia Guide" => "julia.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/Arnime/grasp_ils_vnd_pr.git",
    devbranch = "main",
    push_preview = true,
)
