# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
#
# Julia environment activation script — mirrors .venv\Scripts\Activate.ps1
# Usage (from repo root):
#   . .\.julia-env\Activate.ps1
#
# After activation:
#   julia                  → opens REPL with the GIVPOptimizer project active
#   julia script.jl        → runs script within the project environment
#   deactivate-julia       → removes Julia from PATH and clears JULIA_PROJECT

$script:REPO_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$script:JULIA_BIN = $null

# ── Locate julia.exe ─────────────────────────────────────────────────────────
$script:candidates = @(
    "C:\Julia-1.12.6\bin\julia.exe",
    "$env:LOCALAPPDATA\Programs\Julia-1.12.6\bin\julia.exe",
    "$env:LOCALAPPDATA\Programs\Julia\bin\julia.exe",
    "C:\Program Files\Julia\bin\julia.exe"
)
foreach ($c in $script:candidates) {
    if (Test-Path $c) { $script:JULIA_BIN = Split-Path $c -Parent; break }
}
if (-not $script:JULIA_BIN) {
    # Fall back to whatever is in PATH (may already be set)
    $found = Get-Command julia -ErrorAction SilentlyContinue
    if ($found) { $script:JULIA_BIN = Split-Path $found.Path -Parent }
}
if (-not $script:JULIA_BIN) {
    Write-Error "julia.exe not found. Install Julia from https://julialang.org/downloads/ or set the path in .julia-env\Activate.ps1"
    return
}

# ── Deactivate function ───────────────────────────────────────────────────────
function global:Disable-JuliaEnvironment {
    if (Test-Path variable:_OLD_JULIA_PATH) {
        $env:PATH = $global:_OLD_JULIA_PATH
        Remove-Variable "_OLD_JULIA_PATH" -Scope global -ErrorAction SilentlyContinue
    }
    if (Test-Path env:JULIA_PROJECT) {
        Remove-Item env:JULIA_PROJECT -ErrorAction SilentlyContinue
    }
    if (Test-Path function:_old_julia_prompt) {
        $function:prompt = $function:_old_julia_prompt
        Remove-Item function:\_old_julia_prompt -ErrorAction SilentlyContinue
    }
    Remove-Item env:JULIA_ENV -ErrorAction SilentlyContinue
    Remove-Item function:Disable-JuliaEnvironment -ErrorAction SilentlyContinue
    Remove-Item alias:deactivate-julia -ErrorAction SilentlyContinue
    Write-Host "Julia environment deactivated." -ForegroundColor Yellow
}
# Friendly alias — mirrors Python venv's 'deactivate' convention
Set-Alias -Name deactivate-julia -Value Disable-JuliaEnvironment -Scope global

# ── Save old state and activate ──────────────────────────────────────────────
$global:_OLD_JULIA_PATH = $env:PATH

# Add Julia bin to PATH if not already present
if ($env:PATH -notlike "*$script:JULIA_BIN*") {
    $env:PATH = "$script:JULIA_BIN;$env:PATH"
}

# Point JULIA_PROJECT at the repo's julia/ package
$env:JULIA_PROJECT = Join-Path $script:REPO_ROOT "julia"
$env:JULIA_ENV = "givp-julia-0.8.0"

# ── Update prompt ─────────────────────────────────────────────────────────────
if (Test-Path function:prompt) {
    $function:_old_julia_prompt = $function:prompt
}
function global:prompt {
    "(givp-julia) " + (& $function:_old_julia_prompt)
}

Write-Host ""
Write-Host "  Julia environment activated" -ForegroundColor Cyan
Write-Host "  Julia   : $(& julia --version 2>&1)" -ForegroundColor Green
Write-Host "  Project : $env:JULIA_PROJECT" -ForegroundColor Green
Write-Host "  Run 'deactivate-julia' to exit." -ForegroundColor DarkGray
Write-Host ""
