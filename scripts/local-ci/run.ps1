# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$repoRootWsl = (wsl -e wslpath -a "$repoRoot").Trim()

if ([string]::IsNullOrWhiteSpace($repoRootWsl)) {
    Write-Error "Could not resolve repository path inside WSL."
    exit 1
}

$escapedArgs = @()
foreach ($arg in $Arguments) {
    $escapedArgs += "'" + $arg.Replace("'", "'\''") + "'"
}
$argString = ($escapedArgs -join " ")

$command = "cd '$repoRootWsl' && bash scripts/local-ci/run.sh $argString"
Write-Host "Executing in WSL: $command"
wsl -e bash -lc "$command"
exit $LASTEXITCODE
