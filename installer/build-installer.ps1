<#
Maintainer-run, on Windows: stages a clean snapshot of the current commit
(only git-tracked files, so .venv/, clips/, models/*.pt, etc. never leak in)
and compiles it into shot-clipper-setup.exe with Inno Setup.

Requires Inno Setup 6 (ISCC.exe) installed - https://jrsoftware.org/isinfo.php

-Version stamps the build (installer.iss falls back to its own default when
omitted); the release workflow passes the pushed tag so the two never drift.
#>

param(
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$installerDir = $PSScriptRoot
$stageDir = Join-Path $installerDir 'stage'

Set-Location $repoRoot

Write-Host "Staging a clean git snapshot into $stageDir ..."
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

git archive HEAD | tar -x -C $stageDir
if ($LASTEXITCODE -ne 0) { throw "git archive/tar staging failed (exit $LASTEXITCODE)." }

$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $default = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $default) { $iscc = Get-Item $default } else {
        throw "ISCC.exe not found on PATH. Install Inno Setup 6: https://jrsoftware.org/isinfo.php"
    }
}

$isccArgs = @()
if ($Version) {
    Write-Host "Stamping version $Version"
    $isccArgs += "/DAppVersion=$Version"
}

Write-Host "Compiling installer..."
& $iscc.Path @isccArgs "$installerDir\installer.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC compile failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "Built: $installerDir\dist\shot-clipper-setup.exe" -ForegroundColor Green
