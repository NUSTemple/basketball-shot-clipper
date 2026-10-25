<#
Sets up a Windows PC to run the WHOLE pipeline locally (/make-highlights:
videos -> clips, AI make scores, AI stars, goals by player, highlight reels).

Run from the repo root (the setup-local skill does this for you):

    powershell -NoProfile -ExecutionPolicy Bypass -File installer\setup-local-pipeline.ps1

Steps: setup-deps.ps1 (Python, Poetry, ffmpeg, Python packages, CUDA torch
- the cu132 build, which RTX 50-series cards need - and the detector
weights), then Node.js + the Remotion packages for rendering reels, the
person re-ID weights, and Ollama + the qwen3-vl vision model for AI stars.
Ends with shot-clipper-doctor. Safe to re-run: every step skips what's
already there.

  -SkipOllama        no vision model (AI stars fall back to the make score)
  -OllamaModel NAME  default qwen3-vl:30b (~20GB, 24GB+ GPU); qwen3-vl:8b for smaller GPUs
#>
param(
    [switch]$SkipOllama,
    [string]$OllamaModel = 'qwen3-vl:30b'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

function Step($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host ""; Write-Host "SETUP FAILED: $text" -ForegroundColor Red; exit 1 }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
    $poetryBin = Join-Path $env:APPDATA 'Python\Scripts'
    if (Test-Path $poetryBin) { $env:Path = "$env:Path;$poetryBin" }
    $ollamaBin = Join-Path $env:LOCALAPPDATA 'Programs\Ollama'
    if (Test-Path $ollamaBin) { $env:Path = "$env:Path;$ollamaBin" }
}
function Winget-Install($id, $what) {
    Write-Host "Installing $what via winget..."
    winget install --id $id --source winget --accept-package-agreements --accept-source-agreements --silent
    # -1978335189 = "already installed"
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) { Fail "$what install via winget failed (exit $LASTEXITCODE)." }
    Refresh-Path
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Fail "winget not found. Install 'App Installer' from the Microsoft Store, then re-run."
}

Step "1/5 Python, Poetry, ffmpeg, Python packages, CUDA torch, detector weights"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'installer\setup-deps.ps1')
if ($LASTEXITCODE -ne 0) { Fail "installer\setup-deps.ps1 failed - see setup-log.txt" }
Refresh-Path

Step "2/5 Node.js + Remotion (renders the reels)"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Winget-Install 'OpenJS.NodeJS.LTS' 'Node.js LTS' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Fail "Node.js installed but npm isn't on PATH yet - open a new terminal and re-run." }
Push-Location (Join-Path $repo 'remotion-reel')
npm ci
$npmExit = $LASTEXITCODE
Pop-Location
if ($npmExit -ne 0) { Fail "npm ci in remotion-reel failed (exit $npmExit)." }

Step "3/5 Person re-ID weights (who scored)"
poetry run python -c "from shot_clipper.person_id.reid import ensure_weights; print(ensure_weights())"
if ($LASTEXITCODE -ne 0) { Write-Host "Re-ID weight download failed - it retries on first use." -ForegroundColor Yellow }

Step "4/5 Ollama + $OllamaModel (AI stars)"
if ($SkipOllama) {
    Write-Host "-SkipOllama: no vision model; AI stars will use the make score only."
} else {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Winget-Install 'Ollama.Ollama' 'Ollama' }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Fail "Ollama installed but isn't on PATH yet - open a new terminal and re-run." }
    # the Ollama app normally starts its server at login; make sure one is up now
    try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3 | Out-Null }
    catch {
        Start-Process -WindowStyle Hidden ollama 'serve'
        Start-Sleep -Seconds 5
    }
    Write-Host "Downloading $OllamaModel (~20GB for the 30b model - this takes a while)..."
    ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $OllamaModel failed." }
    if ($OllamaModel -ne 'qwen3-vl:30b') {
        [Environment]::SetEnvironmentVariable('SHOT_CLIPPER_OLLAMA_MODEL', $OllamaModel, 'User')
        $env:SHOT_CLIPPER_OLLAMA_MODEL = $OllamaModel
        Write-Host "Saved SHOT_CLIPPER_OLLAMA_MODEL=$OllamaModel for your user."
    }
}

Step "5/5 Check"
poetry run shot-clipper-doctor
$doctorExit = $LASTEXITCODE
Write-Host ""
if ($doctorExit -eq 0) {
    Write-Host "Ready. Close this terminal (and restart Claude Code) so new tools are on PATH, then:" -ForegroundColor Green
    Write-Host "  /make-highlights <your video folder>      (in Claude Code)"
    Write-Host "  poetry run shot-clipper-highlights `"<your video folder>`"   (by hand)"
} else {
    Write-Host "Some checks failed - see the 'fix:' lines above." -ForegroundColor Yellow
    exit 1
}
