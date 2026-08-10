<#
Runs once, at install time, from the Inno Setup [Run] step. Installs every
prerequisite the current manual README / docs/GPU_SETUP.md workflow needs,
then leaves the app ready to detect/clip immediately - see the "install ->
immediately usable" bar in the installer plan.
#>

$ErrorActionPreference = 'Stop'
# This script lives at <repo root>\installer\setup-deps.ps1, but poetry
# install / model downloads need to run from the repo root itself (all
# project paths - pyproject.toml, models/, data/ - are resolved relative to
# cwd), so go up one level from the script's own folder.
$installerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = Split-Path -Parent $installerDir
Set-Location $appDir

function Fail($message) {
    Write-Host ""
    Write-Host "SETUP FAILED: $message" -ForegroundColor Red
    Write-Host "You can re-run this step later from: $appDir\installer\setup-deps.ps1"
    exit 1
}

function Refresh-Path {
    # winget/Poetry installs update the registry PATH, not this process's copy -
    # re-read both scopes so newly installed tools are usable without restarting
    # the terminal (docs/GPU_SETUP.md's workaround for this is "restart your
    # terminal"; do that automatically instead).
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

Write-Host "=== shot-clipper setup ===" -ForegroundColor Cyan

# --- winget -----------------------------------------------------------
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Fail "winget was not found. Install 'App Installer' from the Microsoft Store, then re-run this installer."
}

# --- Python -------------------------------------------------------------
$havePython = $false
foreach ($cmd in @('py', 'python')) {
    $exe = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($exe) {
        try {
            $verOut = & $cmd --version 2>&1
            if ($verOut -match '(\d+)\.(\d+)') {
                $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) { $havePython = $true }
            }
        } catch {}
    }
}
if (-not $havePython) {
    Write-Host "Installing Python 3.12 via winget..."
    winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Fail "Python install via winget failed (exit $LASTEXITCODE)." }
    Refresh-Path
    if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
        Fail "Python installed but isn't on PATH yet. Close this window, open a new terminal, and re-run installer\setup-deps.ps1."
    }
}

# --- Poetry ---------------------------------------------------------------
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Poetry..."
    # Poetry's own recommended installer, not winget - more reliable across
    # whichever Python install ends up on PATH.
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    if ($LASTEXITCODE -ne 0) { Fail "Poetry install failed (exit $LASTEXITCODE)." }
    Refresh-Path
    # Poetry's installer adds its own bin dir to the user PATH but only takes
    # effect on the next login for some Windows configs - add it explicitly too.
    $poetryBin = Join-Path $env:APPDATA 'Python\Scripts'
    if (Test-Path $poetryBin) { $env:Path = "$env:Path;$poetryBin" }
}
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Fail "Poetry installed but isn't on PATH yet. Close this window, open a new terminal, and re-run installer\setup-deps.ps1."
}

# --- ffmpeg -----------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing ffmpeg via winget..."
    winget install --id Gyan.FFmpeg --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { Fail "ffmpeg install via winget failed (exit $LASTEXITCODE)." }
    Refresh-Path
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Fail "ffmpeg installed but isn't on PATH yet. Close this window, open a new terminal, and re-run installer\setup-deps.ps1."
}

# --- Python dependencies (always the ml group - Detect must work out of the box) ---
Write-Host "Installing Python dependencies (poetry install --with ml)..."
poetry install --with ml
if ($LASTEXITCODE -ne 0) { Fail "poetry install --with ml failed (exit $LASTEXITCODE)." }

# --- Optional NVIDIA CUDA torch swap --------------------------------------
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host ""
    Write-Host "NVIDIA GPU detected:"
    & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    $answer = Read-Host "Install CUDA-accelerated torch for faster detection? [Y/n]"
    if ($answer -notmatch '^[Nn]') {
        # plain `poetry install` always lands the CPU wheel (no CUDA index pinned
        # in pyproject.toml on purpose - see docs/GPU_SETUP.md), so the CUDA
        # build has to be installed straight into Poetry's own venv afterward.
        $venvPath = (poetry env info -p).Trim()
        $pipExe = Join-Path $venvPath 'Scripts\pip.exe'
        Write-Host "Reinstalling torch/torchvision with CUDA support (this can take a few minutes)..."
        & $pipExe install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CUDA torch install failed - continuing with CPU torch. See docs\GPU_SETUP.md to retry manually." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "No NVIDIA GPU detected (nvidia-smi not found) - using CPU/MPS torch."
}

# --- Model weights ------------------------------------------------------
Write-Host ""
Write-Host "Downloading YOLO model weights (models/yolov8m.pt, models/yolov8l.pt)..."
New-Item -ItemType Directory -Force -Path (Join-Path $appDir 'models') | Out-Null
poetry run python -c "from ultralytics import YOLO; YOLO('models/yolov8m.pt'); YOLO('models/yolov8l.pt')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model weight download failed - Detect won't work until this succeeds." -ForegroundColor Yellow
    Write-Host "Retry later with: poetry run python -c \"from ultralytics import YOLO; YOLO('models/yolov8m.pt'); YOLO('models/yolov8l.pt')\"" -ForegroundColor Yellow
}

# --- Summary ------------------------------------------------------------
Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
poetry run python -c "from shot_clipper.device_config import get_device; print('inference device:', get_device())" 2>$null
Write-Host "Clips folder: $env:USERPROFILE\Videos\shot-clipper\clips"
Write-Host "Launch shot-clipper from the Start Menu or Desktop shortcut whenever you're ready."
