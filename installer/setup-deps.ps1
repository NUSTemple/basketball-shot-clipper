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

# Every poetry command below - `install`, `run`, `env info` alike - resolves to
# an already-activated virtualenv in preference to the project's own. If this
# script runs from a shell where some other checkout's venv is active, the whole
# install (deps, CUDA torch, model download) silently lands in THAT venv, while
# the Start Menu shortcut later launches from a clean shell and gets the
# project's venv instead - one with plain CPU torch. That is exactly how an
# RTX 4070 machine ended up detecting at 0.13x realtime. Clearing these makes
# install time and launch time agree on which venv is "the" venv.
$env:VIRTUAL_ENV = $null
$env:POETRY_ACTIVE = $null
Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
Remove-Item Env:\POETRY_ACTIVE -ErrorAction SilentlyContinue

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

# Inno's [Run] window closes with the wizard, so anything printed here is gone
# the moment setup finishes - which left no way to tell whether the CUDA step
# ran, was skipped, or failed. Transcript it. Best-effort: a setup that cannot
# open a log is not a setup that should abort.
$logPath = Join-Path $appDir 'setup-log.txt'
try { Start-Transcript -Path $logPath -Force | Out-Null } catch { $logPath = '(transcript unavailable)' }

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
    # Deliberately not Read-Host: this script runs from Inno's [Run] step, where
    # stdin is not a real console, so the prompt that used to be here could be
    # skipped without anyone seeing it - which is how a machine with an RTX 4070
    # ended up running detection on CPU at 0.13x realtime. Detection is roughly
    # an order of magnitude slower on CPU, so install CUDA torch by default and
    # let SHOT_CLIPPER_SKIP_CUDA=1 opt out.
    if ($env:SHOT_CLIPPER_SKIP_CUDA -eq '1') {
        Write-Host "SHOT_CLIPPER_SKIP_CUDA=1 - keeping the CPU torch build."
    } else {
        # plain `poetry install` always lands the CPU wheel (no CUDA index pinned
        # in pyproject.toml on purpose - see docs/GPU_SETUP.md), so the CUDA
        # build has to be installed into Poetry's own venv afterward.
        #
        # `poetry run python -m pip`, not `(poetry env info -p)\Scripts\pip.exe`:
        # this is the same command form the launcher uses, so the wheels land in
        # whichever venv `poetry run shot-clipper-label-ui` will pick. (That is
        # only true because VIRTUAL_ENV was cleared at the top of this script -
        # `poetry run` honours an active virtualenv just as much as `env info`
        # does, so the command form alone is not what makes this correct.)
        # The +cu132 local version is load-bearing, not decoration. `poetry
        # install --with ml` above has already put torch 2.13.0+cpu in place,
        # and pip treats that as satisfying a plain `torch==2.13.0` - it prints
        # "Requirement already satisfied", changes nothing, and exits 0. That
        # no-op, not the venv, is why this machine kept running on CPU through
        # three releases. Naming the local version makes 2.13.0+cpu a mismatch,
        # so pip actually replaces it.
        Write-Host "Reinstalling torch/torchvision with CUDA support (this can take a few minutes)..."
        poetry run python -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CUDA torch install failed - continuing with CPU torch. See docs\GPU_SETUP.md to retry manually." -ForegroundColor Yellow
        }
    }
    # Verify rather than assume. A silent fall back to CPU torch on a GPU box is
    # invisible in the UI (it just runs ~10x slower), so say so at install time.
    $cudaOk = (poetry run python -c "import torch; print(torch.cuda.is_available())" 2>$null | Out-String).Trim()
    if ($cudaOk -ne 'True') {
        Write-Host ""
        Write-Host "WARNING: an NVIDIA GPU is present but torch cannot use it - detection will" -ForegroundColor Yellow
        Write-Host "run on CPU and be roughly 10x slower. Retry with:" -ForegroundColor Yellow
        Write-Host "  cd `"$appDir`"; poetry run python -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132" -ForegroundColor Yellow
        Write-Host "Full setup output was saved to: $logPath" -ForegroundColor Yellow
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
Write-Host "Setup log:    $logPath"
Write-Host "Launch shot-clipper from the Start Menu or Desktop shortcut whenever you're ready."
