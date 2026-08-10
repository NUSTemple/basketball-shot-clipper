@echo off
setlocal
rem The repo root, not this script's folder: every project path (models\,
rem data\configs\, data\jobs\) is resolved relative to cwd, so launching from
rem installer\ silently gave the app a second, empty data root - the trained
rem shot filter at models\shot_filter.joblib went missing (use_filter turns
rem itself off when the file isn't there), calibrations landed somewhere the
rem uninstaller would wipe, and ultralytics re-downloaded yolov8l.pt.
cd /d "%~dp0.."

rem Match setup-deps.ps1: poetry prefers an activated virtualenv over the
rem project's own, so launching from a shell that has one active would run the
rem app out of a venv the installer never provisioned. setlocal keeps this
rem scoped to this script.
set VIRTUAL_ENV=
set POETRY_ACTIVE=

if not exist "%USERPROFILE%\Videos\shot-clipper\clips" mkdir "%USERPROFILE%\Videos\shot-clipper\clips"

set SHOT_CLIPPER_CLIPS_DIR=%USERPROFILE%\Videos\shot-clipper\clips
set SHOT_CLIPPER_MEDIA_ROOT=%USERPROFILE%\Videos

where poetry >nul 2>nul
if errorlevel 1 (
    echo Poetry was not found on PATH. Re-run the shot-clipper installer, or open a
    echo new terminal if you just installed it ^(PATH changes need a fresh shell^).
    pause
    exit /b 1
)

start "" http://127.0.0.1:5050
poetry run shot-clipper-label-ui

pause
