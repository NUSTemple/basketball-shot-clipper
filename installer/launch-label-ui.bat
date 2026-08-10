@echo off
setlocal
cd /d "%~dp0"

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
