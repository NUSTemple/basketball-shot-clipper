@echo off
setlocal
rem The repo root, not this script's folder: this is an unfrozen install, so
rem paths.app_root() is the working directory and every project path
rem (models\, data\configs\, data\jobs\) hangs off it. Launching from
rem installer\ silently gave the app a second, empty data root - the trained
rem shot filter at models\shot_filter.joblib went missing (use_filter turns
rem itself off when the file isn't there), calibrations landed somewhere the
rem uninstaller would wipe, and ultralytics re-downloaded yolov8l.pt.
rem A frozen build won't need this: it resolves its data root per-user
rem instead (see src\shot_clipper\paths.py, docs\PACKAGING.md).
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

rem ---------------------------------------------------------------------
rem Find Poetry WITHOUT trusting this process's PATH.
rem
rem This used to be a bare `where poetry`, and it failed for people whose
rem install had just succeeded. Poetry's installer writes its bin directory
rem into the user PATH in the registry, but Explorer caches the environment
rem at login and hands that stale copy to every shortcut it launches - so
rem the Start Menu entry sees no Poetry until the user logs out and back in.
rem setup-deps.ps1 hid the problem from itself by patching its own process
rem PATH (see Refresh-Path), which is the worst possible split: install
rem reports success, then the shortcut fails with "Poetry was not found".
rem
rem Three sources, cheapest first. The recorded path is written by
rem setup-deps.ps1 at install time and is the only one that can't go stale.
rem ---------------------------------------------------------------------
set "POETRY="

if exist "%~dp0poetry-path.txt" set /p POETRY=<"%~dp0poetry-path.txt"
if defined POETRY if not exist "%POETRY%" set "POETRY="

if not defined POETRY (
    for /f "delims=" %%p in ('where poetry 2^>nul') do if not defined POETRY set "POETRY=%%p"
)

if not defined POETRY (
    for %%p in (
        "%APPDATA%\Python\Scripts\poetry.exe"
        "%APPDATA%\pypoetry\venv\Scripts\poetry.exe"
        "%LOCALAPPDATA%\pypoetry\venv\Scripts\poetry.exe"
        "%LOCALAPPDATA%\Programs\Python\Scripts\poetry.exe"
    ) do if not defined POETRY if exist %%p set "POETRY=%%~p"
)

if not defined POETRY (
    echo Could not find Poetry. Looked in:
    echo   %~dp0poetry-path.txt ^(recorded at install time^)
    echo   your PATH
    echo   %APPDATA%\Python\Scripts
    echo   %APPDATA%\pypoetry\venv\Scripts
    echo.
    echo Re-run the installer, or finish the setup by hand with:
    echo   powershell -ExecutionPolicy Bypass -File "%~dp0setup-deps.ps1"
    pause
    exit /b 1
)

rem shot_clipper.entry, not shot-clipper-label-ui: entry opens the app in its
rem own window on a free port, so there's no browser tab to open here and
rem nothing to collide with if the shortcut is clicked twice. It falls back to
rem serving a browser by itself when no window toolkit is installed.
rem
rem Run as a module rather than `poetry run shot-clipper`. Console scripts only
rem exist once `poetry install` has registered them, and running an
rem unregistered one prints "The support to run uninstalled scripts will be
rem removed in a future release" - i.e. today's warning is tomorrow's hard
rem failure. `python -m` depends on nothing but the package being importable,
rem which is the same condition the app needs anyway.
"%POETRY%" run python -m shot_clipper.entry
if errorlevel 1 pause
