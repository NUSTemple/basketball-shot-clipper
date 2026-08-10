# Windows installer

Packages the manual install flow from the root [README.md](../README.md) into
a double-click `shot-clipper-setup.exe`, so a user ends up able to detect and
cut clips immediately - no separate Poetry/ffmpeg/model-download steps.

## Building the installer (maintainer, on Windows)

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`ISCC.exe`) on
`PATH` or in its default install location.

```powershell
installer\build-installer.ps1
```

This stages a clean `git archive` snapshot of the current commit (so
`.venv/`, `clips/`, `models/*.pt`, etc. never leak into the installer - see
`.gitignore`) into `installer\stage\`, then compiles
`installer\dist\shot-clipper-setup.exe`.

## What the installer does

1. Copies the staged repo into `%LOCALAPPDATA%\shot-clipper` (no admin
   rights needed).
2. Runs `setup-deps.ps1`, which:
   - Installs Python 3.11+, Poetry, and ffmpeg via `winget` if missing.
   - Runs `poetry install --with ml` (always the full ML pipeline, not the
     lightweight label-UI-only install - Detect must work out of the box).
   - If an NVIDIA GPU is detected (`nvidia-smi`), offers to reinstall
     torch/torchvision with CUDA support, per
     [../docs/GPU_SETUP.md](../docs/GPU_SETUP.md).
   - Downloads `models/yolov8m.pt` and `models/yolov8l.pt`.
3. Adds Start Menu and (optional) Desktop shortcuts that run
   `launch-label-ui.bat` - starts the label UI pointed at
   `%USERPROFILE%\Videos\shot-clipper\clips`, with the Detect tab's folder
   browser rooted at the user's real `Videos` folder, and opens
   `http://127.0.0.1:5050` in the browser.

Uninstalling (via *Apps & Features*) removes the app files but leaves
`data/` and `clips/` behind, since those hold the user's calibrations,
labels, and cut clips.

## Files

| File | Purpose |
|---|---|
| `installer.iss` | Inno Setup script: files, shortcuts, install steps |
| `build-installer.ps1` | Maintainer-run: stages the repo, invokes `ISCC.exe` |
| `setup-deps.ps1` | Runs at install time: winget installs, `poetry install`, CUDA prompt, model download |
| `launch-label-ui.bat` | Shortcut target: starts the label UI and opens the browser |
| `icon.ico` | App icon used by the installer, shortcuts, and uninstaller |

Testing changes to any of these requires compiling and running the installer
on an actual Windows machine or VM - `ISCC.exe` doesn't run on macOS/Linux.
