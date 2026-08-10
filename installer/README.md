# Windows installer

Packages the manual install flow from the root [README.md](../README.md) into
a double-click `shot-clipper-setup.exe`, so a user ends up able to detect and
cut clips immediately - no separate Poetry/ffmpeg/model-download steps.

## Building the installer (maintainer, on Windows)

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`ISCC.exe`) on
`PATH` or in its default install location.

```powershell
installer\build-installer.ps1              # version falls back to installer.iss's default
installer\build-installer.ps1 -Version 0.2.0
```

This stages a clean `git archive` snapshot of the current commit (so
`.venv/`, `clips/`, `models/*.pt`, etc. never leak into the installer - see
`.gitignore`) into `installer\stage\`, then compiles
`installer\dist\shot-clipper-setup.exe`.

## Cutting a release

`.github/workflows/release.yml` builds on a `windows-latest` runner, so
releases don't need a Windows machine. It publishes a GitHub Release **only
for tag pushes** - a `workflow_dispatch` run just uploads the exe as a run
artifact, which is the way to smoke-test a build without publishing:

```bash
git tag -a v0.1.0 -m "shot-clipper 0.1.0"
git push origin v0.1.0
```

The tag's version (minus the leading `v`) is stamped into the exe, so keep
tags to plain `vX.Y.Z`.

## SmartScreen

The exe is unsigned, so Defender SmartScreen shows *"Windows protected your
PC"* on first run and users must click **More info** > **Run anyway** (and
possibly **Unblock** the file in its Properties first). The release notes and
the root README say so. This is not fixable by build flags - it needs an
Authenticode signature:

| Option | Cost | Effect |
|---|---|---|
| Nothing (today) | free | Warning on every download until the exe accrues reputation, which resets each new build |
| [Azure Trusted Signing](https://learn.microsoft.com/azure/trusted-signing/) | ~$10/month | Cheapest real fix; needs an identity check (individual validation available) |
| OV certificate | ~$200-400/yr | Signs, but reputation still has to accumulate |
| EV certificate | ~$400-600/yr | Immediate SmartScreen trust; requires hardware token or Azure Key Vault |

If a certificate is ever obtained, Inno Setup signs via a `SignTool`
directive in `installer.iss` plus a signing step in the workflow.

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
