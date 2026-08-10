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
   - Records Poetry's absolute path to `installer\poetry-path.txt` (see
     "Finding Poetry at launch" below).
   - Runs `poetry install --with ml,desktop` (always the full ML pipeline,
     not the lightweight label-UI-only install - Detect must work out of the
     box - plus pywebview for the app window).
   - If an NVIDIA GPU is detected (`nvidia-smi`), offers to reinstall
     torch/torchvision with CUDA support, per
     [../docs/GPU_SETUP.md](../docs/GPU_SETUP.md).
   - Downloads `models/yolov8m.pt` and `models/yolov8l.pt`.
3. Adds Start Menu and (optional) Desktop shortcuts that run
   `launch-label-ui.bat` - opens the app in its own window on a free port,
   pointed at `%USERPROFILE%\Videos\shot-clipper\clips`, with the Detect
   tab's folder browser rooted at the user's real `Videos` folder.

## Finding Poetry at launch

The launcher used to be a bare `where poetry`, and it failed for people whose
install had just succeeded — the reported symptom was *"stuck at poetry not
installed"* on first launch.

Poetry's installer writes its bin directory into the user `PATH` in the
registry, but Explorer caches the environment at login and hands that stale
copy to every shortcut it launches. So the Start Menu entry sees no Poetry
until the user logs out and back in. `setup-deps.ps1` hid this from itself by
patching its own process `PATH` (`Refresh-Path`), which is the worst possible
split: setup reports success, then the shortcut fails.

`launch-label-ui.bat` now resolves Poetry from three sources, cheapest first,
and only fails if all three miss:

1. `installer\poetry-path.txt`, written by `setup-deps.ps1` at install time —
   the only source that can't go stale (a path that no longer exists is
   ignored, so a moved or removed Poetry falls through rather than wedging).
2. `PATH`, for a shell that has it.
3. The known install locations (`%APPDATA%\Python\Scripts`,
   `%APPDATA%\pypoetry\venv\Scripts`, …).

It then runs `python -m shot_clipper.entry` rather than `poetry run
shot-clipper`. Console scripts only exist once `poetry install` has
registered them, and Poetry already warns that running an unregistered one
"will be removed in a future release" — today's warning is tomorrow's hard
failure. `python -m` needs nothing but an importable package.

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
