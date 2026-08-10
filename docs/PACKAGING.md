# Packaging shot-clipper as a native Windows app

Today's `shot-clipper-setup.exe` (see [../installer/README.md](../installer/README.md))
is a *bootstrapper*, not a self-contained app: it ships the source tree, then
at install time uses `winget` to install Python, Poetry and ffmpeg, runs
`poetry install --with ml`, pip-pulls CUDA torch and downloads YOLO weights.
The "app" is a `.bat` that starts Flask and opens a browser tab.

A native app means no network at install time, no Python on the user's
machine, and a real window. That's four pieces of work, in this order.

## 1. Path and process assumptions — **done**

Freezing breaks three things that a checkout never notices. All three are
fixed, and none of them changed behaviour in a checkout or in Docker.

- **Bare relative paths.** `Path("data/configs")`, `models/yolov8l.pt` and
  friends were resolved against the working directory. They now come from
  [`paths.py`](../src/shot_clipper/paths.py), which splits two roots that are
  the same directory today and different ones once frozen: `resource_dir()`
  for read-only bundled payload, `app_root()` for everything the user
  creates. Frozen, `app_root()` moves to `%LOCALAPPDATA%\shot-clipper` so an
  upgrade replacing the install directory can't take the user's calibrations
  and labels with it.
- **`sys.executable` worker spawn.** `jobs._spawn_worker` ran
  `[sys.executable, "-m", "shot_clipper.label_ui.worker", ...]`. Frozen,
  `sys.executable` is the app's own launcher and has no `-m`.
  [`entry.py`](../src/shot_clipper/entry.py) is now the single entry point:
  it starts the label UI normally, or a job worker when passed `--worker`,
  and owns the argv builder both halves share.
- **ffmpeg via PATH, and console flashes.**
  [`external.py`](../src/shot_clipper/external.py) resolves ffmpeg/ffprobe
  from `$SHOT_CLIPPER_FFMPEG` → a bundled `bin/` → PATH, and passes
  `CREATE_NO_WINDOW` so a windowed build doesn't flash a console for every
  clip cut.

Environment overrides, all optional:

| Variable | Effect |
|---|---|
| `SHOT_CLIPPER_HOME` | Writable root (`data/`, `models/` hang off it) |
| `SHOT_CLIPPER_DATA_DIR` | Just the data directory |
| `SHOT_CLIPPER_MODELS_DIR` | Just the models directory |
| `SHOT_CLIPPER_CLIPS_DIR` | Default clips folder |
| `SHOT_CLIPPER_MEDIA_ROOT` | Where the in-app folder browser is rooted |
| `SHOT_CLIPPER_FFMPEG` / `SHOT_CLIPPER_FFPROBE` | Explicit binary paths |

## 2. Decide what happens to torch — **not started**

This dominates the download size. A PyInstaller bundle carrying CUDA torch is
roughly **3 GB** (the `cu132` wheels bundle cuDNN/cuBLAS), so ~1.3 GB
compressed, and it only helps NVIDIA users.

The alternative is exporting the YOLO weights to ONNX and running inference
on `onnxruntime-directml`: drops torch *and* ultralytics from the runtime,
lands around 300–400 MB, works on AMD and Intel GPUs too, and deletes the
CUDA-swap branch of `setup-deps.ps1` along with the `+cu132` footgun
documented in [GPU_SETUP.md](GPU_SETUP.md). The cost is re-validating
detection output against the current model and re-implementing the
pre/post-processing (letterbox + NMS) that ultralytics does today.

Detection is decode-bound rather than inference-bound (~77% decode / ~23%
inference), so leaving CUDA behind costs less throughput than it looks like
it should.

## 3. A window instead of a browser tab — **not started**

Keep Flask on 127.0.0.1 and wrap it in **pywebview**, which uses WebView2 —
preinstalled on Windows 11, bootstrapper-installable on 10. Costs a few MB
and gets a real titled window and taskbar identity. Electron or Tauri would
mean rewriting the front end for no gain here.

## 4. Freeze and install — **not started**

**PyInstaller `--onedir`**, not `--onefile`: onefile re-extracts hundreds of
MB to temp on every launch and reliably trips AV heuristics. Target
`shot_clipper.entry:main`. Ship `ffmpeg.exe`/`ffprobe.exe` in `bin/` and the
weights in `models/` inside the bundle — `external.py` and `paths.py` already
look there first.

The PyInstaller output directory then replaces the staged `git archive` as
`installer.iss`'s `[Files]` source. Most of that file survives as-is; what
goes away is the entire `[Run]` setup-deps step, and with it winget, Poetry
and the network dependency at install time.

Two things to plan for:

- **Migration.** Existing installs keep their data under
  `%LOCALAPPDATA%\shot-clipper\data` *inside* the app directory, because
  they run unfrozen with the working directory as the root. A frozen build
  resolves the same-named directory per-user; check whether those coincide on
  a real upgrade before shipping, and copy forward if they don't.
- **Signing.** Unsigned PyInstaller binaries get flagged by AV engines
  noticeably more often than an unsigned Inno installer does, so the
  SmartScreen table in [../installer/README.md](../installer/README.md)
  becomes more pressing, not less.
