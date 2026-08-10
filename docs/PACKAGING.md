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

## 2. An ONNX inference path, torch kept as the reference — **done**

This dominates the download size. A PyInstaller bundle carrying CUDA torch is
roughly **3 GB** (the `cu132` wheels bundle cuDNN/cuBLAS), so ~1.3 GB
compressed, and it only helps NVIDIA users. `onnxruntime-directml` is ~120 MB
and runs on any DirectX 12 GPU.

Both backends now exist behind [`inference.py`](../src/shot_clipper/inference.py),
selected by `$SHOT_CLIPPER_INFERENCE` (`torch` | `onnx`, default: torch if
installed). torch stays the reference implementation — every detection result
in this repo was produced with it — and is the rollback if ONNX ever drifts.

Everything funnels through `detect_shots.detect_ball_centers_batch`, so
`run_detection`, the clip filter and `train_filter` all switch together.

**NMS is skipped in the ONNX path, deliberately.** Every consumer keeps only
the single highest-confidence box per frame, and NMS never removes the
top-scoring box — it only drops lower-scoring overlaps. So running it would
cost time and change nothing. That assumption is load-bearing and is written
down at the top of `inference.py`; a caller that ever wants the full box set
has to revisit it.

### Validation

`shot-clipper-compare-backends <video>` decodes each frame **once** and feeds
the identical pixels to both backends, so any difference is inference rather
than decode, sampling or crop geometry. It reports per-frame detection
agreement, centre distance where both fired, and the resulting makes — in
that order, because makes are the number that matters but the least sensitive
signal (a geometric rule can absorb or amplify per-frame disagreement).

Measured on this machine (RTX 4070, yolov8l, full videos):

| video | frames | torch dets | onnx dets | max centre dist | makes |
|---|---|---|---|---|---|
| `…185229_0012_D` | 10,405 | 672 | 678 (+6) | 0.07 px | 38 vs 38, **all matched** |
| `…173300_0005_D` | 16,951 | 749 | 752 (+3) | 0.25 px | 33 vs 32, **3 differ** |

**ONNX is not bit-identical to torch.** On the second video it produced one
fewer make: two of torch's makes are absent and one is new. That is a real
difference, not measurement noise — see the noise floor below.

What the numbers rule out. ONNX found a strict *superset* of torch's
detections on both videos (0 torch-only frames), and every shared box centre
agreed to within a quarter of a pixel. A preprocessing mismatch — wrong
letterbox padding, wrong channel order, wrong rescale — cannot produce that;
it would shift centres by whole pixels and lose detections, not gain them.
The pad arithmetic was checked line-by-line against `ultralytics.utils.ops
.scale_boxes` and matches, including its asymmetric `round(x - 0.1)` split.
TF32 was ruled out directly: `allow_tf32` is already off in this torch build,
and forcing cuDNN to fp32 changed timing but not one result.

What remains is ordinary numerical difference between CUDA and DirectML
convolution kernels, landing on candidates that sit right at
`BALL_CONF_THRESHOLD = 0.1`. `find_makes()` then amplifies it: an extra ball
sighting outside the horizontal window *resets* `above_sighting`, so more
detections can produce **fewer** makes.

### The noise floor is zero

The obvious defence — "the pipeline is already unstable at this threshold, so
this is within its usual churn" — was tested and is false. Running the same
torch backend twice with `SHOT_CLIPPER_BATCH_SIZE=16` and `=8` on the second
video gave 33 makes both times, every one matched. So the pipeline is
deterministic run-to-run, and the 3 differing makes are genuinely
attributable to the backend swap.

### Open question before torch can be dropped

Whether ONNX is *worse* is still unknown. The detector is recall-first by
design and only ~34% of its candidates are real makes (PLAN.md decision 5),
so losing one net make could easily be losing a false positive. Deciding
needs the three differing candidates cut and labelled by hand.

Until that happens, **torch stays the default** and ONNX is opt-in via
`SHOT_CLIPPER_INFERENCE=onnx`.

**Speed:** ONNX/DirectML ran at 5.9–7.7 ms/frame against torch/CUDA's
4.5–6.0 ms — roughly 28% slower at inference. Because detection is
decode-bound (~77% decode / ~23% inference), that is about a 6% end-to-end
cost on this hardware, for an install roughly a tenth the size that also
works on AMD and Intel GPUs.

### Trap: exporting clobbers the DirectML runtime

`YOLO.export(format="onnx")` auto-installs plain `onnxruntime` if it isn't
present, which **overwrites `onnxruntime-directml`** — both packages provide
the same `onnxruntime` module. The symptom is silent: providers drop to
`['AzureExecutionProvider', 'CPUExecutionProvider']` and inference quietly
moves to the CPU. Check with:

```powershell
poetry run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expect `DmlExecutionProvider` first. If it's missing:

```powershell
poetry run python -m pip uninstall -y onnxruntime onnxruntime-directml
poetry run python -m pip install onnxruntime-directml
```

This is only a problem on a machine that both exports and runs. In the build
pipeline the export is a separate step from the packaged app, which never
carries ultralytics at all.

### Still to do here

1. Label the three candidates the two backends disagree on, and decide
   whether ONNX's answer is acceptable. This gates everything below.
2. Only then flip the default to ONNX.
3. Dropping torch and ultralytics from the *runtime* is what actually buys
   the size. That happens in step 4, when the bundle is assembled, which is
   why they are still installed today.

## 3. A window instead of a browser tab — **done**

Flask stays on 127.0.0.1, wrapped in **pywebview**
([`desktop.py`](../src/shot_clipper/desktop.py)), which uses WebView2 —
preinstalled on Windows 11, bootstrapper-installable on 10. A few MB, against
Electron's ~150 and a front-end rewrite.

- `shot-clipper` opens the window; `shot-clipper --server` serves a plain
  browser, which is what Docker's CMD and headless runs use. Without a window
  toolkit installed, window mode says so and degrades to the server rather
  than failing.
- **Ephemeral port, not 5050.** A window addresses itself, so there's no
  reason to squat on a fixed port that a second copy or an unrelated dev
  server would collide with.
- Flask runs in a daemon thread; the window owns the main thread, because
  every GUI toolkit here requires that. Closing the window ends the process —
  a running detection job is a separate process by design (jobs.py), so it
  survives and reappears in Job Status.

**Native file pickers, everywhere.** The UI could only open a real file
dialog on macOS, via osascript; Windows and Docker had none, leaving "type or
paste a path" as the only option. A window owns a real toolkit, so
[`native_dialog.py`](../src/shot_clipper/native_dialog.py) now brokers
between whatever shell is hosting the app — the window's own dialog, then
osascript, then nothing — and `/api/capabilities` reports which, so the front
end can hide a button rather than offer one that always errors.

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
