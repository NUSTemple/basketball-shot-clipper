# shot-clipper

Detects made basketball shots in fixed-camera video and cuts each one into
its own clip, plus a small local web UI for manually labeling those clips
goal / no-goal to build a training dataset.

Pipeline: calibrate the hoop region once per video -> run YOLO ball
detection + trajectory-through-hoop geometry to find candidate makes -> cut
each candidate into an independent clip with `ffmpeg` -> optionally review
clips in the label UI and export a goal/no-goal dataset.

See [docs/PLAN.md](docs/PLAN.md) for the full design rationale (in Chinese).

## Install

Requires [Poetry](https://python-poetry.org/) and `ffmpeg` on `PATH`.

```bash
# label UI only (lightweight - just Flask)
poetry install

# detection/clipping pipeline (adds ultralytics/opencv/numpy)
poetry install --with ml

# + test tooling
poetry install --with ml,dev
```

All commands below assume you're running from the repo root, since default
paths (`data/`, `models/`, `clips/`) are resolved relative to the current
directory.

## Detection pipeline

The labeling app's "process new video" panel (see below) wraps steps 1-3 for
you. Use these directly for scripting/batch runs, or if you'd rather not
install the app at all.

```bash
# 1. one-time hoop calibration per video -> data/configs/<name>.json
poetry run shot-clipper-calibrate path/to/video.MP4

# 2. detect candidate makes -> data/ground_truth/<name>_detected.json
poetry run shot-clipper-detect path/to/video.MP4

# 3. cut each candidate into its own clip -> clips/<name>/shot_NNN.mp4
poetry run shot-clipper-clip path/to/video.MP4 data/ground_truth/<name>_detected.json

# optional: compare detected timestamps against a hand-recorded list
poetry run shot-clipper-validate detected.json ground_truth.json
```

Model weights (`yolov8m.pt`) are expected at `models/yolov8m.pt`; download
from Ultralytics if not present. `scripts/run_batch.sh` runs steps 2-3 over
a batch of source videos.

## Using the labeling app

A local web app for going through candidate clips and marking each one
`goal` or `no_goal`, building `data/dataset/labels.json` as you go. This is
the main day-to-day way to use the project - most people won't need the raw
CLI commands above at all.

### 1. Start it

```bash
poetry install --with ml     # needed for the "process new video" panel; skip if you only label existing clips
poetry run shot-clipper-label-ui --clips-dir /path/to/clips
# open http://127.0.0.1:5050
```

`--clips-dir` points at the folder that holds one subfolder of clips per
source video (e.g. `<clips-dir>/DJI_0010/shot_001.mp4`, ...). Omit it to use
`$SHOT_CLIPPER_CLIPS_DIR` or the built-in default.

Or run it in Docker (only needs Flask - no ML deps, so it can't run step 2
below, only label clips that already exist):

```bash
CLIPS_DIR=/path/to/clips docker compose up --build
```

### 2. (Optional) Turn a raw video into candidate clips

If you already have clips in `--clips-dir`, skip to step 3 - the app loads
them automatically. To generate clips from a new source video instead,
without touching the CLI:

1. Open the **"+ Process new video"** panel at the top of the page.
2. Paste the full path to the video file and click **"Detect & cut clips"**.
3. Watch the status line - it runs ball detection (a few minutes for a
   ~100s 4K clip) then cuts each candidate make into its own file. The page
   polls progress automatically; you can keep labeling other clips while it
   runs.
4. When it finishes, the new clips are added to the list below, unlabeled
   and ready to go.

This requires a one-time hoop calibration for that video first
(`poetry run shot-clipper-calibrate path/to/video.MP4`, an OpenCV window
where you drag a box around the hoop) - if it's missing, the panel tells you
exactly which command to run.

### 3. Label clips

The video for the current clip autoplays and loops. Go through them with:

| Key | Action |
|---|---|
| `G` | mark **goal** |
| `N` or `X` | mark **no goal** |
| `Backspace` | clear the label |
| `←` / `→` | previous / next clip |
| `M` | toggle mute |
| `R` | replay from the start |

Every click/keypress saves immediately to `data/dataset/labels.json` - safe
to close the tab and resume later. The "jump to next unlabeled" checkbox
(on by default) skips straight past anything already labeled; the header
shows a running goal / no-goal / unlabeled count and a progress bar.

### 4. Export the dataset

Once you've labeled some clips, materialize them into a flat `goal/`/
`no_goal/` folder layout (symlinks by default, so it's instant and doesn't
duplicate video files):

```bash
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips
# -> data/dataset/goal/, data/dataset/no_goal/
```

## Tests

```bash
poetry install --with ml,dev
poetry run pytest
```

## Project layout

```
src/shot_clipper/     installable package (CLI tools + label_ui Flask app)
data/configs/          per-video hoop calibration (tracked)
data/ground_truth/      detector output / human-recorded make timestamps (tracked)
data/dataset/          labels.json + materialized goal/no_goal folders
models/                 YOLO weights (not tracked - see Install)
clips/                  generated shot clips (not tracked)
scripts/                batch-processing helper scripts
docs/                   design notes
tests/                  unit tests
```
