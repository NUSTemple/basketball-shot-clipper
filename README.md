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

## Labeling UI

A local web app for going through candidate clips and marking each one
`goal` or `no_goal`. Labels are saved incrementally to
`data/dataset/labels.json`.

```bash
poetry run shot-clipper-label-ui --clips-dir /path/to/clips
# open http://127.0.0.1:5050
```

Keyboard shortcuts: `G` goal, `N`/`X` no goal, `Backspace` clear,
`←`/`→` navigate, `M` mute, `R` replay.

Or run it in Docker (only needs Flask, no ML deps):

```bash
CLIPS_DIR=/path/to/clips docker compose up --build
```

### Process a new video straight from the UI

The "+ Process new video" panel runs detection + clipping server-side (in a
background job you can watch progress on) so you can go from a raw video
straight to labeling without touching the CLI. It requires:
- the `ml` extras installed in the process running the label UI
  (`poetry install --with ml`; not available in the plain Docker image above)
- an existing hoop calibration for that video at
  `data/configs/<video_stem>.json` - run `shot-clipper-calibrate` first if
  there isn't one yet

New clips land in `<clips-dir>/<video_stem>/` and show up in the labeler
automatically once the job finishes.

Once you've labeled some clips, materialize them into a flat goal/no_goal
folder layout (symlinks by default):

```bash
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips
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
