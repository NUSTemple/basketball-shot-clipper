# shot-clipper

[中文](README.zh-CN.md)

Detects made basketball shots in fixed-camera video and cuts each one into
its own clip, plus a small local web UI for manually labeling those clips
goal / no-goal to build a training dataset.

Pipeline: calibrate the hoop region once per video -> run YOLO ball
detection + trajectory-through-hoop geometry to find candidate makes -> cut
each candidate into an independent clip with `ffmpeg` -> optionally review
clips in the label UI and export a goal/no-goal dataset.

See [docs/PLAN.md](docs/PLAN.md) for the full design rationale (in Chinese).

**Using Claude Code?** `.claude/skills/process-videos` automates the whole
"new videos in -> rated highlights out" flow described below - point it at a
folder of new videos and it drives detect/cut/filter for you, then hands off
to the label UI for review and helps you export the best-rated clips.

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
a batch of source videos. `shot-clipper-train-filter` (see below) separately
expects `models/yolov8l.pt`.

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

### 3. Label clips - and rate your goals

The video for the current clip autoplays and loops. Go through them with:

| Key | Action |
|---|---|
| `G` | mark **goal** (stays on the clip so you can rate it) |
| `N` or `X` | mark **no goal** (auto-advances) |
| `1`-`5` | rate this clip N stars - marks it goal too if it wasn't already, then advances |
| `0` | clear the star rating (keeps the goal label) |
| `Backspace` | clear the label entirely (and any rating) |
| `←` / `→` | previous / next clip |
| `M` | toggle mute |
| `R` | replay from the start |

Stars (1-5) capture how good/highlight-worthy a make is - rate the ones
you'd actually want in a video edit higher. A clip only counts as "done" once
it's `no_goal`, or `goal` *and* rated - so `no_goal` still auto-advances
immediately, but marking `goal` pauses on the clip until you press a number
key. "jump to next incomplete" (on by default) skips straight to whatever
still needs a label or a rating; the header tracks goal / no-goal / needs
rating / unlabeled counts and a progress bar. Every action saves immediately
to `data/dataset/labels.json` - safe to close the tab and resume later.

### 4. Export the dataset (or just your best highlights)

Once you've labeled some clips, materialize them into a flat `goal/`/
`no_goal/` folder layout (symlinks by default, so it's instant and doesn't
duplicate video files). Goal clips get their rating in the filename, e.g.
`DJI_0001_D__shot_012_5star.mp4`:

```bash
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips
# -> data/dataset/goal/, data/dataset/no_goal/

# only your best-rated goals (e.g. for pulling into a video edit) - no_goal is still exported in full
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips --min-stars 4
```

## Improving precision with a trained filter

`find_makes()` (the geometry rule that decides which ball trajectories look
like a make) is deliberately recall-first (docs/PLAN.md decision 5) - it
over-generates candidates rather than risk missing a real make, which means
a lot of what it flags isn't actually a goal. Once you've labeled enough
clips, train a small classifier to filter those false positives back out:

```bash
poetry install --with ml
poetry run shot-clipper-train-filter --clips-dir /path/to/clips
```

This extracts two complementary feature sets for every labeled clip -
trajectory features (descent speed, dwell time in the hoop box, bounce-back
signal, etc. - `src/shot_clipper/features.py`) and net-motion features
(pixel motion in the net region right after the candidate event - shape of
the motion burst, optical-flow directional coherence - `src/shot_clipper/net_motion.py`,
needs no ball detection) - evaluates the combination with leave-one-video-out
cross-validation, and picks a confidence threshold that keeps recall >= 98%
(`--min-recall` to change) while maximizing precision. It prints a
before/after precision report per video and saves the model to
`models/shot_filter.joblib`.

**Results on the 307-clip labeled dataset** (9 videos): baseline candidate
precision is 34.2%. Two things mattered to get a real improvement out of
this:
- Ball detection quality: the default `yolov8m` detector is too sparse for
  feature extraction - 19% of real goals had no usable above-hoop ->
  through-hoop trajectory at all, capping what any filter could safely do.
  `shot-clipper-train-filter` defaults to the larger `models/yolov8l.pt` for
  feature extraction specifically (100% of real goals get a usable
  trajectory with it, vs. 81%; not validated as the primary detector for
  `shot-clipper-detect` itself).
- Net motion alone carries almost no signal (34.3%, basically baseline) but
  is clearly complementary to trajectory features - combined, they roughly
  double how many false positives the filter can safely drop at the same
  recall target, versus trajectory features alone.

At the default 98%-recall threshold, the deployed model gets **34.2% ->
41.7% precision**, dropping 60 of 307 candidates and risking 2 of 105 real
goals. If you're willing to trade more recall for precision, the achievable
ceiling is much higher - rerun with `--min-recall <x>` to pick a different
point on that curve; the report table shows exactly how many candidates
would be dropped and how many real goals would be at risk at each setting.

Once trained, it's used automatically:
- `shot-clipper-clip --filter-model models/shot_filter.joblib` scores each
  clip right after cutting it (net-motion features need actual pixels, so
  filtering happens after clipping, not during detection) and deletes the
  ones below the threshold; dropped clips are logged to stdout with their
  score.
- The label UI's "process new video" panel applies it automatically whenever
  `models/shot_filter.joblib` exists (pass `"use_filter": false` in the
  `/api/process-video` request body to opt out for one run).

## Tests

```bash
poetry install --with ml,dev
poetry run pytest
```

## Project layout

```
src/shot_clipper/      installable package (CLI tools + label_ui Flask app)
data/configs/          per-video hoop calibration (tracked)
data/ground_truth/      detector output / human-recorded make timestamps (tracked)
data/dataset/          labels.json + materialized goal/no_goal folders
models/                 YOLO weights (not tracked - see Install)
clips/                  generated shot clips (not tracked)
scripts/                batch-processing helper scripts
docs/                   design notes
tests/                  unit tests
.claude/skills/         Claude Code skills (tracked - see process-videos above)
```
