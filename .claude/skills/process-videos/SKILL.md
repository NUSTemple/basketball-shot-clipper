---
name: process-videos
description: Process a folder of new basketball videos through the shot-clipper pipeline (calibrate check, detect, cut, filter) and hand off to the label UI for goal/no-goal review and 1-5 star rating, then export the best-rated highlights for a video edit. Use when the user gives a folder of new videos and wants goal clips cut and reviewed.
---

# Process new videos into rated highlight clips

Full pipeline for turning a folder of raw basketball videos into a curated
set of star-rated highlight clips, using this repo's existing tools
(`shot-clipper-calibrate`/`detect`/`clip`/`label-ui`/`build-dataset`). Run
everything from the repo root with `poetry run <command>`.

The user will give you a folder path containing new source videos. If they
haven't, ask for it before doing anything else.

## Step 1: Find the videos and check calibration

List video files in the given folder (`*.MP4`, `*.mp4`, `*.MOV`, `*.mov`).
For each one, check whether it already has a hoop calibration at
`data/configs/<video_stem>.json` (stem = filename without extension).

- **Calibrated already**: proceed to Step 2 for that video.
- **Not calibrated**: this needs a human to drag a box around the hoop in an
  interactive OpenCV window - you can't do this yourself. Tell the user
  which videos need it and give them the exact command:
  `poetry run shot-clipper-calibrate "<path>"`. Ask whether they want to
  calibrate now (wait for them) or skip those videos for this run.

## Step 2: Confirm the clips directory

Ask the user which `--clips-dir` the label UI is pointed at (or check
`$SHOT_CLIPPER_CLIPS_DIR`) - clips must land there or the label UI won't see
them. Use that same path for `--outdir` in Step 3 below, as
`<clips-dir>/<video_stem>`.

## Step 3: Detect + cut + filter, per calibrated video

Check whether `models/shot_filter.joblib` exists (the trained false-positive
filter, see `shot-clipper-train-filter` / README "Improving precision with a
trained filter"). For each calibrated video, run:

```bash
poetry run shot-clipper-detect "<video_path>"
poetry run shot-clipper-clip "<video_path>" \
  "data/ground_truth/<video_stem>_detected.json" \
  --outdir "<clips-dir>/<video_stem>" \
  --filter-model models/shot_filter.joblib   # omit if it doesn't exist yet
```

If `models/shot_filter.joblib` is missing, still cut the clips (omit
`--filter-model`) but tell the user precision will be lower without it -
more clips to manually reject as no_goal - and that they can train one later
with `shot-clipper-train-filter` once they've labeled enough clips.

These can take a few minutes per video (YOLO ball detection over the whole
clip). Report a running summary as each video finishes: candidates detected,
and if filtered, how many survived vs. got dropped
(`shot-clipper-clip`'s own output already reports both counts).

## Step 4: Hand off to the label UI

Once clips are cut, tell the user to review them:

```bash
poetry run shot-clipper-label-ui --clips-dir "<clips-dir>"
```

They mark each clip goal or no_goal, and for goals, rate 1-5 stars (how
good/highlight-worthy the make is - pressing a number key both marks goal
and rates in one step). Remind them of the shortcuts if useful: `G`/`N`/`X`
label, `1`-`5` rate (marks goal too), `0` clears a rating, `Backspace` clears
the label entirely. This step needs a human watching the clips - don't try
to do it yourself. Wait for them to say they're done (or ask them to let you
know when they've finished this round).

## Step 5: Export the highlights

Once they're done reviewing, ask what star threshold counts as "good enough
for my video" (suggest 4 as a reasonable bar for a highlight reel), then run:

```bash
poetry run shot-clipper-build-dataset --clips-dir "<clips-dir>" --min-stars <N>
```

This symlinks every goal clip rated >= N stars into `data/dataset/goal/`,
named `<video>__shot_NNN_<stars>star.mp4` - ready to pull straight into a
video editor. Tell them where the output landed and how many clips made the
cut (the command prints a per-rating breakdown too).
