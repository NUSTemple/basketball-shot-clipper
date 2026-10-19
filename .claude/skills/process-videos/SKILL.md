---
name: process-videos
description: Process a folder of new basketball videos through the shot-clipper pipeline (calibrate check, detect, cut, optional filter) and hand off to the label UI for goal/no-goal review and 1-5 star rating, then export the best-rated highlights for a video edit. Use when the user gives a folder of new videos and wants goal clips cut and reviewed.
---

# Process new videos into rated highlight clips

Full pipeline for turning a folder of raw basketball videos into a curated
set of star-rated highlight clips, using this repo's existing tools
(`shot-clipper-calibrate`/`detect`/`clip`/`label-ui`/`build-dataset`). Run
everything from the repo root with `poetry run <command>`.

Ball detection uses OpenHoopVision's basketball-trained RF-DETR model
(`models/rf-detr-basketball.pth`). It downloads itself (~128MB,
sha256-checked) on the first detection run, so the first video takes a
minute longer.

The user will give you a folder path containing new source videos. If they
haven't, ask for it before doing anything else.

## Step 1: Find the videos and check calibration

List video files in the given folder (`*.MP4`, `*.mp4`, `*.MOV`, `*.mov`).
For each one, note duration and resolution (`ffprobe`), and check whether it
already has a hoop calibration at `data/configs/<video_stem>.json` (stem =
filename without extension). Also mention the resolution if it differs from
recent footage. A different camera or setting matters for Step 3's filter.

- **Calibrated already**: go on to Step 2 for that video.
- **Not calibrated**: a human has to mark the hoop. You can't, and the
  detected-rim box is not a substitute (see below). The user has two ways:
  - In the browser: start the label UI (Step 4's command) and use the
    Detect tab's calibration panel. This is usually easiest.
  - The OpenCV window: `poetry run shot-clipper-calibrate "<path>"`.

  Ask whether they want to calibrate now (wait for them) or skip those
  videos for this run.

**One calibration per camera position, not per video.** A session usually
splits into several files from the same tripod. Once the user has
calibrated one, extract a frame from each sibling video and look at them
side by side with the hoop box drawn on. If the camera clearly didn't move,
show the user, and with their OK reuse the same box:
`poetry run shot-clipper-calibrate "<path>" --bbox x1,y1,x2,y2` (pixel
coordinates from the first video's config, same resolution). If the camera
moved even slightly, that video needs its own calibration.

Why not auto-calibrate from the detected rim: `find_makes()`'s geometry
thresholds are fractions of the hand-drawn box, and people draw it bigger
than the rim. Measured against the detector's rim box, hand boxes were
1.9-2.6x wider and 2.4-6x taller, varying by camera and by who drew it. A
rim-sized box would quietly shrink every window and cost recall.

## Step 2: Confirm the clips directory

If the user named an output folder, use it. Otherwise ask which
`--clips-dir` the label UI is pointed at (or check `$SHOT_CLIPPER_CLIPS_DIR`).
Clips must land there or the label UI won't see them. Use
`<clips-dir>/<video_stem>` as `--outdir` in Step 3.

**Re-runs replace clips.** Cutting into an `--outdir` that already has clips
deletes the old `shot_*.mp4` (and its score files) first. Labels are keyed
by clip filename, so labels made on the old clips would then point at
different footage. If a video's folder already has labeled clips, confirm
with the user before re-running it.

## Step 3: Detect + cut (+ optional filter), per calibrated video

```bash
poetry run shot-clipper-detect "<video_path>"
poetry run shot-clipper-clip "<video_path>" \
  "data/ground_truth/<video_stem>_detected.json" \
  --outdir "<clips-dir>/<video_stem>"
```

**The trained filter (`--filter-model models/shot_filter.joblib`) is
optional, and off by default for new setups.** It was trained on the
owner's July-September 2026 4K DJI clips, with features extracted by YOLO
(`models/yolov8l.pt`). It's untested on other cameras and resolutions, and
since it *deletes* the clips it rejects, a mismatch silently loses real
makes. Offer it only when the footage matches that original setup and the
user wants fewer clips to review. Ask before adding it.

What to expect, so the numbers don't look like a bug:
- The detector is recall-first and the geometric shot rule over-generates.
  Expect several candidates per minute of play: about 4-5 per minute on a
  1080p side-on camera, fewer on the original 4K DJI view. Most are misses
  for the user to reject.
- Candidates less than 5s apart are merged into one longer clip, so the clip
  count is lower than the "detected N makes" count.
- Speed on the M3 Max (MPS): detection runs about 2.7x faster than real time
  on 1080p and about 0.5x on a 4K crop. Cutting is about 3s per clip. A long
  1080p session (30-40 min) takes roughly 15-20 min plus cutting.

Run long videos in the background and report a running summary as each
finishes: candidates detected, clips cut (and, if filtered, kept vs.
dropped, which `shot-clipper-clip` prints itself).

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

Review shows an "AI: likely make / likely miss / unsure" badge, and "sort by
confidence" uses it, only where a video folder has a
`_prelabel_scores.json`. This pipeline doesn't produce one: those come from
a classifier trained on that camera's own labeled clips. Don't promise
badges for freshly cut videos.

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
