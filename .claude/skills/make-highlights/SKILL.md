---
name: make-highlights
description: Turn a folder of raw basketball session videos into shot clips, AI make scores, AI star ratings, goals grouped by player and three finished highlight reels, end to end with no human review - on any local machine (macOS, or Windows/Linux with an NVIDIA GPU). Use when someone gives a video folder and wants highlight videos and/or clips, or wants the whole pipeline set up and run locally.
---

# Session videos -> clips + highlight reels, locally and unattended

Input: a folder of raw videos, e.g. `D:\Videos\20261024 Basketball Video\DJI_001`
or `/Users/me/Videos/20261024 Basketball Video/DJI_001`. Output: every shot cut
as a clip, AI make scores and stars, goals grouped by player, and three
different reels of the best goals. Users want minimum intervention - don't
stop to ask what the defaults answer.

Everything is one resumable command, run from the repo root
(`src/shot_clipper/highlights.py`):

```bash
poetry run shot-clipper-highlights "<folder>" [--out "<dir>"]
```

## Step 0: is this machine ready? (every first run on a machine)

```bash
poetry run shot-clipper-doctor
```

- All OK: go to step 1.
- **FAIL**, or `poetry` isn't even installed (a fresh clone): follow the
  `setup-local` skill first - it installs everything (on Windows via
  `installer/setup-local-pipeline.ps1`) and ends with doctor passing.
- **WARN on "AI stars"** (no vision model): ask once whether to set it up
  (`ollama pull qwen3-vl:30b`, ~20GB download, needs a 24GB+ GPU) or run now
  with rougher stars (`--no-vlm`). Other WARNs are fine to run with.

## Step 1: run it

- **Where output goes**: default `<folder>/../auto_highlights/`. If the user
  names a clips folder (e.g. "clips in `<session>/clips`"), pass
  `--out "<that folder's parent>"`. Cutting **clears** each
  `<out>/clips/<video>/` it writes, so never point `--out` at a folder whose
  clips already carry labels from an older pipeline run - check first.
- It's long - run it in the background and tell the user roughly how long:
  detection is the slow part (~1.5-2h for a 20GB session on an M3 Max; an
  RTX 5090 is much faster), then make scoring, AI stars (~10-15s a goal),
  player grouping, 3 renders (~1 min each). On macOS wrap it in
  `caffeinate -ims`; on Windows the command itself keeps the PC from
  sleeping (it should stay plugged in); on Linux ask the user to.
  Re-running resumes from where it stopped.
- One longer reel instead of three: `--variants top_plays --count 48
  --spread 10` (~3.5 min; the best 3 goals of every 10 minutes first, so
  every game is in it). Keep it under ~60 clips (~5 min).
- Flags: `--skip <video> ...` (leave videos out, e.g. ones the user says
  were recorded after the game), `--no-vlm` (no vision model), `--no-players`, `--seed N` (a fresh
  set of three), `--variants top_plays,countdown`, `--photos-album ""`.

What it does: auto-calibrates each video from the rim RF-DETR finds (a
video with no rim is skipped and reported - the only case needing a human:
`shot-clipper-calibrate`), detects and cuts shots, scores make/miss with
`models/prelabel/global.joblib` (>= 0.6 = goal; also finds which shot in a
merged clip went in), rates goals with Qwen3-VL (MLX on a Mac, Ollama
elsewhere) into 1-5 AI stars by rank, groups goals by shooter
(`person_id`, suggested names from `data/players/gallery.jsonl`), renders
`top_plays` (best ~20 in game order), `countdown` (top 10, #1 last with a
zoom replay) and `fast_cut` (shuffled, short, no slow motion) with Remotion,
and on macOS adds them to the Photos album "Basketball Highlights". Reels
only use goals with make score >= 0.8 unless a person labeled them.

## Step 2: check, then report

- `<out>/summary.json`: videos skipped, goals, star counts, players, reels.
- Read each `<out>/reels/<name>.jpg` contact sheet: no black/frozen frames,
  titles and end card present.

Report: where the clips and the three reels are (and the Photos album on a
Mac), reel lengths, goals per video, any skipped video and why, AI star
breakdown, goals by player (suggested names vs "Player N" groups). Then
offer the review loop below.

Before running, ask about anything the user usually wants that isn't in the
request: videos recorded after the game (`--skip`), one long reel vs three,
uploading the clips for hosted review, a NAS/backup copy. Do them all in
one go after the run rather than waiting to be asked.

If a render fails with `DataCloneError ... out of memory`, the reel has too
many clips for one browser tab - the render step already caps Remotion's
video cache at 2GB (`--media-cache-size-in-bytes`); keep reels under ~60 clips.

## Step 3 (optional): the user reviews, the reels follow

```bash
poetry run shot-clipper-label-ui --clips-dir "<out>/clips"
```

Each video folder gets `_ai_suggestions.json` (AI stars, in-a-reel, suggested
scorer), so Review only shows what needs a person - the reel's goals and
clips the make model is unsure of (0.4-0.8) - with the AI's call on each:
Enter accepts it (goal + stars + scorer, one write), G/N/1-5 overrides.
Players hides groups under 3 goals. Review = goal/miss + stars; Players =
confirm one name per group. Labels go
to `data/dataset/labels.json` and the next `shot-clipper-highlights` run
picks them up automatically: the person's misses are dropped from reels,
their stars outrank the AI's, their scorer tags name the groups and feed the
gallery. The re-run only re-renders (~2 min).

Each folder also gets `_player_crops/<shot>.jpg` (the shooter cut out, made
by `shot-clipper-players`): the hosted Players tab sends these as files -
cutting them there took 1-6s each and left the tab mostly black. A folder
from before this needs `shot-clipper-players "<out>/clips"` once (~30s,
embeddings are cached) and a re-upload.

Labels made in the **hosted** app instead: download them
(`gcloud storage cp gs://shot-clipper-data/shared/data/labels.json
data/prelabel/training/shared_labels_<date>.json`) and add
`--labels <that file> --label-prefix <account_slug>/`. Clips can be
uploaded for hosted review with `gcloud storage rsync -r -x '.*\.DS_Store$'
"<out>/clips" "gs://shot-clipper-data/clips/<account_slug>"` (deleted
after 7 days). `<account_slug>` is the owner's sign-in email with `@`
replaced by `_`. Upload everything in the clips folder - `_prelabel_scores.json`,
`_players.json`, `_ai_suggestions.json` and `_player_crops/` are what make
the hosted Review/Players fast. Both need access to the project's GCP bucket.
If `gcloud` crashes with `module 'ssl' has no attribute 'PROTOCOL_TLSv1'`,
a Homebrew Python upgrade broke it: run it with
`CLOUDSDK_PYTHON=/opt/homebrew/bin/python3.12` (any 3.10-3.13).

## Backing up the session folder (NAS)

`cp -R "<session folder>" "<NAS share>/"`, then compare file counts and
sizes on both sides. Prefer `cp` to macOS's built-in `rsync` (openrsync,
~half the speed over SMB). On a Mac with both Ethernet and Wi-Fi on the same
network, an SMB share mounted while on Wi-Fi stays on Wi-Fi: check
`netstat -an | grep 445` shows the Ethernet address - over Wi-Fi a 44GB
session took 2h, over 2.5GbE ~4 min.

## Known limits

- The vision model can't tell a make from a near miss; the make model
  (>= 0.8 for reels) is what keeps misses out. On 24 Oct 2026 that left 1
  miss in the top 20; the owner's labels removed it.
- AI stars separate good from bad but not 4 vs 5 stars (Spearman ~0.55).
- Make scoring needs >= 8 shots per video to rank; tiny videos are left out.
- Dates come from video names: `DJI_YYYYMMDDHHMMSS...` or phone-style
  `video_YYYYMMDD_HHMMSS` / `VID_YYYYMMDD_HHMMSS`. Cut clips don't keep the
  original's `creation_time`, so a renamed video shows "Unknown date".
- Player names across weeks are suggestions (1080p: faces and most jersey
  numbers unreadable, clothes change weekly): ~40% of goals get one, ~2/3
  right. Grouping within a session is the reliable part (67-86% pure).
- Remotion is free for individuals / teams of up to 3.
