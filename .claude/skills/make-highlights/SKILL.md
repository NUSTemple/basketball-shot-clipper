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
- **FAIL**: fix it with the `fix:` line it prints (details for Windows/Linux,
  incl. CUDA torch for RTX 50-series, in `docs/LOCAL_SETUP.md`). If
  `poetry` itself is missing, the repo isn't installed yet - follow
  `docs/LOCAL_SETUP.md` section 1 (on Windows `installer\setup-deps.ps1`
  does most of it). Installing system tools needs the user's OK; say what
  you'd install and why.
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
  `caffeinate -ims`; on Windows/Linux ask the user to keep the PC awake.
  Re-running resumes from where it stopped.
- Flags: `--no-vlm` (no vision model), `--no-players`, `--seed N` (a fresh
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

## Step 3 (optional): the user reviews, the reels follow

```bash
poetry run shot-clipper-label-ui --clips-dir "<out>/clips"
```

Review = goal/miss + stars; Players = confirm one name per group. Labels go
to `data/dataset/labels.json` and the next `shot-clipper-highlights` run
picks them up automatically: the person's misses are dropped from reels,
their stars outrank the AI's, their scorer tags name the groups and feed the
gallery. The re-run only re-renders (~2 min).

Labels made in the **hosted** app instead: download them
(`gcloud storage cp gs://shot-clipper-data/shared/data/labels.json
data/prelabel/training/shared_labels_<date>.json`) and add
`--labels <that file> --label-prefix <account_slug>/`. Clips can be
uploaded for hosted review with `gcloud storage rsync -r -x '.*\.DS_Store$'
"<out>/clips" "gs://shot-clipper-data/clips/<account_slug>"` (deleted
after 7 days). Both need access to the project's GCP bucket.

## Known limits

- The vision model can't tell a make from a near miss; the make model
  (>= 0.8 for reels) is what keeps misses out. On 24 Oct 2026 that left 1
  miss in the top 20; the owner's labels removed it.
- AI stars separate good from bad but not 4 vs 5 stars (Spearman ~0.55).
- Make scoring needs >= 8 shots per video to rank; tiny videos are left out.
- Player names across weeks are suggestions (1080p: faces and most jersey
  numbers unreadable, clothes change weekly): ~40% of goals get one, ~2/3
  right. Grouping within a session is the reliable part (67-86% pure).
- Remotion is free for individuals / teams of up to 3.
