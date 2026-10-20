---
name: make-highlights
description: Turn a folder of raw basketball session videos into three finished highlight reels end to end with no human review - auto hoop calibration, shot detection, AI goal scoring, AI star rating with a local vision model, and Remotion rendering. Use when the user gives a video folder and wants highlight videos (not clips to label).
---

# Make highlight reels from a session folder, unattended

Input: a folder of raw videos, e.g.
`/Users/pengtan/Videos/20261017 Basketball Video/DJI_001`. Output: three
different reels of the session's best goals. The user has asked for minimum
intervention - don't stop to ask questions the defaults answer.

Everything is one resumable command (`src/shot_clipper/highlights.py`):

```bash
poetry run shot-clipper-highlights "<folder>"
```

## What it does

1. **Auto-calibrate** each video from the rim RF-DETR finds (writes
   `data/configs/<stem>.json` unless one exists). A video with no rim found
   is skipped and reported - that's the only case needing a human
   (`shot-clipper-calibrate`, then re-run).
2. **Detect + cut** shots (reuses `data/ground_truth/<stem>_detected.json`
   if present). Clips go to `<folder>/../auto_highlights/clips/<stem>/` -
   deliberately NOT the label UI's `clips/`, since cutting clears the
   destination.
3. **AI make score** per clip with the global model (`global.joblib`,
   from `$SHOT_CLIPPER_PRELABEL_MODEL_DIR`, `models/prelabel/` or
   `data/prelabel/models/`; fetch it from
   `gs://shot-clipper-data/models/prelabel/global.joblib` if missing). Clips
   scoring >= 0.6 are the goals; it also picks which shot in a merged clip
   went in, so reels cut on the real make.
4. **AI stars**: the local Qwen3-VL (MLX venv `~/.venvs/mlx-vlm`, ~13 s a
   clip, run under `caffeinate`) rates each goal; rank within the session ->
   1-5 stars (top 15% = 5). Written to `auto_highlights/ai_stars.csv`. Not
   uploaded to the label UI.
5. **Render** three reels with Remotion (`remotion-reel/`, props written per
   reel): `top_plays` (best ~20, in game order, mixed effects),
   `countdown` (top 10, #1 last with a zoom replay), `fast_cut` (shuffled
   pick from the top 20% of goals, short trims, no slow motion). No names, star
   captions or music - the user's choice.
6. **Add to Photos**: the three reels are imported into the Mac Photos album
   "Basketball Highlights" (created if missing; iCloud syncs them to the
   phone). `--photos-album ""` skips it.

## Running it

- Long: detection is ~hours for a fresh 20+ GB session on the M3 Max; the
  vision model ~30 min for ~150 goals. Run it in the background and wrap in
  `caffeinate -ims` (a sleeping Mac stalls the GPU). Re-running resumes: a
  video with `_highlight_shots.json` is done, rated clips in `vlm.jsonl` are
  skipped.
- `--no-vlm` ranks by make score only (fast, rougher stars).
- `--seed N` reshuffles effects/transitions/the fast_cut pick for a fresh
  set of three; `--variants top_plays,countdown` renders a subset.
- First run in a fresh checkout: `npm ci` in `remotion-reel/` happens
  automatically; the Remotion agent skills are restored with
  `npx skills experimental_install` there (only needed to edit the reel).

## When it finishes

Check before reporting:
- `auto_highlights/summary.json` - videos skipped, goals, star counts, reels.
- Look at each `reels/<name>.jpg` contact sheet (Read tool) - no black or
  frozen frames, zooms aimed at the rim.

Report: where the three reels are (and that they're in the Photos album), their lengths, goals found per video,
any skipped video and why, and the AI star breakdown. Mention the
`ai_stars.csv` ranking in case they want to swap clips.

## Known limits

- AI stars separate good from bad well (validated: 0 one-star clips in the
  VLM's top 23) but not 4 vs 5 stars (Spearman 0.55 vs the owner's ratings).
- Make scoring needs >= 8 shots per video to rank; tiny videos are left out.
- Remotion is free for individuals / teams of up to 3.
