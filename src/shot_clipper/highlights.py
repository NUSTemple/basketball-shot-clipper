"""Folder of raw session videos -> three highlight reels, with no human in
the loop.

    shot-clipper-highlights "/Users/me/Videos/20261017 Basketball Video/DJI_001"

Per video: find the rim with RF-DETR and write the hoop calibration from it
(no hand-drawn box), detect and cut shot clips, and score every clip with the
global AI make model (prelabel.py), which also says which shot in a merged
clip is the make. Clips scoring >= prelabel.UI_MAKE_FROM are kept as goals.
A local vision-language model (highlight_vlm.py) then rates how
highlight-worthy each goal is; goals are ranked and given 1-5 AI stars by
rank within the session, and three different reels are rendered with the
Remotion project in remotion-reel/:

  top_plays  - the best ~20 goals in the order they happened, mixed effects
  countdown  - the top 10 from #10 to #1, the best one last with a zoom replay
  fast_cut   - a shuffled pick from the top 20% of goals, short trims, quick cuts,
               no slow motion

Everything lands in <out> (default: "<folder>/../auto_highlights"):
clips/<video>/shot_NNN.mp4, ai_stars.csv, vlm.jsonl, reels/*.mp4 and
summary.json. Every stage is resumable: re-running skips videos already
scored and clips already rated, so an interrupted run picks up where it was.
"""
import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import calibrate_hoop, clip_shots, detect_shots, inference, prelabel
from .device_config import device_summary, get_device

VIDEO_EXTS = {".mp4", ".mov"}
REPO = Path(__file__).resolve().parents[2]
REMOTION_DIR = REPO / "remotion-reel"
VLM_PYTHON = Path(os.environ.get("SHOT_CLIPPER_VLM_PYTHON", Path.home() / ".venvs/mlx-vlm/bin/python"))
SHOTS_FILENAME = "_highlight_shots.json"
# AI stars by rank within the session: the top 15% get 5, the next 30% 4, ...
STAR_SHARES = [(5, 0.15), (4, 0.30), (3, 0.30), (2, 0.15), (1, 0.10)]
SLOW_EFFECTS = {"speed_ramp", "replay", "zoom_replay"}
# Reels only use goals the make model is this sure of. The VLM can't tell a
# make from a near miss (it calls most layups "swishes"), so on 24 Oct 2026
# 5 of the 20 top-ranked "goals" were misses; >= 0.8 cut that to 1 while
# keeping 78 of the owner's 80 real goals (17 Oct: no change, 0 misses).
REEL_MIN_MAKE = 0.8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# per video: calibrate, detect, cut, score
# --------------------------------------------------------------------------

def auto_calibrate(video: Path, detector) -> list[float] | None:
    """Detected rim box (pixels), writing data/configs/<stem>.json from it
    unless a calibration already exists. None when no rim was found."""
    import cv2

    rim = prelabel.detect_rim(video, detector)
    if rim is None:
        return None
    config = calibrate_hoop.config_path_for(video)
    if not config.exists():
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        x1, y1, x2, y2 = prelabel.rim_hoop_box(rim, w, h)
        box = (max(0, round(x1 * w)), max(0, round(y1 * h)), min(w, round(x2 * w)), min(h, round(y2 * h)))
        calibrate_hoop.save_config(video, n // 2, w, h, box)
    return rim


def process_video(video: Path, clips_root: Path, detector, device: str, bundle: dict) -> dict:
    """Calibrate, detect, cut and score one video. Writes
    clips_root/<stem>/SHOTS_FILENAME, whose presence marks the video done."""
    outdir = clips_root / video.stem
    done = outdir / SHOTS_FILENAME
    if done.exists():
        log(f"{video.name}: already processed - reusing")
        return json.loads(done.read_text())

    rim = auto_calibrate(video, detector)
    if rim is None:
        log(f"{video.name}: SKIPPED - no rim found (calibrate it by hand, then re-run)")
        return {"video": video.name, "skipped": "no rim found"}
    config = calibrate_hoop.config_path_for(video)
    truth = Path("data/ground_truth") / f"{video.stem}_detected.json"
    if truth.exists():
        log(f"{video.name}: reusing detection {truth}")
        makes = clip_shots.load_timestamps(truth)
    else:
        log(f"{video.name}: detecting shots on {device_summary(device)} (the slow part)...")
        last = [0.0]

        def progress(t):
            if t - last[0] >= 300:
                last[0] = t
                log(f"  {video.name}: {t / 60:.0f} min of video scanned")
        makes = detect_shots.run_detection(video, config, truth, device=device, progress_cb=progress)
    log(f"{video.name}: {len(makes)} shot candidates - cutting clips")
    cut = clip_shots.cut_all(video, makes, outdir)

    log(f"{video.name}: AI make-scoring {len(cut)} clips")
    groups = clip_shots.cluster_timestamps(makes)
    rows, times = {}, {}
    for i, _, path in sorted(cut):
        name = Path(path).name
        times[name] = prelabel.event_times(groups[i - 1])
        rows[name] = prelabel.clip_shot_features(Path(path), rim, times[name], detector, device=device)
    best = prelabel.best_shots(bundle, rows)
    if not best:
        log(f"{video.name}: only {sum(map(len, rows.values()))} shots - too few to rank, clips left unscored")
    (outdir / prelabel.PRELABEL_FILENAME).write_text(
        json.dumps({c: s for c, (s, _) in best.items()}, indent=1, sort_keys=True))
    cfg = json.loads(config.read_text())
    result = {"video": video.name, "rim_px": rim, "hoop_norm": cfg["hoop_bbox_norm"],
              "frame": [cfg["frame_width"], cfg["frame_height"]], "n_clips": len(cut),
              "clips": {c: {"score": s, "make": round(times[c][i], 2)} for c, (s, i) in sorted(best.items())}}
    done.write_text(json.dumps(result, indent=1))
    n_goals = sum(v["score"] >= prelabel.UI_MAKE_FROM for v in result["clips"].values())
    log(f"{video.name}: {n_goals} likely goals of {len(cut)} clips")
    return result


# --------------------------------------------------------------------------
# highlight rating
# --------------------------------------------------------------------------

def rate_goals(goals: list[dict], out: Path, use_vlm: bool) -> dict[str, dict]:
    """VLM attributes per goal clip key ({} for clips it couldn't rate)."""
    if not use_vlm or not goals:
        return {}
    if not VLM_PYTHON.exists():
        log(f"no MLX-VLM venv at {VLM_PYTHON} - ranking by make score only")
        return {}
    jobs = out / "vlm_jobs.json"
    jobs.write_text(json.dumps([{"clip": g["key"], "path": str(g["path"]), "hoop": g["hoop"], "make": g["make"]}
                                for g in goals]))
    results = out / "vlm.jsonl"
    log(f"rating {len(goals)} goals with the local vision model (~13s each)...")
    # -I: don't let anything in the cwd shadow a stdlib module in that venv
    cmd = [str(VLM_PYTHON), "-I", str(Path(__file__).with_name("highlight_vlm.py")), str(jobs), str(results)]
    if shutil.which("caffeinate"):  # a sleeping Mac stalls the GPU for minutes at a time
        cmd = ["caffeinate", "-ims", *cmd]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log(f"vision model exited with {proc.returncode} - ranking what it finished, the rest by make score")
    rated = {}
    if results.exists():
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if row.get("vlm"):
                rated[row["clip"]] = row["vlm"]
    return rated


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def ai_score(goal: dict, vlm: dict | None) -> float:
    """Highlight score: the VLM's 1-10 rating nudged by its difficulty and
    athleticism, minus a warm-up penalty; make confidence breaks ties. Without
    a VLM rating it's the make confidence alone, on the same rough scale."""
    if not vlm:
        return 4.0 + 2.0 * goal["score"]
    s = _num(vlm.get("highlight_score"), 4.0)
    s += 0.25 * (_num(vlm.get("difficulty"), 2) + _num(vlm.get("athleticism"), 2) - 4)
    if vlm.get("game_context") == "warm_up":
        s -= 1.0
    return round(s + 0.5 * goal["score"], 3)


def assign_stars(ranked: list[dict]) -> None:
    """ranked is best-first; sets each goal's "stars" by rank share."""
    n, i = len(ranked), 0
    for stars, share in STAR_SHARES:
        take = max(1, round(share * n)) if stars == 5 else round(share * n)
        for g in ranked[i:i + take]:
            g["stars"] = stars
        i += take
    for g in ranked[i:]:
        g["stars"] = STAR_SHARES[-1][0]


# --------------------------------------------------------------------------
# reels
# --------------------------------------------------------------------------

def pick_effects(n: int, weights: dict[str, int], rnd: random.Random) -> list[str]:
    """Random effect per clip; never two slow-motion effects in a row."""
    names, w = list(weights), list(weights.values())
    out = []
    for _ in range(n):
        while True:
            e = rnd.choices(names, weights=w)[0]
            if not (e in SLOW_EFFECTS and out and out[-1] in SLOW_EFFECTS):
                break
        out.append(e)
    return out


VARIANTS = {
    "top_plays": dict(
        count=20, pre=2.5, post=1.5, transition=14, leak_every=5,
        presentations=["fade", "slideRight", "slideLeft", "wipe", "pushCut", "crossZoom", "zoomBlur", "filmBurn"],
        effects={"plain": 40, "freeze": 15, "punch_zoom": 15, "replay": 10, "zoom_replay": 10, "speed_ramp": 10}),
    "countdown": dict(
        count=10, pre=2.5, post=1.5, transition=16, leak_every=4,
        presentations=["slideRight", "wipe", "pushCut", "zoomBlur"],
        effects={"plain": 50, "freeze": 20, "punch_zoom": 30}),
    "fast_cut": dict(
        count=16, pre=1.7, post=0.9, transition=8, leak_every=0,
        presentations=["pushCut", "zoomBlur", "slideRight", "slideLeft", "fade"],
        effects={"plain": 60, "punch_zoom": 30, "freeze": 10}),
}


def session_title(folder: Path) -> tuple[str, str]:
    """("SATURDAY RUN", "17 Oct 2026") from a DJI_YYYYMMDD... name in or
    above the folder, else the folder's name."""
    for name in [*(p.name for p in sorted(folder.iterdir())), folder.name, folder.parent.name]:
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
        if m:
            try:
                d = dt.date(*map(int, m.groups()))
            except ValueError:
                continue
            return f"{d.strftime('%A').upper()} RUN", f"{d.day} {d.strftime('%b %Y')}"
    return "HIGHLIGHTS", folder.parent.name if folder.name.startswith("DJI") else folder.name


def group_players(goals: list[dict], clips_root: Path, detector, device: str, labels: dict) -> dict:
    """Group the goals by shooter (person_id) and count goals per group,
    keyed by the suggested name or "Player N". A failure here never costs
    the reels."""
    from .person_id import PLAYERS_FILENAME, scan

    try:
        scan.scan(clips_root, labels, detector=detector, device=device, only={g["key"] for g in goals}, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"player grouping failed ({type(e).__name__}: {e}) - reels go ahead without it")
        return {}
    counts, unnamed = {}, {}
    for folder in sorted({clips_root / g["video"] for g in goals}):
        for info in scan.read_json(folder / PLAYERS_FILENAME).get("clips", {}).values():
            if not info.get("group"):
                continue
            if info.get("suggested"):
                key = info["suggested"]
            else:
                key = unnamed.setdefault(info["group"], f"Player {len(unnamed) + 1}")
            counts[key] = counts.get(key, 0) + 1
    log("goals by player: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def human_label(labels: dict, key: str) -> dict:
    """A person's label for the clip, ignoring the AI's bulk-marked ones."""
    entry = labels.get(key) or {}
    return entry if entry.get("label") and entry.get("source") != "ai" else {}


def reel_candidates(ranked: list[dict], labels: dict, minimum: int = 10) -> list[dict]:
    """Goals to build reels from, best first. A person's labels win: their
    misses are dropped, their goals count whatever the make score, and their
    stars outrank the AI's. Unlabeled goals need make score >= REEL_MIN_MAKE."""
    picked = []
    for g in ranked:
        human = human_label(labels, g["key"])
        if human.get("label") == "no_goal":
            continue
        if human.get("label") == "goal" or g["score"] >= REEL_MIN_MAKE:
            picked.append((human.get("stars") or g["stars"], g["ai_score"], g))
    picked.sort(key=lambda t: (-t[0], -t[1]))
    out = [g for _, _, g in picked]
    if len(out) < minimum:  # too strict for a small session: top up in AI order
        out += [g for g in ranked if g not in out and human_label(labels, g["key"]).get("label") != "no_goal"]
    return out


def build_variant(name: str, ranked: list[dict], videos: dict, title: tuple[str, str], seed: int) -> dict | None:
    v = VARIANTS[name]
    rnd = random.Random(seed)
    if name == "top_plays":
        picks = sorted(ranked[:v["count"]], key=lambda g: (g["video"], g["clip"]))
        effects = pick_effects(len(picks), v["effects"], rnd)
        lines = [title[0], title[1]]
    elif name == "countdown":
        picks = list(reversed(ranked[:v["count"]]))
        effects = pick_effects(len(picks), v["effects"], rnd)
        if effects:
            effects[-1] = "zoom_replay"  # the #1 play gets the slow-mo replay
        lines = [f"TOP {len(picks)}", f"{title[0]} · {title[1]}"]
    else:
        # the top 20%: on 17 Oct 2026 the owner had rated none of the AI's top
        # 30 (of 149) 1-star or a miss, but 4 of the next 10
        pool = ranked[:max(v["count"] + 6, round(0.2 * len(ranked)))]
        picks = rnd.sample(pool, min(v["count"], len(pool)))
        effects = pick_effects(len(picks), v["effects"], rnd)
        lines = [title[0], title[1]]
    if not picks:
        return None
    clips = []
    for g, effect in zip(picks, effects):
        x1, y1, x2, y2 = videos[g["video"]]["rim_px"]
        fw, fh = videos[g["video"]]["frame"]
        clips.append({"file": f"reels/{g['video']}/{g['clip']}", "make": g["make"], "pre": v["pre"],
                      "post": v["post"], "effect": effect,
                      "rimX": round((x1 + x2) / 2 / fw * 100, 1), "rimY": round((y1 + y2) / 2 / fh * 100, 1)})
    return {"titleLines": lines, "clips": clips, "transitionFrames": v["transition"],
            "leakEvery": v["leak_every"], "presentations": v["presentations"], "seed": seed}


def stage_clips(goals: list[dict], public: Path) -> None:
    """Hard-link (copy across filesystems) the goal clips into Remotion's
    public/reels, replacing whatever an earlier session left there."""
    shutil.rmtree(public, ignore_errors=True)
    for g in goals:
        dst = public / g["video"] / g["clip"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(g["path"], dst)
        except OSError:
            shutil.copy2(g["path"], dst)


def render(props: dict, out_mp4: Path) -> None:
    if not (REMOTION_DIR / "node_modules").exists():
        log("installing the Remotion project's npm packages (first run only)...")
        subprocess.run(["npm", "ci"], cwd=REMOTION_DIR, check=True)
    props_path = out_mp4.with_suffix(".props.json")
    props_path.write_text(json.dumps(props, indent=1))
    subprocess.run(["npx", "remotion", "render", "HighlightReel", str(out_mp4), f"--props={props_path}",
                    "--log=error"], cwd=REMOTION_DIR, check=True)


def contact_sheet(mp4: Path) -> None:
    secs = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(mp4)], capture_output=True, text=True, check=True).stdout)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    f"fps={16 / secs:.4f},scale=480:-1,tile=4x4", "-frames:v", "1",
                    str(mp4.with_suffix(".jpg"))], check=True)


def add_to_photos(files: list[Path], album: str) -> None:
    """Import files into an album in the Mac's Photos library (created if
    missing) - with iCloud Photos they then reach the phone too."""
    paths = ", ".join(f'POSIX file "{f}"' for f in files)
    script = f'''tell application "Photos"
    if not (exists album "{album}") then make new album named "{album}"
    import {{{paths}}} into album "{album}" skip check duplicates true
end tell'''
    subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=600)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="folder of raw session videos")
    ap.add_argument("--out", type=Path, default=None, help='default: "<folder>/../auto_highlights"')
    ap.add_argument("--model", type=Path, default=None,
                    help="AI make model (default: global.joblib in $SHOT_CLIPPER_PRELABEL_MODEL_DIR, "
                         "models/prelabel or data/prelabel/models)")
    ap.add_argument("--no-vlm", action="store_true", help="rank by make score only (much faster, rougher)")
    ap.add_argument("--no-players", action="store_true", help="skip grouping the goals by who scored")
    ap.add_argument("--seed", type=int, default=None, help="randomness for effects/picks (default: from the date)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labels.json with a person's goal/miss/stars/scorer for these clips (e.g. downloaded from "
                         "the hosted app) - they override the AI when picking reel clips")
    ap.add_argument("--label-prefix", default="", help="strip this from --labels keys, e.g. '<account>/'")
    ap.add_argument("--photos-album", default="Basketball Highlights",
                    help="Photos album to import the reels into ('' to skip)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list from {','.join(VARIANTS)}")
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    os.chdir(REPO)  # data/configs and data/ground_truth are repo-relative
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
    if not videos:
        sys.exit(f"no videos in {folder}")
    out = (args.out or folder.parent / "auto_highlights").expanduser().resolve()
    clips_root = out / "clips"
    (out / "reels").mkdir(parents=True, exist_ok=True)
    candidates = [args.model] if args.model else [prelabel.MODEL_DIR / prelabel.GLOBAL_MODEL_NAME,
                                                   Path("data/prelabel/models") / prelabel.GLOBAL_MODEL_NAME]
    bundle = next((b for b in map(prelabel.load_model, candidates) if b is not None), None)
    if bundle is None:
        sys.exit(f"no AI make model at {' or '.join(map(str, candidates))} - fetch "
                 "gs://shot-clipper-data/models/prelabel/global.joblib")

    started = time.time()
    device = get_device()
    log(f"{len(videos)} videos -> {out} ({device_summary(device)})")
    detector = inference.load_detector(device=device)
    results = {}
    for video in videos:
        results[video.stem] = process_video(video, clips_root, detector, device, bundle)

    goals = []
    for stem, r in results.items():
        for clip, info in r.get("clips", {}).items():
            if info["score"] >= prelabel.UI_MAKE_FROM:
                goals.append({"key": f"{stem}/{clip}", "video": stem, "clip": clip, "path": clips_root / stem / clip,
                              "hoop": r["hoop_norm"], **info})
    n_clips = sum(r.get("n_clips", 0) for r in results.values())
    log(f"{len(goals)} likely goals of {n_clips} clips")

    rated = rate_goals(goals, out, use_vlm=not args.no_vlm)
    for g in goals:
        g["vlm"] = rated.get(g["key"])
        g["ai_score"] = ai_score(g, g["vlm"])
    ranked = sorted(goals, key=lambda g: -g["ai_score"])
    assign_stars(ranked)
    with (out / "ai_stars.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "clip", "ai_stars", "ai_score", "make_score", "make_sec", "vlm_highlight", "description"])
        for n, g in enumerate(ranked, start=1):
            vlm = g["vlm"] or {}
            w.writerow([n, g["key"], g["stars"], g["ai_score"], g["score"], g["make"],
                        vlm.get("highlight_score", ""), vlm.get("description", "")])

    from .dataset_labels import load_labels
    labels = load_labels()
    if args.labels:
        extra = json.loads(args.labels.read_text())
        labels.update({k[len(args.label_prefix):]: v for k, v in extra.items() if k.startswith(args.label_prefix)})
    players = {} if args.no_players else group_players(ranked, clips_root, detector, device, labels)
    candidates = reel_candidates(ranked, labels)
    n_human = sum(1 for g in ranked if human_label(labels, g["key"]))
    log(f"reels pick from {len(candidates)} goals" + (f" ({n_human} labeled by you)" if n_human else ""))

    title = session_title(folder)
    seed = args.seed if args.seed is not None else int(re.sub(r"\D", "", title[1]) or 0) or 7
    stage_clips(ranked, REMOTION_DIR / "public" / "reels")
    previous = out / "summary.json"  # a --variants re-run keeps the other reels listed
    reels = json.loads(previous.read_text()).get("reels", {}) if previous.exists() else {}
    rendered = []
    for i, name in enumerate(n.strip() for n in args.variants.split(",")):
        props = build_variant(name, candidates, results, title, seed + i)
        if props is None:
            log(f"{name}: no goals to use - skipped")
            continue
        mp4 = out / "reels" / f"{name}.mp4"
        log(f"rendering {name} ({len(props['clips'])} clips)...")
        render(props, mp4)
        contact_sheet(mp4)
        reels[name] = {"file": str(mp4), "clips": [c["file"].split("/", 1)[1] for c in props["clips"]]}
        rendered.append(mp4)

    if args.photos_album and rendered and sys.platform == "darwin":
        try:
            add_to_photos(rendered, args.photos_album)
            log(f"added {len(rendered)} reels to the Photos album \"{args.photos_album}\"")
        except (subprocess.SubprocessError, OSError) as e:
            log(f"could not add the reels to Photos ({e}) - they're still in {out / 'reels'}")

    summary = {
        "folder": str(folder), "videos": {k: {kk: v.get(kk) for kk in ("n_clips", "skipped")} for k, v in results.items()},
        "n_clips": n_clips, "n_goals": len(goals), "n_vlm_rated": len(rated),
        "stars": {s: sum(g["stars"] == s for g in ranked) for s in (5, 4, 3, 2, 1)},
        "players": players, "reels": reels, "minutes": round((time.time() - started) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"done in {summary['minutes']} min: {len(rendered)} reels in {out / 'reels'}")


if __name__ == "__main__":
    main()
