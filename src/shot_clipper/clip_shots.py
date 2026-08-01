"""Cut independent clips around each detected/confirmed shot timestamp.

Reads full original-resolution/framerate video (decision 6/7: independent
files, not a merged highlight reel; 5s before / 2s after each make).

Usage:
    shot-clipper-clip <video_path> <timestamps_json> [--pre 5] [--post 2]
        [--outdir clips] [--filter-model models/shot_filter.joblib]

Run from the repo root (or pass --outdir) - default output is ./clips/<video_stem>/.

<timestamps_json> is either the output of detect_shots.py
({"makes_sec": [...]}) or a plain JSON list of seconds.

--filter-model (optional) applies a classifier trained by
shot-clipper-train-filter to drop false-positive clips after cutting: it
needs both ball-trajectory features and net-motion features (pixel motion
in the net region), and the latter needs actual frames, so filtering
happens here on the cut clip files rather than in detect_shots.py - see
features.score_clip().
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_timestamps(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data["makes_sec"]
    return data


def cut_clip(video_path: Path, start: float, duration: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start):.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def cut_all(video_path: Path, timestamps: list[float], outdir: Path,
            pre: float = 5.0, post: float = 2.0, progress_cb=None) -> list[tuple[int, float, Path]]:
    """Cut one clip per timestamp into outdir/shot_NNN.mp4. Returns
    (index, timestamp, out_path) tuples in completion order. progress_cb(i, total),
    if given, is called after each clip finishes.
    """
    duration = pre + post

    def do_one(item):
        i, t = item
        start = t - pre
        out_path = outdir / f"shot_{i:03d}.mp4"
        cut_clip(video_path, start, duration, out_path)
        return i, t, out_path

    results = []
    # ffmpeg cuts don't touch the GPU and barely touch each other's I/O, so
    # cutting several in parallel is a straightforward, zero-risk speedup
    # over doing them one at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, t, out_path in pool.map(do_one, enumerate(timestamps, start=1)):
            results.append((i, t, out_path))
            if progress_cb:
                progress_cb(i, len(timestamps))
    return results


# must match the fps train_filter.py extracted features at, or scoring drifts
# from what the model was trained on
FILTER_FPS = 15.0


SCORES_FILENAME = "_filter_scores.json"


def filter_clips(cut_results: list[tuple[int, float, Path]], hoop_bbox_norm,
                  filter_model_path: Path, filter_meta_path: Path | None = None,
                  model: str = "models/yolov8l.pt", device: str | None = None,
                  progress_cb=None):
    """Score each cut clip (trajectory + net-motion features, see
    features.score_clip) and delete the ones below the trained filter's
    threshold. Returns (kept, dropped), each a list of
    (index, timestamp, path, score) tuples; dropped clips are already
    deleted from disk by the time this returns.

    Scores for kept clips are also written to <outdir>/_filter_scores.json
    (filename -> score) so the label UI can show them - the geometric
    detector is recall-first and over-generates (~34% of raw candidates are
    real makes), so the score is a useful triage signal even though the
    human still makes the final goal/no_goal call.
    """
    if not cut_results:
        return [], []

    from . import features

    meta_path = filter_meta_path or filter_model_path.with_name(
        filter_model_path.stem + "_meta.json")
    clf, meta = features.load_filter_model(filter_model_path, meta_path)
    threshold = meta["threshold"]

    import torch
    from ultralytics import YOLO
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    yolo_model = YOLO(model)

    kept, dropped = [], []
    scores_by_filename = {}
    for idx, (i, t, path) in enumerate(cut_results, start=1):
        score = features.score_clip(path, hoop_bbox_norm, yolo_model, clf,
                                     meta["feature_names"], device=device, fps=FILTER_FPS)
        if score >= threshold:
            kept.append((i, t, path, score))
            scores_by_filename[path.name] = score
        else:
            path.unlink()
            dropped.append((i, t, path, score))
        if progress_cb:
            progress_cb(idx, len(cut_results))

    outdir = cut_results[0][2].parent
    (outdir / SCORES_FILENAME).write_text(json.dumps(scores_by_filename, indent=2))
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("timestamps", type=Path)
    parser.add_argument("--pre", type=float, default=5.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--filter-model", type=Path, default=None,
                         help="optional classifier from shot-clipper-train-filter "
                              "(e.g. models/shot_filter.joblib) to drop false-positive "
                              "clips after cutting; off by default")
    parser.add_argument("--filter-meta", type=Path, default=None,
                         help="default: <filter-model stem>_meta.json next to --filter-model")
    parser.add_argument("--config", type=Path, default=None,
                         help="hoop calibration for --filter-model; default: "
                              "data/configs/<video_stem>.json")
    parser.add_argument("--detect-model", type=str, default="models/yolov8l.pt",
                         help="YOLO weights for --filter-model's trajectory features - should "
                              "match whatever shot-clipper-train-filter used (yolov8l by default)")
    args = parser.parse_args()

    outdir = args.outdir or Path("clips") / args.video.stem
    timestamps = load_timestamps(args.timestamps)
    duration = args.pre + args.post

    results = cut_all(args.video, timestamps, outdir, args.pre, args.post)
    for i, t, out_path in sorted(results):
        start = max(0.0, t - args.pre)
        print(f"[{i}/{len(timestamps)}] make@{t:.2f}s -> {out_path} "
              f"({start:.2f}s .. +{duration:.2f}s)")
    print(f"done: {len(timestamps)} clips in {outdir}")

    if args.filter_model:
        from .detect_shots import load_config

        config_path = args.config or Path("data/configs") / f"{args.video.stem}.json"
        hoop_bbox_norm = load_config(config_path)
        kept, dropped = filter_clips(results, hoop_bbox_norm, args.filter_model,
                                      filter_meta_path=args.filter_meta, model=args.detect_model)
        for i, t, path, score in sorted(dropped):
            print(f"  dropped shot_{i:03d} (score={score:.3f}): {path}")
        print(f"filter: kept {len(kept)}, dropped {len(dropped)} of {len(timestamps)}")


if __name__ == "__main__":
    main()
