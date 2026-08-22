"""Cut independent clips around each detected/confirmed shot timestamp.

Reads full original-resolution/framerate video (decision 6/7: independent
files, not a merged highlight reel; 5s before / 2s after each make).

Usage:
    shot-clipper-clip <video_path> <timestamps_json> [--pre 5] [--post 2]
        [--merge-gap 5] [--outdir clips] [--filter-model models/shot_filter.joblib]

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import external, inference, paths


def load_timestamps(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data["makes_sec"]
    return data


# How much of the run-up and follow-through a clip keeps around its shot.
# Named because several callers have to agree on them: features.py and
# net_motion.py both assume the event sits at clip-local DEFAULT_PRE, and
# the review view cuts new clips that have to match the ones cut_all made.
DEFAULT_PRE = 5.0
DEFAULT_POST = 2.0

# Quality settings for a clip a human will only glance at on the way to
# saying goal/no_goal, versus one that ends up in a highlight reel. Preview
# clips are cut from an already-downscaled proxy (see proxy.py), so there is
# nothing to preserve and every reason to be fast.
PREVIEW_ARGS = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]
DELIVERY_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


def cut_clip(video_path: Path, start: float, duration: float, out_path: Path,
             preview: bool = False):
    """Cut [start, start+duration) out of video_path.

    preview=True trades fidelity for speed - see PREVIEW_ARGS. It must only
    be used for clips that are reviewed and discarded, never for ones that
    feed features.py/net_motion.py or get exported: the filter model was
    trained on DELIVERY_ARGS pixels, and scoring ultrafast/crf28 frames
    against it compares the clip to something it never saw.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No -hwaccel here on purpose: measured at 1.03x, i.e. nothing. -ss sits
    # before -i, so ffmpeg seeks to a keyframe and only decodes the ~7s being
    # cut - x264 encoding is the entire cost. (Hardware *encoding* would move
    # the needle, but h264_nvenc changes the written pixels, and these files
    # feed net_motion's features for a filter trained on x264 clips.)
    cmd = [
        external.ffmpeg_exe(), "-y",
        "-ss", f"{max(0.0, start):.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        *(PREVIEW_ARGS if preview else DELIVERY_ARGS),
        "-c:a", "aac",
        str(out_path),
    ]
    external.run(cmd, check=True, capture_output=True)


def cluster_timestamps(timestamps: list[float], merge_gap: float = 5.0) -> list[list[float]]:
    """Group makes less than merge_gap seconds apart, so they become one
    longer clip instead of several near-duplicate ones to review.

    Independent of the clip's pre/post padding on purpose - the two are
    different questions (how close must two makes be to belong in one clip,
    vs. how much lead-in/lead-out that clip gets). Covers two distinct
    cases with one rule: find_makes' cooldown is only 1.5s, so a single
    physical shot can produce two to four re-detections a couple seconds
    apart, and separately, two genuinely distinct makes close enough
    together (e.g. a fast make-then-make sequence) would otherwise land in
    two clips whose padded windows mostly duplicate each other's footage.
    """
    clusters: list[list[float]] = []
    for t in sorted(timestamps):
        if clusters and (t - clusters[-1][-1]) < merge_gap:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return clusters


def cut_all(video_path: Path, timestamps: list[float], outdir: Path,
            pre: float = DEFAULT_PRE, post: float = DEFAULT_POST, progress_cb=None,
            cancel_check=None, merge_overlapping: bool = True,
            merge_gap: float = 5.0, preview: bool = False,
            start_index: int = 1) -> list[tuple[int, float, Path]]:
    """Cut clips into outdir/shot_NNN.mp4. Returns (index, timestamp, out_path)
    tuples in completion order. progress_cb(i, total), if given, is called
    after each clip finishes.

    merge_overlapping (default on) cuts one clip per *cluster* of candidates
    rather than one per candidate - see cluster_timestamps, which merge_gap
    (seconds) is passed straight through to. A cluster is cut as
    [first - pre, last + post], which deliberately keeps the FIRST
    candidate at clip-local `pre` seconds: net_motion's POST_WINDOW and
    contact_sheet's default thumbnail offset both assume the event sits at
    5.0s, so only the tail of the clip grows and those stay valid. The
    timestamp reported for a merged clip is the first candidate's, which is
    also the crossing features.py will find.

    preview cuts fast, low-fidelity files (see cut_clip) - for the review
    pass, where the clip is looked at once and thrown away.

    start_index offsets the shot_NNN numbering, so a later call can append
    to a directory without colliding with clips already in it. Callers
    tracking shots in a manifest should pass its next_index rather than
    counting files, since names are never reused (see shot_manifest).

    cancel_check, if given, is polled after each clip finishes; once it
    returns truthy, any clips not yet started are dropped and this returns
    early with whatever finished so far. Clips already mid-cut are left to
    finish rather than killed - interrupting ffmpeg partway through a write
    risks a corrupt output file, and it's only ~8 clips (max_workers) away
    from done anyway.
    """
    groups = (cluster_timestamps(timestamps, merge_gap) if merge_overlapping
              else [[t] for t in sorted(timestamps)])

    def do_one(item):
        i, group = item
        start = group[0] - pre
        duration = (group[-1] + post) - start
        out_path = outdir / f"shot_{i:03d}.mp4"
        cut_clip(video_path, start, duration, out_path, preview=preview)
        return i, group[0], out_path

    results = []
    # ffmpeg cuts don't touch the GPU and barely touch each other's I/O, so
    # cutting several in parallel is a straightforward, zero-risk speedup
    # over doing them one at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(do_one, item): item for item in enumerate(groups, start=start_index)}
        for future in as_completed(futures):
            i, t, out_path = future.result()
            results.append((i, t, out_path))
            if progress_cb:
                # how many have finished, not which one just did - as_completed
                # yields in arbitrary order, so reporting i made the status read
                # "10/10" while seven clips were still being written.
                progress_cb(len(results), len(groups))
            if cancel_check and cancel_check():
                for f in futures:
                    f.cancel()
                break
    return results


# must match the fps train_filter.py extracted features at, or scoring drifts
# from what the model was trained on
FILTER_FPS = 15.0


SCORES_FILENAME = "_filter_scores.json"


def filter_clips(cut_results: list[tuple[int, float, Path]], hoop_bbox_norm,
                  filter_model_path: Path, filter_meta_path: Path | None = None,
                  model: str | Path | None = None, device: str | None = None,
                  progress_cb=None, cancel_check=None):
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

    cancel_check, if given, is polled between clips; once truthy, scoring
    stops early and returns whatever was scored so far (unscored clips are
    left on disk as-is, neither kept nor dropped).
    """
    if not cut_results:
        return [], []

    from . import features

    meta_path = filter_meta_path or filter_model_path.with_name(
        filter_model_path.stem + "_meta.json")
    clf, meta = features.load_filter_model(filter_model_path, meta_path)
    threshold = meta["threshold"]

    from .device_config import get_device
    device = device or get_device()
    detector = inference.load_detector(model, device=device)

    kept, dropped = [], []
    scores_by_filename = {}
    for idx, (i, t, path) in enumerate(cut_results, start=1):
        score = features.score_clip(path, hoop_bbox_norm, detector, clf,
                                     meta["feature_names"], device=device, fps=FILTER_FPS)
        if score >= threshold:
            kept.append((i, t, path, score))
            scores_by_filename[path.name] = score
        else:
            path.unlink()
            dropped.append((i, t, path, score))
        if progress_cb:
            progress_cb(idx, len(cut_results))
        if cancel_check and cancel_check():
            break

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
    parser.add_argument("--no-merge", action="store_true",
                         help="cut one clip per candidate even when they're close together "
                              "(default merges makes less than --merge-gap seconds apart into "
                              "a single longer clip - see cluster_timestamps)")
    parser.add_argument("--merge-gap", type=float, default=5.0,
                         help="merge two makes into one clip if less than this many seconds "
                              "apart (default: 5.0); independent of --pre/--post padding")
    parser.add_argument("--filter-model", type=Path, default=None,
                         help="optional classifier from shot-clipper-train-filter "
                              "(e.g. models/shot_filter.joblib) to drop false-positive "
                              "clips after cutting; off by default")
    parser.add_argument("--filter-meta", type=Path, default=None,
                         help="default: <filter-model stem>_meta.json next to --filter-model")
    parser.add_argument("--config", type=Path, default=None,
                         help="hoop calibration for --filter-model; default: "
                              "<data dir>/configs/<video_stem>.json")
    parser.add_argument("--detect-model", type=str, default=None,
                         help="YOLO weights for --filter-model's trajectory features - should "
                              f"match whatever shot-clipper-train-filter used (default: "
                              f"{paths.DEFAULT_DETECT_WEIGHTS})")
    args = parser.parse_args()

    outdir = args.outdir or Path("clips") / args.video.stem
    timestamps = load_timestamps(args.timestamps)
    duration = args.pre + args.post

    results = cut_all(args.video, timestamps, outdir, args.pre, args.post,
                       merge_overlapping=not args.no_merge, merge_gap=args.merge_gap)
    for i, t, out_path in sorted(results):
        start = max(0.0, t - args.pre)
        print(f"[{i}/{len(timestamps)}] make@{t:.2f}s -> {out_path} "
              f"({start:.2f}s .. +{duration:.2f}s)")
    print(f"done: {len(timestamps)} clips in {outdir}")

    if args.filter_model:
        from .detect_shots import load_config

        config_path = args.config or paths.find_config(args.video)
        hoop_bbox_norm = load_config(config_path)
        kept, dropped = filter_clips(results, hoop_bbox_norm, args.filter_model,
                                      filter_meta_path=args.filter_meta, model=args.detect_model)
        for i, t, path, score in sorted(dropped):
            print(f"  dropped shot_{i:03d} (score={score:.3f}): {path}")
        print(f"filter: kept {len(kept)}, dropped {len(dropped)} of {len(timestamps)}")


if __name__ == "__main__":
    main()
