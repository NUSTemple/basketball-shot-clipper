"""Train a small classifier on data/dataset/labels.json to filter false
positives out of find_makes()'s candidates.

find_makes() (see detect_shots.py) is a hand-tuned geometric yes/no rule,
validated against a single video (docs/PLAN.md decision 9): 70% precision
there, but only 34% across the full 307-clip labeled dataset covering 9
videos. This trains a classifier on two complementary feature sets extracted
from each labeled clip - continuous trajectory stats about the above/through
-hoop event (features.py) and net-motion stats (net_motion.py, pixel motion
in the net region - motion alone is ~useless, but combined with trajectory
features it nearly doubles how many false positives can be dropped at the
same recall target; see README) - using the real labels, instead of
hand-picked constants.

Usage:
    shot-clipper-train-filter --clips-dir /path/to/clips

Extracts (and caches to data/dataset/features.csv) features for every
labeled clip, trains with leave-one-video-out cross-validation, picks a
confidence threshold that keeps recall >= --min-recall (default 0.98 -
PLAN.md decision 5 treats missed makes as much costlier than false
positives, so the threshold search deliberately protects recall over
precision), and saves the final model to models/shot_filter.joblib.
"""
import argparse
import csv
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import clip_shots, inference, paths, shot_manifest
from .dataset_labels import load_labels
from .device_config import get_device, device_summary
from .features import FEATURE_NAMES, extract_features_for_clip
from .net_motion import MOTION_FEATURE_NAMES, extract_motion_features_for_clip

ALL_FEATURE_NAMES = FEATURE_NAMES + MOTION_FEATURE_NAMES


def features_cache_path() -> Path:
    return paths.dataset_dir() / "features.csv"


def model_out_path() -> Path:
    return paths.models_dir() / "shot_filter.joblib"


def meta_out_path() -> Path:
    return paths.models_dir() / "shot_filter_meta.json"


def hoop_bbox_for_video(video: str, clips_dir: Path):
    """The calibrated hoop box for a clips subfolder name.

    Calibrations live next to their source video now (see paths.py), and
    all this has is the video's stem - so the clips folder's own manifest,
    which records where the video came from, is what turns one into the
    other. Falls back to the old app-data location for anything cut before
    manifests existed.
    """
    manifest = shot_manifest.load(clips_dir / video)
    source = (manifest or {}).get("source_video")
    config_path = (paths.find_config(Path(source)) if source
                   else paths.configs_dir() / f"{video}.json")
    return json.loads(config_path.read_text())["hoop_bbox_norm"]


def build_feature_rows(labels: dict, clips_dir: Path, detector, device: str,
                        fps: float, exclude_videos=()):
    """Yields one dict per labeled clip: video, shot, clip, label, **features.

    exclude_videos names clips-folder subdirectories to leave out entirely -
    for footage that isn't the thing being detected. A video of practice
    shooting produces candidates at a completely different rate to a game
    (69 for 12 makes, against roughly 45 for 20), and training on both
    teaches the threshold to sit between two distributions instead of on
    one.
    """
    configs_cache = {}
    manifest_cache = {}
    excluded = set(exclude_videos or ())
    n_excluded = 0
    for clip_rel, entry in sorted(labels.items()):
        video, shot = clip_rel.split("/")
        if video in excluded:
            n_excluded += 1
            continue
        clip_path = clips_dir / clip_rel
        if not clip_path.is_file():
            print(f"skip (missing on disk): {clip_rel}")
            continue

        # cached per video, not per clip: this loop runs over every label in
        # the dataset (1000+), and the manifest is one file per video
        if video not in manifest_cache:
            manifest_cache[video] = shot_manifest.load(clips_dir / video)
        manifest = manifest_cache[video]

        if video not in configs_cache:
            configs_cache[video] = hoop_bbox_for_video(video, clips_dir)
        hoop_bbox_norm = configs_cache[video]

        with feature_source(clip_path, manifest, shot) as source_clip:
            if source_clip is None:
                print(f"skip (no delivery-quality source): {clip_rel}")
                continue
            traj_features = extract_features_for_clip(
                source_clip, hoop_bbox_norm, detector, device=device, fps=fps)
            motion_features = extract_motion_features_for_clip(source_clip, hoop_bbox_norm)

        yield {
            "video": video, "shot": shot, "clip": clip_rel,
            "label": entry["label"], **traj_features, **motion_features,
        }

    if n_excluded:
        print(f"excluded {n_excluded} clip(s) from {len(excluded)} video(s)")


@contextmanager
def feature_source(clip_path: Path, manifest: dict | None, clip_name: str):
    """A clip whose pixels the feature extractors can trust.

    Review clips are 720p ultrafast re-encodes of a proxy (see
    clip_shots.cut_clip). net_motion reads raw pixel differences, so
    extracting from those would compare a clip against a model trained on
    something else. Where the manifest knows the original and the
    timestamp, the window is re-cut from it at delivery quality and thrown
    away afterwards - which is also what makes it possible to train on a
    shot whose only clip on disk is a preview.
    """
    if not manifest or manifest.get("preview_source") != "proxy":
        yield clip_path
        return

    timestamp = shot_manifest.timestamp_for(manifest, clip_name)
    source = manifest.get("source_video")
    if timestamp is None or not source or not Path(source).is_file():
        yield None
        return

    start = max(0.0, timestamp - clip_shots.DEFAULT_PRE)
    duration = (timestamp + clip_shots.DEFAULT_POST) - start
    with tempfile.TemporaryDirectory() as tmp:
        window = Path(tmp) / "window.mp4"
        clip_shots.cut_clip(Path(source), start, duration, window, preview=False)
        yield window


def load_or_build_features(labels: dict, clips_dir: Path, detector, device: str, fps: float,
                            cache_path: Path | None = None, refresh: bool = False,
                            exclude_videos=()) -> list[dict]:
    cache_path = cache_path or features_cache_path()
    if cache_path.is_file() and not refresh:
        with cache_path.open() as f:
            rows = list(csv.DictReader(f))
        cached_clips = {r["clip"] for r in rows}
        wanted = {c for c in labels if c.split("/")[0] not in set(exclude_videos or ())}
        if cached_clips == wanted and set(ALL_FEATURE_NAMES) <= set(rows[0]):
            print(f"using cached features from {cache_path} ({len(rows)} clips)")
            return _coerce_row_types(rows)
        print("cache is stale (label set or feature set changed) - re-extracting features")

    print(f"extracting trajectory + net-motion features for {len(labels)} labeled clips "
          f"(runs YOLO on each - this takes a while)...")
    rows = list(build_feature_rows(labels, clips_dir, detector, device, fps,
                                   exclude_videos=exclude_videos))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "shot", "clip", "label", *ALL_FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    print(f"cached features -> {cache_path}")
    return rows


def _coerce_row_types(rows: list[dict]) -> list[dict]:
    int_fields = {"has_crossing", "bounced_back", "n_detections", "n_below_after"}
    for r in rows:
        for k in ALL_FEATURE_NAMES:
            r[k] = int(r[k]) if k in int_fields else float(r[k])
    return rows


def train_and_evaluate(rows: list[dict], min_recall: float):
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import GroupKFold

    X = np.array([[r[f] for f in ALL_FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "goal" else 0 for r in rows])
    groups = np.array([r["video"] for r in rows])
    n_groups = len(set(groups))

    oof_proba = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_groups)
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = GradientBoostingClassifier(random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, oof_proba)
    # precision_recall_curve appends a final (precision=1, recall=0) point with
    # no matching threshold - drop it so precision/recall/thresholds stay aligned.
    precision, recall = precision[:-1], recall[:-1]

    # recall at the lowest threshold in the curve is always 1.0 (predicting
    # positive for every score >= the global minimum score means everyone is
    # kept), so for any min_recall in (0, 1] some threshold always satisfies
    # the constraint - argparse enforces that range in main().
    ok = recall >= min_recall
    best = int(np.argmax(np.where(ok, precision, -1)))
    threshold = float(thresholds[best])
    achieved_precision, achieved_recall = float(precision[best]), float(recall[best])

    kept = oof_proba >= threshold
    baseline_precision = float(y.mean())
    filtered_precision = float(y[kept].mean()) if kept.any() else float("nan")
    n_dropped = int((~kept).sum())
    n_dropped_real_goals = int(((~kept) & (y == 1)).sum())

    final_model = GradientBoostingClassifier(random_state=0)
    final_model.fit(X, y)

    report = {
        "n_examples": len(y),
        "n_videos": n_groups,
        "min_recall_target": min_recall,
        "threshold": threshold,
        "cross_val_precision_at_threshold": achieved_precision,
        "cross_val_recall_at_threshold": achieved_recall,
        "baseline_precision": baseline_precision,
        "n_candidates_dropped": n_dropped,
        "n_real_goals_dropped": n_dropped_real_goals,
    }
    return final_model, report, oof_proba, y, groups


def print_report(report: dict, oof_proba, y, groups):
    import numpy as np

    print()
    print("=" * 60)
    print(f"{report['n_examples']} labeled clips across {report['n_videos']} videos "
          f"(leave-one-video-out cross-validation)")
    print(f"baseline precision (no filter):      {report['baseline_precision']:.1%}")
    print(f"filter threshold (score >= this):    {report['threshold']:.3f}")
    print(f"precision at threshold:              {report['cross_val_precision_at_threshold']:.1%}")
    print(f"recall at threshold:                 {report['cross_val_recall_at_threshold']:.1%}  "
          f"(target >= {report['min_recall_target']:.0%})")
    print(f"candidates the filter would drop:    {report['n_candidates_dropped']} "
          f"of {report['n_examples']}")
    print(f"real goals among those dropped:      {report['n_real_goals_dropped']}")
    print()
    print(f"{'video':45s} {'n':>4} {'precision before':>17} {'precision after':>17}")
    kept = oof_proba >= report["threshold"]
    for video in sorted(set(groups)):
        mask = groups == video
        n = mask.sum()
        before = y[mask].mean()
        after_mask = mask & kept
        after = y[after_mask].mean() if after_mask.any() else float("nan")
        print(f"{video:45s} {n:4d} {before:16.1%} {after:16.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=None,
                         help=f"default: {paths.default_clips_dir()} "
                              f"(or ${paths.CLIPS_DIR_ENV})")
    parser.add_argument("--labels", type=Path, default=None,
                         help="default: <data dir>/dataset/labels.json")
    parser.add_argument("--min-recall", type=float, default=0.98,
                         help="minimum recall required of the chosen threshold, in (0, 1]")
    parser.add_argument("--model", type=str, default=None,
                         help="YOLO weights for feature extraction (yolov8l: validated to catch "
                              "far more of the above/through-hoop trajectory than yolov8m - see README)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--refresh-features", action="store_true",
                         help="re-run YOLO feature extraction even if a matching cache exists")
    parser.add_argument("--exclude-video", action="append", default=[], metavar="VIDEO",
                         help="clips-folder subdirectory to leave out of training; repeatable. "
                              "For footage that isn't what the detector will meet - practice "
                              "shooting produces candidates at a very different rate to a game, "
                              "and mixing the two puts the threshold between two distributions")
    args = parser.parse_args()
    if not (0 < args.min_recall <= 1.0):
        parser.error("--min-recall must be in (0, 1]")

    labels = load_labels() if args.labels is None else json.loads(args.labels.read_text())
    if not labels:
        raise SystemExit("no labels found - label some clips with shot-clipper-label-ui first")

    device = args.device or get_device()
    print(f"device: {device_summary(device)}")
    detector = inference.load_detector(args.model, device=device)
    print(f"backend: {detector.describe()}")

    clips_dir = args.clips_dir or paths.default_clips_dir()
    rows = load_or_build_features(labels, clips_dir, detector, device, args.fps,
                                   exclude_videos=args.exclude_video,
                                   refresh=args.refresh_features)

    final_model, report, oof_proba, y, groups = train_and_evaluate(rows, args.min_recall)
    print_report(report, oof_proba, y, groups)

    import joblib
    model_out, meta_out = model_out_path(), meta_out_path()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_out)
    meta_out.write_text(json.dumps({"feature_names": ALL_FEATURE_NAMES, **report}, indent=2))
    print(f"saved model -> {model_out}")
    print(f"saved meta  -> {meta_out}")


if __name__ == "__main__":
    main()
