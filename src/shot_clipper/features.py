"""Trajectory features for scoring how likely a detected candidate is a real
make, trained on the hand-labeled goal/no_goal dataset (data/dataset/labels.json).

find_makes() in detect_shots.py is a binary yes/no geometric rule, tuned by
hand against a single video (see docs/PLAN.md decision 9) - it doesn't
generalize well (34% precision measured across the full labeled dataset vs.
70% on the one video it was tuned on). extract_features_from_track() computes
continuous descriptive stats about the same above-hoop -> through-hoop event
instead of a single boolean, so a classifier can be trained on real labels
instead of hand-picked constants.

extract_features_from_track() is pure (list of (t, cx, cy) in, dict out).
score_clip() combines it with net_motion.py's pixel-motion features (which
need actual frames, so filtering happens on cut clip files - see
clip_shots.py's --filter-model, not detect_shots.py) - both extracted
straight from the clip file the same way whether training or scoring, so
there's no train/serve skew.
"""
from pathlib import Path

from .detect_shots import (
    ABOVE_WINDOW_FRAC,
    BELOW_WINDOW_FRAC,
    HORIZONTAL_MARGIN_FRAC,
    TARGET_FPS,
    TARGET_HEIGHT,
    compute_roi,
    detect_ball_centers_batch,
    iter_sampled_frames,
)
from .device_config import get_optimal_batch_size

FEATURE_NAMES = [
    "has_crossing",
    "gap_sec",
    "vertical_drop",
    "horiz_offset_above",
    "horiz_offset_through",
    "bounced_back",
    "time_to_bounce",
    "n_detections",
    "detection_rate",
    "n_below_after",
]

# how far past the crossing to look for bounce-back / continued-descent signal
POST_CROSSING_WINDOW_SEC = 2.0
NO_CROSSING_FEATURES = {
    "has_crossing": 0, "gap_sec": 999.0, "vertical_drop": 0.0,
    "horiz_offset_above": 1.0, "horiz_offset_through": 1.0,
    "bounced_back": 0, "time_to_bounce": 999.0,
    "n_detections": 0, "detection_rate": 0.0, "n_below_after": 0,
}


def extract_features_from_track(track, hoop_bbox_norm) -> dict:
    """track: list of (t, cx, cy) with cx/cy possibly None (no detection that
    frame), t relative to some fixed origin (clip start, or a shifted window
    start - the feature values only depend on relative gaps, not absolute t).
    """
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw, hh = hx2 - hx1, hy2 - hy1
    hoop_cx = (hx1 + hx2) / 2
    x_lo = hx1 - hw * HORIZONTAL_MARGIN_FRAC
    x_hi = hx2 + hw * HORIZONTAL_MARGIN_FRAC
    above_y_lo = hy1 - hh * ABOVE_WINDOW_FRAC
    below_y_hi = hy2 + hh * BELOW_WINDOW_FRAC

    detections = [(t, cx, cy) for t, cx, cy in track if cx is not None]
    n_detections = len(detections)
    detection_rate = n_detections / len(track) if track else 0.0

    above_sighting = None  # (t, cx, cy)
    best = None  # (t_above, cx_above, cy_above, t_through, cx_through, cy_through, idx)
    for idx, (t, cx, cy) in enumerate(track):
        if cx is None:
            continue
        if not (x_lo <= cx <= x_hi):
            above_sighting = None
            continue
        if cy < hy1:
            if above_y_lo <= cy:
                above_sighting = (t, cx, cy)
            continue
        if cy > below_y_hi:
            continue
        if hy1 <= cy and above_sighting is not None and best is None:
            t_above, cx_above, cy_above = above_sighting
            best = (t_above, cx_above, cy_above, t, cx, cy, idx)

    if best is None:
        return {**NO_CROSSING_FEATURES, "n_detections": n_detections, "detection_rate": detection_rate}

    t_above, cx_above, cy_above, t_through, cx_through, cy_through, idx = best

    bounced_back = 0
    time_to_bounce = 999.0
    n_below_after = 0
    for t2, cx2, cy2 in track[idx + 1:]:
        if t2 - t_through > POST_CROSSING_WINDOW_SEC:
            break
        if cx2 is None:
            continue
        if cy2 < hy1 and not bounced_back:
            bounced_back = 1
            time_to_bounce = t2 - t_through
        if cy2 > hy1:
            n_below_after += 1

    return {
        "has_crossing": 1,
        "gap_sec": t_through - t_above,
        "vertical_drop": (cy_through - cy_above) / hh,
        "horiz_offset_above": abs(cx_above - hoop_cx) / hw,
        "horiz_offset_through": abs(cx_through - hoop_cx) / hw,
        "bounced_back": bounced_back,
        "time_to_bounce": time_to_bounce,
        "n_detections": n_detections,
        "detection_rate": detection_rate,
        "n_below_after": n_below_after,
    }


def build_ball_track_for_clip(clip_path: Path, hoop_bbox_norm, model, device="cpu", fps: float = TARGET_FPS):
    """Run ball detection over a whole (short) clip file and return its
    (t, cx, cy) track, t relative to the clip's own start (0.0)."""
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    rx1, ry1, rx2, ry2 = compute_roi(hoop_bbox_norm, frame_w, frame_h)

    track = []
    batch_frames, batch_times = [], []
    BATCH_SIZE = get_optimal_batch_size(device)

    def flush():
        if not batch_frames:
            return
        centers = detect_ball_centers_batch(model, batch_frames, device=device,
                                             roi_offset=(rx1, ry1), full_size=(frame_w, frame_h))
        for tt, c in zip(batch_times, centers):
            track.append((tt, c[0], c[1]) if c is not None else (tt, None, None))
        batch_frames.clear()
        batch_times.clear()

    for t, frame in iter_sampled_frames(clip_path, fps, TARGET_HEIGHT, roi=(rx1, ry1, rx2, ry2)):
        batch_frames.append(frame)
        batch_times.append(t)
        if len(batch_frames) >= BATCH_SIZE:
            flush()
    flush()
    return track


def extract_features_for_clip(clip_path: Path, hoop_bbox_norm, model, device="cpu", fps: float = TARGET_FPS) -> dict:
    track = build_ball_track_for_clip(clip_path, hoop_bbox_norm, model, device=device, fps=fps)
    return extract_features_from_track(track, hoop_bbox_norm)


def load_filter_model(model_path: Path, meta_path: Path):
    """Load a classifier trained by train_filter.py plus its metadata
    (feature order, chosen threshold)."""
    import json

    import joblib

    clf = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return clf, meta


def score_clip(clip_path: Path, hoop_bbox_norm, yolo_model, clf, feature_names,
                device: str = "cpu", fps: float = TARGET_FPS) -> float:
    """Score an already-cut candidate clip with a trained filter classifier
    (see train_filter.py). Combines trajectory features (ball detection,
    same as extract_features_for_clip) with net-motion features (pixel
    motion in the net region, no ball detection needed - see net_motion.py)
    - both extracted straight from the clip file, exactly matching how the
    model was trained, so there's no train/serve skew.
    """
    from .net_motion import extract_motion_features_for_clip

    traj_feats = extract_features_for_clip(clip_path, hoop_bbox_norm, yolo_model, device=device, fps=fps)
    motion_feats = extract_motion_features_for_clip(clip_path, hoop_bbox_norm)
    all_feats = {**traj_feats, **motion_feats}
    x = [[all_feats[f] for f in feature_names]]
    return float(clf.predict_proba(x)[0, 1])
