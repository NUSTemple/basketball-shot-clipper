"""AI make/miss suggestions ("pre-labels") for cut clips.

A gradient-boosted classifier scores every candidate shot in a clip from two
kinds of evidence:
- the ball's path around the rim (features.py, on RF-DETR detections);
- how the net moves just after (net_motion.py). This carries most of the
  signal: a make swishes the net in a way a rim-out doesn't.

One model works across cameras because nothing depends on the hand-drawn
hoop box or on the camera's absolute pixel values:
- Geometry comes from the rim RF-DETR detects (detect_rim). The net-motion
  region hangs below that rim, and the trajectory windows use a box scaled
  from it. The old region hung off the hand-drawn box; on the owner's DJI
  videos, whose boxes reached far below the net, it measured the wall.
- Every feature is replaced by its rank among all shots in the same video
  (rank_within_video), which removes camera-wide offsets such as resolution,
  distance and lighting.
Trained on one camera and tested on the other, this scored AUC 0.946 / 0.939
(was 0.661 / 0.776), with 2.5% / 8% of real makes landing in "likely miss".

A merged clip holds several shots less than 5s apart, so each shot is
featurized with every window shifted to sit on it, and the clip scores as
its most make-like shot. Scoring only the first shot hid 7 of 15 real makes
in merged clips; per shot, 2 of 15.

Scores are written per video folder as clip_shots.PRELABEL_FILENAME and shown
in Review as a badge ("likely make / unsure / likely miss"), never as a
label. Raw probabilities are remapped so the model's own cut-offs, chosen on
held-out videos, land on the UI's fixed bins (UI_MISS_BELOW / UI_MAKE_FROM,
which index.html mirrors as AI_MISS_BELOW / AI_MAKE_FROM).

Train with `shot-clipper-train-prelabel` (see main()). In the hosted app,
deploy it as <bucket>/models/prelabel/global.joblib. An optional
<account slug>.joblib next to it overrides that for one account.
"""
import argparse
import json
import os
import re
import statistics
from pathlib import Path

from .clip_shots import PRELABEL_FILENAME, cluster_timestamps
from .features import build_ball_track_for_clip, extract_features_from_track
from .net_motion import extract_motion_features_for_clip

MODEL_PATH = Path(os.environ.get("SHOT_CLIPPER_PRELABEL_MODEL", "models/prelabel.joblib"))
MODEL_DIR = Path(os.environ.get("SHOT_CLIPPER_PRELABEL_MODEL_DIR", "models/prelabel"))
GLOBAL_MODEL_NAME = "global.joblib"
# A bundle records the feature recipe it was trained on. One trained on any
# other recipe is ignored, since its inputs would mean something different.
METHOD = "rim-rank-v1"
# must match clip_shots.cut_all's defaults: a clip is [first - PRE, last + POST]
PRE, POST = 5.0, 2.0
EVENT_AT = PRE
UI_MISS_BELOW, UI_MAKE_FROM = 0.2, 0.6
# at most this share of real makes may land in "likely miss" on held-out videos
MAX_HIDDEN_MAKES = 0.05
MIN_MAKE_PRECISION = 0.85
# bookkeeping fields on a shot row - never model inputs
SHOT_META = {"clip", "group", "video", "event", "n_events", "t_local"}
# a rank needs enough other shots in the video to mean anything
MIN_SHOTS_TO_RANK = 8

# rim geometry
RIM_SAMPLES = 15
RIM_MIN_CONF = 0.5
NET_DEPTH_RIM_WIDTHS = 1.1   # a net hangs about one rim diameter
NET_SIDE_PAD_RIM_WIDTHS = 0.1
# Rim box -> a hoop box the size people draw by hand, which is what
# features.py's window fractions were tuned on (hand boxes measured
# 2.3-2.6x the rim's width and 2.4-2.9x its height on a 1080p camera).
HOOP_W_SCALE, HOOP_H_SCALE = 2.4, 2.6


def event_times(group: list[float]) -> list[float]:
    """Clip-local times of a cluster's shots, given how cut_all cuts it."""
    clip_start = max(0.0, group[0] - PRE)
    return [t - clip_start for t in group]


def detect_rim(video_path: Path, detector, samples: int = RIM_SAMPLES) -> list[float] | None:
    """Median pixel box of the video's main rim, from frames sampled across
    the whole video. "Main" is the rim position seen most often, so a second
    hoop in the background or a brief camera bump doesn't drag the median."""
    import cv2

    rim_class = detector.class_ids["rim"]
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boxes = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * n / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detector.detect_batch([frame], classes=(rim_class,), conf=RIM_MIN_CONF, imgsz=0)[0]
        if dets:
            d = max(dets, key=lambda d: d.conf)
            boxes.append((d.x1, d.y1, d.x2, d.y2))
    cap.release()
    if not boxes:
        return None
    width = statistics.median(b[2] - b[0] for b in boxes)

    def centre(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def near(a, b):
        (ax, ay), (bx, by) = centre(a), centre(b)
        return abs(ax - bx) < width and abs(ay - by) < width

    anchor = max(boxes, key=lambda a: sum(near(a, b) for b in boxes))
    main = [b for b in boxes if near(anchor, b)]
    return [statistics.median(b[i] for b in main) for i in range(4)]


def rim_hoop_box(rim_px, frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """Normalized hoop box for features.py, scaled from the detected rim."""
    x1, y1, x2, y2 = rim_px
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * HOOP_W_SCALE, (y2 - y1) * HOOP_H_SCALE
    return ((cx - w / 2) / frame_w, (cy - h / 2) / frame_h, (cx + w / 2) / frame_w, (cy + h / 2) / frame_h)


def rim_net_roi(rim_px, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Pixel region where the net hangs: rim mid-height down one rim width."""
    x1, y1, x2, y2 = rim_px
    rw = x2 - x1
    return (max(0, int(x1 - NET_SIDE_PAD_RIM_WIDTHS * rw)), max(0, int((y1 + y2) / 2)),
            min(frame_w, int(x2 + NET_SIDE_PAD_RIM_WIDTHS * rw)), min(frame_h, int(y2 + NET_DEPTH_RIM_WIDTHS * rw)))


def shot_features(track, clip_path: Path, hoop_bbox_norm, net_roi, local_t: float) -> dict:
    """Features for the shot at clip-local local_t, windowed exactly as for a
    single-shot clip's shot at EVENT_AT."""
    shift = local_t - EVENT_AT
    window = [(t - shift, cx, cy) for t, cx, cy in track if local_t - PRE <= t <= local_t + POST]
    return {**extract_features_from_track(window, hoop_bbox_norm),
            **extract_motion_features_for_clip(clip_path, hoop_bbox_norm, shift=shift, roi=net_roi)}


def clip_shot_features(clip_path: Path, rim_px, local_times, detector, device="cpu", track=None) -> list[dict]:
    """One feature row per shot in the clip. track lets a caller reuse a ball
    track it already has; otherwise RF-DETR runs over the clip."""
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    hoop = rim_hoop_box(rim_px, frame_w, frame_h)
    roi = rim_net_roi(rim_px, frame_w, frame_h)
    if track is None:
        track = build_ball_track_for_clip(clip_path, hoop, detector, device=device)
    return [shot_features(track, clip_path, hoop, roi, t) for t in local_times]


def rank_within_video(shot_rows: list[dict], features: list[str]) -> list[list[float]]:
    """Each feature -> its rank among all shots of one video, scaled to 0..1
    (ties share the average rank)."""
    from scipy.stats import rankdata

    n = len(shot_rows)
    cols = []
    for f in features:
        r = rankdata([row[f] for row in shot_rows])
        cols.append([(v - 1) / (n - 1) if n > 1 else 0.5 for v in r])
    return [list(vals) for vals in zip(*cols)]


def remap(raw: float, lo: float, hi: float) -> float:
    """Monotonic map putting the model's cut-offs lo/hi on the UI bins."""
    if raw < lo:
        return UI_MISS_BELOW * raw / lo if lo > 0 else 0.0
    if raw < hi:
        return UI_MISS_BELOW + (UI_MAKE_FROM - UI_MISS_BELOW) * (raw - lo) / (hi - lo)
    return UI_MAKE_FROM + (1 - UI_MAKE_FROM) * (raw - hi) / (1 - hi) if hi < 1 else 1.0


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """The trained bundle, or None when there's none for this recipe -
    scoring is then simply skipped."""
    if not Path(path).is_file():
        return None
    import joblib

    bundle = joblib.load(path)
    return bundle if isinstance(bundle, dict) and bundle.get("method") == METHOD else None


def account_slug(email: str) -> str:
    """Same folder name label_ui.app.user_slug gives an account."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", email)


def model_for(user: str | None) -> dict | None:
    """The model to score a job with: in the hosted app the account's own
    override if one exists, else the global model; for a local run with no
    account, MODEL_PATH. None means don't score."""
    if not user:
        return load_model(MODEL_PATH)
    return load_model(MODEL_DIR / f"{account_slug(user)}.joblib") or load_model(MODEL_DIR / GLOBAL_MODEL_NAME)


def _raw_clip_scores(model, features, rows_by_clip: dict[str, list[dict]]) -> dict[str, float]:
    """Max make-probability per clip, with features ranked across every shot
    of the clips given - which must be one whole video."""
    import numpy as np

    clips = list(rows_by_clip)
    flat = [(c, row) for c in clips for row in rows_by_clip[c]]
    X = np.array(rank_within_video([row for _, row in flat], features))
    p = model.predict_proba(X)[:, 1]
    out = {}
    for (c, _), prob in zip(flat, p):
        out[c] = max(out.get(c, 0.0), float(prob))
    return out


def score_video(bundle: dict, rows_by_clip: dict[str, list[dict]]) -> dict[str, float]:
    """Display scores for all clips of ONE video, or {} when it has too few
    shots for ranks to mean anything."""
    if sum(len(r) for r in rows_by_clip.values()) < MIN_SHOTS_TO_RANK:
        return {}
    raw = _raw_clip_scores(bundle["model"], bundle["features"], rows_by_clip)
    return {c: round(remap(s, bundle["lo"], bundle["hi"]), 3) for c, s in raw.items()}


def score_cut_clips(cut_results, makes, rim_px, outdir: Path, bundle: dict, detector,
                    device="cpu", progress_cb=None, cancel_check=None) -> dict[str, float]:
    """Score the clips cut_all just wrote for one video (cut_results:
    (index, t, path)) and write outdir/PRELABEL_FILENAME. makes is the same
    timestamp list cut_all clustered, so cluster i belongs to shot_{i:03d}.
    All clips are featurized first, since ranks need the whole video."""
    groups = cluster_timestamps(makes)
    rows = {}
    ordered = sorted(cut_results)
    for n, (i, _, path) in enumerate(ordered, start=1):
        if cancel_check and cancel_check():
            return {}
        rows[Path(path).name] = clip_shot_features(Path(path), rim_px, event_times(groups[i - 1]), detector,
                                                   device=device)
        if progress_cb:
            progress_cb(n, len(ordered))
    scores = score_video(bundle, rows)
    if scores:
        (outdir / PRELABEL_FILENAME).write_text(json.dumps(scores, indent=1, sort_keys=True))
    return scores


def bin_counts(scores) -> dict:
    vals = list(scores.values()) if isinstance(scores, dict) else list(scores)
    return {"make": sum(s >= UI_MAKE_FROM for s in vals),
            "unsure": sum(UI_MISS_BELOW <= s < UI_MAKE_FROM for s in vals),
            "miss": sum(s < UI_MISS_BELOW for s in vals)}


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def new_classifier():
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=2,
                                      learning_rate=0.05, subsample=0.8)


def _ranked(rows: dict[str, list[dict]], group_of: dict[str, str], features) -> dict[str, list[list[float]]]:
    """Per clip, its shots' feature vectors ranked within the clip's video
    (group) - over every shot of that video, labeled or not, exactly as
    scoring ranks a freshly cut video. Videos too small to rank are dropped."""
    by_group: dict[str, list[str]] = {}
    for c in rows:
        by_group.setdefault(group_of[c], []).append(c)
    out = {}
    for clips in by_group.values():
        flat = [(c, row) for c in clips for row in rows[c]]
        if len(flat) < MIN_SHOTS_TO_RANK:
            continue
        for (c, _), vec in zip(flat, rank_within_video([row for _, row in flat], features)):
            out.setdefault(c, []).append(vec)
    return out


def _fit(ranked, labels, clips):
    """Training rows: every shot of a single-shot clip, and every shot of a
    merged clip labeled no_goal (they all missed). A merged clip labeled goal
    is left out - which of its shots went in is unknown."""
    import numpy as np

    X, y = [], []
    for c in clips:
        shots = ranked[c]
        if len(shots) > 1 and labels[c] == "goal":
            continue
        X.extend(shots)
        y.extend([labels[c] == "goal"] * len(shots))
    return new_classifier().fit(np.array(X), np.array(y))


def _clip_score(model, ranked_shots) -> float:
    import numpy as np

    return float(max(model.predict_proba(np.array(ranked_shots))[:, 1]))


def train(rows: dict[str, list[dict]], labels: dict[str, str], group_of: dict[str, str]) -> tuple[dict, dict]:
    """rows: clip -> its shots' feature dicts, for EVERY clip of each video
    (unlabeled ones too - they shape the within-video ranks); labels: clip ->
    goal/no_goal; group_of: clip -> its video. Cut-offs come from
    leave-one-video-out predictions, so they reflect footage the model
    hasn't seen. Returns (bundle, report)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    first = next(iter(rows.values()))[0]
    features = sorted(k for k, v in first.items() if k not in SHOT_META and isinstance(v, (int, float)))
    ranked = _ranked(rows, group_of, features)
    clips = sorted(c for c in ranked if c in labels)
    groups = sorted({group_of[c] for c in clips})
    oof = {}
    for g in groups:
        model = _fit(ranked, labels, [c for c in clips if group_of[c] != g])
        for c in clips:
            if group_of[c] == g:
                oof[c] = _clip_score(model, ranked[c])
    y = np.array([labels[c] == "goal" for c in clips])
    p = np.array([oof[c] for c in clips])
    cands = sorted(set(np.round(p, 4)))
    lo = max([t for t in cands if (y & (p < t)).sum() <= MAX_HIDDEN_MAKES * y.sum()], default=0.0)
    hi = min([t for t in cands if t > lo and (p >= t).sum() >= 5 and y[p >= t].mean() >= MIN_MAKE_PRECISION],
             default=max(lo + 1e-3, 0.5))
    report = {
        "method": METHOD, "n_clips": len(clips), "n_goals": int(y.sum()), "groups": groups,
        "held_out_auc": round(float(roc_auc_score(y, p)), 3),
        "lo": float(lo), "hi": float(hi),
        "held_out_likely_miss": int((p < lo).sum()), "held_out_makes_hidden": int((y & (p < lo)).sum()),
        "held_out_likely_make": int((p >= hi).sum()),
        "held_out_make_precision": round(float(y[p >= hi].mean()), 3) if (p >= hi).any() else None,
    }
    bundle = {"method": METHOD, "model": _fit(ranked, labels, clips), "features": features,
              "lo": float(lo), "hi": float(hi), "report": report}
    return bundle, report


def main() -> None:
    """shot-clipper-train-prelabel: fit from per-shot feature rows.

    --shots is a JSON-lines file, one line per shot of EVERY clip of each
    video (unlabeled clips too - they shape the within-video ranks):
    {"clip": <label key>, "group": <video>, <feature>: value, ...}, with
    features from clip_shot_features() - the same function scoring uses, so
    training and serving can't drift apart.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--shots", type=Path, required=True, help="per-shot feature rows (JSON lines)")
    parser.add_argument("--labels", type=Path, required=True, help="labels.json (clip key -> {label: ...})")
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    import joblib

    rows, group_of = {}, {}
    for line in args.shots.open():
        r = json.loads(line)
        clip = r.pop("clip")
        group_of[clip] = r.pop("group")
        rows.setdefault(clip, []).append({k: v for k, v in r.items()
                                          if k not in SHOT_META and isinstance(v, (int, float))})
    labels = {k: v["label"] for k, v in json.loads(args.labels.read_text()).items()
              if v.get("label") in ("goal", "no_goal")}
    bundle, report = train(rows, labels, group_of)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(json.dumps(report, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
