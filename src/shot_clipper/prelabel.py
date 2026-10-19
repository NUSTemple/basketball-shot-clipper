"""AI make/miss suggestions ("pre-labels") for cut clips.

A gradient-boosted classifier scores every candidate shot in a clip from two
kinds of evidence:
- the ball's path around the rim (features.py, on RF-DETR detections);
- how the net moves just after (net_motion.py). This carries most of the
  signal: a make swishes the net in a way a rim-out doesn't.

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
deploy a model to <bucket>/models/prelabel/<account slug>.joblib.
"""
import argparse
import json
import os
import re
from pathlib import Path

from .clip_shots import PRELABEL_FILENAME, cluster_timestamps
from .features import build_ball_track_for_clip, extract_features_from_track
from .net_motion import extract_motion_features_for_clip

# Single-user/local runs use one model file. The hosted app uses one model
# per account (MODEL_DIR/<account slug>.joblib): a model only works on the
# camera it was trained on. A friend-camera model scored the owner's DJI
# clips at AUC 0.67 and put 36 of 80 real makes in "likely miss", so an
# account with no model of its own gets no scores rather than a wrong one.
MODEL_PATH = Path(os.environ.get("SHOT_CLIPPER_PRELABEL_MODEL", "models/prelabel.joblib"))
MODEL_DIR = Path(os.environ.get("SHOT_CLIPPER_PRELABEL_MODEL_DIR", "models/prelabel"))
# must match clip_shots.cut_all's defaults: a clip is [first - PRE, last + POST]
PRE, POST = 5.0, 2.0
EVENT_AT = PRE
UI_MISS_BELOW, UI_MAKE_FROM = 0.2, 0.6
# at most this share of real makes may land in "likely miss" on held-out videos
MAX_HIDDEN_MAKES = 0.05
MIN_MAKE_PRECISION = 0.85
# bookkeeping fields on a shot row - never model inputs
SHOT_META = {"clip", "group", "video", "event", "n_events", "t_local"}


def event_times(group: list[float]) -> list[float]:
    """Clip-local times of a cluster's shots, given how cut_all cuts it."""
    clip_start = max(0.0, group[0] - PRE)
    return [t - clip_start for t in group]


def shot_features(track, clip_path: Path, hoop_bbox_norm, local_t: float) -> dict:
    """Features for the shot at clip-local local_t, windowed exactly as for a
    single-shot clip's shot at EVENT_AT."""
    shift = local_t - EVENT_AT
    window = [(t - shift, cx, cy) for t, cx, cy in track if local_t - PRE <= t <= local_t + POST]
    return {**extract_features_from_track(window, hoop_bbox_norm),
            **extract_motion_features_for_clip(clip_path, hoop_bbox_norm, shift=shift)}


def clip_shot_features(clip_path: Path, hoop_bbox_norm, local_times, detector, device="cpu") -> list[dict]:
    track = build_ball_track_for_clip(clip_path, hoop_bbox_norm, detector, device=device)
    return [shot_features(track, clip_path, hoop_bbox_norm, t) for t in local_times]


def remap(raw: float, lo: float, hi: float) -> float:
    """Monotonic map putting the model's cut-offs lo/hi on the UI bins."""
    if raw < lo:
        return UI_MISS_BELOW * raw / lo if lo > 0 else 0.0
    if raw < hi:
        return UI_MISS_BELOW + (UI_MAKE_FROM - UI_MISS_BELOW) * (raw - lo) / (hi - lo)
    return UI_MAKE_FROM + (1 - UI_MAKE_FROM) * (raw - hi) / (1 - hi) if hi < 1 else 1.0


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """The trained bundle, or None when no model has been trained/deployed -
    scoring is then simply skipped."""
    if not Path(path).is_file():
        return None
    import joblib

    return joblib.load(path)


def account_slug(email: str) -> str:
    """Same folder name label_ui.app.user_slug gives an account."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", email)


def model_for(user: str | None) -> dict | None:
    """The model to score a job with: the job owner's own model in the
    hosted app, or MODEL_PATH for a local run with no account. None means
    don't score."""
    if user:
        return load_model(MODEL_DIR / f"{account_slug(user)}.joblib")
    return load_model(MODEL_PATH)


def score_shots(bundle: dict, shot_rows: list[dict]) -> float:
    import numpy as np

    X = np.array([[row[f] for f in bundle["features"]] for row in shot_rows])
    raw = float(max(bundle["model"].predict_proba(X)[:, 1]))
    return round(remap(raw, bundle["lo"], bundle["hi"]), 3)


def score_cut_clips(cut_results, makes, hoop_bbox_norm, outdir: Path, bundle: dict, detector,
                    device="cpu", progress_cb=None, cancel_check=None) -> dict[str, float]:
    """Score the clips cut_all just wrote (cut_results: (index, t, path))
    and write outdir/PRELABEL_FILENAME. makes is the same timestamp list
    cut_all clustered, so cluster i belongs to shot_{i:03d}."""
    groups = cluster_timestamps(makes)
    scores = {}
    ordered = sorted(cut_results)
    for n, (i, _, path) in enumerate(ordered, start=1):
        if cancel_check and cancel_check():
            break
        rows = clip_shot_features(Path(path), hoop_bbox_norm, event_times(groups[i - 1]), detector, device=device)
        scores[Path(path).name] = score_shots(bundle, rows)
        if progress_cb:
            progress_cb(n, len(ordered))
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


def _fit(rows, labels, clips, features):
    """Training rows: every shot of a single-shot clip, and every shot of a
    merged clip labeled no_goal (they all missed). A merged clip labeled goal
    is left out - which of its shots went in is unknown."""
    import numpy as np

    X, y = [], []
    for c in clips:
        shots = rows[c]
        if len(shots) > 1 and labels[c] == "goal":
            continue
        for s in shots:
            X.append([s[f] for f in features])
            y.append(labels[c] == "goal")
    return new_classifier().fit(np.array(X), np.array(y))


def _raw_clip_score(model, shots, features):
    import numpy as np

    return float(max(model.predict_proba(np.array([[s[f] for f in features] for s in shots]))[:, 1]))


def train(rows: dict[str, list[dict]], labels: dict[str, str], group_of: dict[str, str]) -> tuple[dict, dict]:
    """rows: clip -> its shots' feature dicts; labels: clip -> goal/no_goal;
    group_of: clip -> the video (or camera) it came from. Cut-offs come from
    leave-one-group-out predictions, so they reflect footage the model hasn't
    seen. Returns (bundle, report)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    clips = sorted(c for c in rows if c in labels)
    features = sorted(k for k, v in rows[clips[0]][0].items()
                      if k not in SHOT_META and isinstance(v, (int, float)))
    groups = sorted({group_of[c] for c in clips})
    oof = {}
    for g in groups:
        model = _fit(rows, labels, [c for c in clips if group_of[c] != g], features)
        for c in clips:
            if group_of[c] == g:
                oof[c] = _raw_clip_score(model, rows[c], features)
    y = np.array([labels[c] == "goal" for c in clips])
    p = np.array([oof[c] for c in clips])
    cands = sorted(set(np.round(p, 4)))
    lo = max([t for t in cands if (y & (p < t)).sum() <= MAX_HIDDEN_MAKES * y.sum()], default=0.0)
    hi = min([t for t in cands if t > lo and (p >= t).sum() >= 5 and y[p >= t].mean() >= MIN_MAKE_PRECISION],
             default=max(lo + 1e-3, 0.5))
    report = {
        "n_clips": len(clips), "n_goals": int(y.sum()), "groups": groups,
        "held_out_auc": round(float(roc_auc_score(y, p)), 3),
        "lo": float(lo), "hi": float(hi),
        "held_out_likely_miss": int((p < lo).sum()), "held_out_makes_hidden": int((y & (p < lo)).sum()),
        "held_out_likely_make": int((p >= hi).sum()),
        "held_out_make_precision": round(float(y[p >= hi].mean()), 3) if (p >= hi).any() else None,
    }
    bundle = {"model": _fit(rows, labels, clips, features), "features": features,
              "lo": float(lo), "hi": float(hi), "report": report}
    return bundle, report


def main() -> None:
    """shot-clipper-train-prelabel: fit from per-shot feature rows.

    --shots is a JSON-lines file, one line per shot: {"clip": <label key>,
    "group": <video or camera id>, <feature>: value, ...}, with features from
    clip_shot_features() - the same function scoring uses, so training and
    serving can't drift apart.
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
