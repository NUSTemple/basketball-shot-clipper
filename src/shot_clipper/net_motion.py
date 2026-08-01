"""Net-motion features: does the net move like it was just shot through?

Complements the ball-trajectory features in features.py. Validated (see
docs/PLAN.md and README) on the labeled dataset: motion features alone carry
almost no signal (34.3% vs 34.2% baseline precision), but combined with
trajectory features they nearly double how many false positives a trained
filter can safely drop at the same recall target (35 -> 60 of 307, same 2
real goals at risk). Needs no ball detection - pure pixel motion in the net
region, so it's cheap (~2s/clip) compared to the YOLO-based trajectory pass.

Every candidate clip is cut as [t_make - 5s, t_make + 2s] (clip_shots.py), so
the candidate event sits at clip-local t=5.0s in (almost) every clip; that's
what POST_WINDOW/BASELINE_WINDOW are relative to.
"""
from pathlib import Path

MOTION_FEATURE_NAMES = [
    "diff_peak",
    "diff_time_to_peak",
    "diff_decay_ratio",
    "diff_impulsiveness",
    "baseline_mean",
    "diff_peak_over_baseline",
    "flow_coherence_peak",
    "flow_coherence_mean",
    "baseline_flow_coherence_mean",
]

# narrow ROI right below the rim: wide enough to catch the net, narrow enough
# to mostly avoid rebounding players reaching in from the side
NET_Y_MARGIN_FRAC = 0.9
NET_X_MARGIN_FRAC = 0.05

POST_WINDOW = (4.8, 6.5)      # clip-local seconds: around + just after the candidate event
BASELINE_WINDOW = (1.0, 2.5)  # well before the event: per-clip background-jitter control


def net_roi_px(hoop_bbox_norm, frame_w: int, frame_h: int):
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw, hh = hx2 - hx1, hy2 - hy1
    x1 = max(0.0, hx1 - hw * NET_X_MARGIN_FRAC)
    x2 = min(1.0, hx2 + hw * NET_X_MARGIN_FRAC)
    y1 = hy2
    y2 = min(1.0, hy2 + hh * NET_Y_MARGIN_FRAC)
    return (round(x1 * frame_w), round(y1 * frame_h), round(x2 * frame_w), round(y2 * frame_h))


def read_window_frames(cap, roi, fps: float, window: tuple[float, float]):
    """Grayscale frames cropped to roi, for the [start, end] clip-local
    seconds window. cap: an already-open cv2.VideoCapture."""
    import cv2

    x1, y1, x2, y2 = roi
    start_frame = int(window[0] * fps)
    end_frame = int(window[1] * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []

    for _ in range(end_frame - start_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))
    return frames


def diff_series(frames) -> list[float]:
    """Mean absolute frame-to-frame pixel difference, one value per
    consecutive frame pair."""
    import cv2

    return [float(cv2.absdiff(frames[i], frames[i - 1]).mean()) for i in range(1, len(frames))]


def flow_coherence_series(frames) -> list[float]:
    """Per-frame-pair optical flow directional coherence: |sum of flow
    vectors| / sum of |flow vectors|. Near 1 = all motion pointing the same
    way (consistent with a swinging net); near 0 = motion pointing in random
    directions (consistent with chaotic activity - hands, occlusion, several
    things moving at once)."""
    import cv2
    import numpy as np

    coherences = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(frames[i - 1], frames[i], None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx ** 2 + fy ** 2)
        total_mag = mag.sum()
        if total_mag < 1e-6:
            coherences.append(0.0)
            continue
        resultant = np.sqrt(fx.sum() ** 2 + fy.sum() ** 2)
        coherences.append(float(resultant / total_mag))
    return coherences


def shape_features(series: list[float], prefix: str) -> dict:
    """Describes the *shape* of a motion time series - a real net swish
    should be a short sharp burst that decays, not sustained/erratic
    activity, so peak magnitude alone isn't enough to tell them apart."""
    import numpy as np

    if not series:
        return {f"{prefix}_peak": 0.0, f"{prefix}_time_to_peak": 0.0,
                f"{prefix}_decay_ratio": 0.0, f"{prefix}_impulsiveness": 0.0}
    arr = np.array(series)
    peak = float(arr.max())
    peak_idx = int(arr.argmax())
    time_to_peak = peak_idx / len(arr)
    tail = arr[peak_idx:]
    decay_ratio = float(tail[-1] / peak) if peak > 1e-6 else 0.0
    median_excl_peak = float(np.median(np.delete(arr, peak_idx))) if len(arr) > 1 else 0.0
    impulsiveness = peak / (median_excl_peak + 1e-3)
    return {f"{prefix}_peak": peak, f"{prefix}_time_to_peak": time_to_peak,
            f"{prefix}_decay_ratio": decay_ratio, f"{prefix}_impulsiveness": impulsiveness}


def extract_motion_features_for_clip(clip_path: Path, hoop_bbox_norm) -> dict:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = net_roi_px(hoop_bbox_norm, frame_w, frame_h)

    post_frames = read_window_frames(cap, roi, fps, POST_WINDOW)
    baseline_frames = read_window_frames(cap, roi, fps, BASELINE_WINDOW)
    cap.release()

    post_diff = diff_series(post_frames)
    baseline_diff = diff_series(baseline_frames)
    post_flow_coh = flow_coherence_series(post_frames)
    baseline_flow_coh = flow_coherence_series(baseline_frames)

    feats = shape_features(post_diff, "diff")
    feats["baseline_mean"] = float(np.mean(baseline_diff)) if baseline_diff else 0.0
    feats["diff_peak_over_baseline"] = feats["diff_peak"] / (feats["baseline_mean"] + 1e-3)
    feats["flow_coherence_peak"] = float(max(post_flow_coh)) if post_flow_coh else 0.0
    feats["flow_coherence_mean"] = float(np.mean(post_flow_coh)) if post_flow_coh else 0.0
    feats["baseline_flow_coherence_mean"] = float(np.mean(baseline_flow_coh)) if baseline_flow_coh else 0.0
    return feats
