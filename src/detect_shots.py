"""Detect made shots in a fixed-camera basketball video.

Approach (see PLAN.md decisions 3/5/10):
  - Run YOLO (COCO "sports ball" class) on frames at native resolution,
    sampled at ~15fps (spatial downsampling killed ball recall in testing).
  - Track the ball center per sampled frame.
  - A "make" is flagged when the ball's trajectory crosses the calibrated
    hoop box moving downward: seen above the hoop, then seen inside/below it
    with a small horizontal offset, within a short time window.
  - Recall is prioritized over precision (decision 5): the geometric window
    is intentionally generous, and a cooldown just prevents counting the
    same make twice.

Usage:
    python src/detect_shots.py <video_path> [--config configs/<name>.json]
        [--output ground_truth/<name>_detected.json]
"""
import argparse
import json
from pathlib import Path

import cv2

COCO_SPORTS_BALL_CLASS = 32
# Ball detection needs high spatial resolution: a basketball at 720p is only
# ~10px wide after YOLO's internal resize and yolov8n/720p misses it almost
# entirely (empirically ~1% recall). Run at native frame resolution with a
# mid-size model instead; only the timestamp sampling is downsampled.
TARGET_HEIGHT = None  # None = native resolution, no spatial downsample
TARGET_FPS = 15.0
BALL_CONF_THRESHOLD = 0.1  # recall-first (decision 5): low bar, filtered by trajectory logic
# how far outside the hoop box (as a fraction of box width/height) a ball
# center still counts as "near" the rim, to tolerate detection jitter.
HORIZONTAL_MARGIN_FRAC = 0.6
ABOVE_WINDOW_FRAC = 2.5  # how far above the hoop counts as "approaching"
MAKE_COOLDOWN_SEC = 1.5
MAX_GAP_SEC = 0.6  # max time between "above" and "through" sightings


def load_config(config_path: Path):
    cfg = json.loads(config_path.read_text())
    x1, y1, x2, y2 = cfg["hoop_bbox_norm"]
    return x1, y1, x2, y2


def iter_sampled_frames(video_path: Path, target_fps: float, target_height: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(src_fps / target_fps))
    scale = target_height / src_h if target_height else None

    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if scale is not None:
                frame = cv2.resize(frame, (round(src_w * scale), target_height))
            timestamp = idx / src_fps
            yield timestamp, frame
        idx += 1
    cap.release()


def detect_ball_center(model, frame, device="cpu"):
    imgsz = max(frame.shape[0], frame.shape[1])
    imgsz = (imgsz + 31) // 32 * 32  # round up to multiple of 32
    results = model.predict(frame, classes=[COCO_SPORTS_BALL_CLASS], conf=BALL_CONF_THRESHOLD,
                             imgsz=imgsz, device=device, verbose=False)
    best = None
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if best is None or conf > best[0]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2 / frame.shape[1]
                cy = (y1 + y2) / 2 / frame.shape[0]
                best = (conf, cx, cy)
    if best is None:
        return None
    _, cx, cy = best
    return cx, cy


def find_makes(ball_track, hoop_bbox_norm):
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw = hx2 - hx1
    hh = hy2 - hy1
    x_lo = hx1 - hw * HORIZONTAL_MARGIN_FRAC
    x_hi = hx2 + hw * HORIZONTAL_MARGIN_FRAC
    above_y_lo = hy1 - hh * ABOVE_WINDOW_FRAC

    makes = []
    last_make_t = -1e9
    above_sighting = None  # (t, cy)

    for t, cx, cy in ball_track:
        if cx is None:
            continue
        near_x = x_lo <= cx <= x_hi
        if not near_x:
            above_sighting = None
            continue

        if cy < hy1:
            if above_y_lo <= cy:
                above_sighting = (t, cy)
            continue

        # ball is at-or-below the hoop's top edge, i.e. inside/through the box
        if hy1 <= cy and above_sighting is not None:
            t_above, cy_above = above_sighting
            if (t - t_above) <= MAX_GAP_SEC and cy > cy_above and t - last_make_t > MAKE_COOLDOWN_SEC:
                makes.append(round((t_above + t) / 2, 2))
                last_make_t = t
                above_sighting = None

    return makes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                         help="default: configs/<video_stem>.json")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: ground_truth/<video_stem>_detected.json")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--device", type=str, default=None,
                         help="cpu/mps/cuda; default: mps if available else cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = args.config or root / "configs" / f"{args.video.stem}.json"
    output_path = args.output or root / "ground_truth" / f"{args.video.stem}_detected.json"

    hoop_bbox_norm = load_config(config_path)

    import torch
    from ultralytics import YOLO
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(args.model)

    ball_track = []
    for t, frame in iter_sampled_frames(args.video, TARGET_FPS, TARGET_HEIGHT):
        center = detect_ball_center(model, frame, device=device)
        if center is None:
            ball_track.append((t, None, None))
        else:
            cx, cy = center
            ball_track.append((t, cx, cy))

    makes = find_makes(ball_track, hoop_bbox_norm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"video": args.video.name, "makes_sec": makes}, indent=2))
    print(f"detected {len(makes)} makes -> {output_path}")
    for m in makes:
        print(f"  {m:.2f}s")


if __name__ == "__main__":
    main()
