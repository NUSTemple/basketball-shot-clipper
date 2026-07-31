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
    shot-clipper-detect <video_path> [--config data/configs/<name>.json]
        [--output data/ground_truth/<name>_detected.json]

Run from the repo root (or pass explicit paths) - defaults are resolved
relative to the current directory, e.g. data/configs/, models/yolov8m.pt.
"""
import argparse
import json
import time
from pathlib import Path

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
BELOW_WINDOW_FRAC = 1.0  # how far below the hoop still counts as "through" (was unbounded)
MAKE_COOLDOWN_SEC = 1.5
MAX_GAP_SEC = 0.6  # max time between "above" and "through" sightings
# If the ball reappears above the rim within this window after a "through"
# sighting, that's an unambiguous rim-bounce (miss), not a make. Validated
# against a fully human-checked video (DJI_20260725170144_0010_D): catches
# 2 of 3 known false positives with zero recall loss on the 8 known makes.
BOUNCE_BACK_CONFIRM_SEC = 0.4

# find_makes() only ever looks at ball positions within HORIZONTAL_MARGIN_FRAC of
# the hoop horizontally, and above hoop_y2+BELOW_WINDOW_FRAC*hoop_height vertically
# (the court floor below that, where players stand, is never consulted). Cropping
# YOLO's input to that region before inference - instead of feeding it the full
# 4K frame - cuts pixel count roughly 8-10x with zero effect on detection results,
# since ball detections outside that region get discarded by find_makes anyway.
CROP_X_MARGIN_FRAC = 1.5  # extra horizontal buffer beyond HORIZONTAL_MARGIN_FRAC
CROP_Y_MARGIN_FRAC = 0.5  # extra buffer below BELOW_WINDOW_FRAC


def load_config(config_path: Path):
    cfg = json.loads(config_path.read_text())
    x1, y1, x2, y2 = cfg["hoop_bbox_norm"]
    return x1, y1, x2, y2


def compute_roi(hoop_bbox_norm, frame_w, frame_h):
    """Pixel-space crop box covering everything find_makes() can possibly use."""
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw, hh = hx2 - hx1, hy2 - hy1
    x_margin = hw * (HORIZONTAL_MARGIN_FRAC + CROP_X_MARGIN_FRAC)
    y_bottom_margin = hh * (BELOW_WINDOW_FRAC + CROP_Y_MARGIN_FRAC)
    x1 = max(0, round((hx1 - x_margin) * frame_w))
    x2 = min(frame_w, round((hx2 + x_margin) * frame_w))
    y1 = 0  # find_makes has no lower bound on how far above the hoop counts
    y2 = min(frame_h, round((hy2 + y_bottom_margin) * frame_h))
    return x1, y1, x2, y2


def iter_sampled_frames(video_path: Path, target_fps: float, target_height: int | None, roi=None):
    import cv2

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
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                frame = frame[ry1:ry2, rx1:rx2]
            if scale is not None:
                frame = cv2.resize(frame, (round(src_w * scale), target_height))
            timestamp = idx / src_fps
            yield timestamp, frame
        idx += 1
    cap.release()


def detect_ball_center(model, frame, device="cpu", roi_offset=(0, 0), full_size=None):
    """roi_offset/full_size let the caller pass a cropped frame while getting
    back cx,cy normalized against the ORIGINAL full frame (so they compare
    directly against hoop_bbox_norm)."""
    centers = detect_ball_centers_batch(model, [frame], device=device,
                                         roi_offset=roi_offset, full_size=full_size)
    return centers[0]


def detect_ball_centers_batch(model, frames, device="cpu", roi_offset=(0, 0), full_size=None):
    """Batched version of detect_ball_center - one model.predict() call for
    several frames instead of one call per frame. Purely an efficiency
    change: per-frame results are identical to calling detect_ball_center in
    a loop, just cheaper in call overhead."""
    if not frames:
        return []
    imgsz = max(frames[0].shape[0], frames[0].shape[1])
    imgsz = (imgsz + 31) // 32 * 32  # round up to multiple of 32
    results = model.predict(frames, classes=[COCO_SPORTS_BALL_CLASS], conf=BALL_CONF_THRESHOLD,
                             imgsz=imgsz, device=device, verbose=False)
    full_w, full_h = full_size or (frames[0].shape[1], frames[0].shape[0])
    ox, oy = roi_offset
    centers = []
    for r in results:
        best = None
        for box in r.boxes:
            conf = float(box.conf[0])
            if best is None or conf > best[0]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (ox + (x1 + x2) / 2) / full_w
                cy = (oy + (y1 + y2) / 2) / full_h
                best = (conf, cx, cy)
        centers.append((best[1], best[2]) if best is not None else None)
    return centers


def find_makes(ball_track, hoop_bbox_norm):
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw = hx2 - hx1
    hh = hy2 - hy1
    x_lo = hx1 - hw * HORIZONTAL_MARGIN_FRAC
    x_hi = hx2 + hw * HORIZONTAL_MARGIN_FRAC
    above_y_lo = hy1 - hh * ABOVE_WINDOW_FRAC
    below_y_hi = hy2 + hh * BELOW_WINDOW_FRAC

    makes = []
    last_make_t = -1e9
    above_sighting = None  # (t, cy)
    n = len(ball_track)

    for idx, (t, cx, cy) in enumerate(ball_track):
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

        if cy > below_y_hi:
            continue

        # ball is at-or-below the hoop's top edge, i.e. inside/through the box
        if hy1 <= cy and above_sighting is not None:
            t_above, cy_above = above_sighting
            if (t - t_above) <= MAX_GAP_SEC and cy > cy_above and t - last_make_t > MAKE_COOLDOWN_SEC:
                bounced_back = False
                for t2, cx2, cy2 in ball_track[idx + 1:]:
                    if t2 - t > BOUNCE_BACK_CONFIRM_SEC:
                        break
                    if cx2 is None:
                        continue
                    if cy2 < hy1:
                        bounced_back = True
                        break
                if not bounced_back:
                    makes.append(round((t_above + t) / 2, 2))
                    last_make_t = t
                above_sighting = None

    return makes


def run_detection(video: Path, config_path: Path, output_path: Path,
                   model: str = "models/yolov8m.pt", device: str | None = None,
                   fps: float = TARGET_FPS, progress_cb=None) -> list[float]:
    """Run the full ball-detection -> trajectory pipeline for one video and
    write the result to output_path. Returns the list of detected make
    timestamps (seconds). progress_cb(timestamp_sec), if given, is called
    after each processed frame - callers (CLI, web job) render it however
    they like instead of this function assuming a terminal.
    """
    hoop_bbox_norm = load_config(config_path)

    import cv2

    cap = cv2.VideoCapture(str(video))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    roi = compute_roi(hoop_bbox_norm, frame_w, frame_h)
    rx1, ry1, rx2, ry2 = roi

    import torch
    from ultralytics import YOLO
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    yolo_model = YOLO(model)

    BATCH_SIZE = 8
    ball_track = []
    batch_frames, batch_times = [], []

    def flush_batch():
        nonlocal batch_frames, batch_times
        if not batch_frames:
            return
        centers = detect_ball_centers_batch(yolo_model, batch_frames, device=device,
                                             roi_offset=(rx1, ry1), full_size=(frame_w, frame_h))
        for tt, c in zip(batch_times, centers):
            ball_track.append((tt, c[0], c[1]) if c is not None else (tt, None, None))
        batch_frames, batch_times = [], []

    for t, frame in iter_sampled_frames(video, fps, TARGET_HEIGHT, roi=roi):
        batch_frames.append(frame)
        batch_times.append(t)
        if len(batch_frames) >= BATCH_SIZE:
            flush_batch()
        if progress_cb:
            progress_cb(t)
    flush_batch()

    makes = find_makes(ball_track, hoop_bbox_norm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"video": video.name, "makes_sec": makes}, indent=2))
    return makes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                         help="default: data/configs/<video_stem>.json")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: data/ground_truth/<video_stem>_detected.json")
    parser.add_argument("--model", type=str, default="models/yolov8m.pt")
    parser.add_argument("--device", type=str, default=None,
                         help="cpu/mps/cuda; default: mps if available else cpu")
    parser.add_argument("--fps", type=float, default=TARGET_FPS,
                         help=f"temporal sampling rate (default {TARGET_FPS})")
    args = parser.parse_args()

    config_path = args.config or Path("data/configs") / f"{args.video.stem}.json"
    output_path = args.output or Path("data/ground_truth") / f"{args.video.stem}_detected.json"

    hoop_bbox_norm = load_config(config_path)
    import cv2
    cap = cv2.VideoCapture(str(args.video))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    rx1, ry1, rx2, ry2 = compute_roi(hoop_bbox_norm, frame_w, frame_h)
    print(f"cropping detection input to {rx2-rx1}x{ry2-ry1} "
          f"(from {frame_w}x{frame_h}) based on hoop calibration")

    last_report = time.time()

    def progress_cb(t):
        nonlocal last_report
        if time.time() - last_report > 60:
            print(f"progress: {t:.1f}s of video processed", flush=True)
            last_report = time.time()

    makes = run_detection(args.video, config_path, output_path, model=args.model,
                           device=args.device, fps=args.fps, progress_cb=progress_cb)

    print(f"detected {len(makes)} makes -> {output_path}")
    for m in makes:
        print(f"  {m:.2f}s")


if __name__ == "__main__":
    main()
