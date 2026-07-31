"""Calibrate the hoop region for a fixed-camera video.

Extracts a frame from the video, lets the user drag a rectangle around the
hoop with the mouse, and saves the region to data/configs/<video_stem>.json.

Usage:
    shot-clipper-calibrate <video_path> [--frame-index N]
    shot-clipper-calibrate <video_path> --bbox x1,y1,x2,y2   # non-interactive

The bbox is stored normalized (0-1) against the source frame size, so the
same config works regardless of any later resizing/downsampling. Run from
the repo root so the default output path (data/configs/) resolves correctly.
"""
import argparse
import json
from pathlib import Path

import cv2

WINDOW = "calibrate hoop - drag a box around the hoop, 's' save, 'r' reset, 'q' quit"


def config_path_for(video_path: Path) -> Path:
    return Path("data/configs") / f"{video_path.stem}.json"


def extract_frame(video_path: Path, frame_index: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index is None:
        frame_index = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_index} from {video_path}")
    return frame, frame_index, width, height


def save_config(video_path: Path, frame_index: int, width: int, height: int, bbox_px):
    x1, y1, x2, y2 = bbox_px
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cfg = {
        "video": video_path.name,
        "frame_index": frame_index,
        "frame_width": width,
        "frame_height": height,
        "hoop_bbox_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
    }
    out_path = config_path_for(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"saved {out_path}")
    return out_path


def interactive_calibrate(video_path: Path, frame_index: int | None):
    frame, frame_index, width, height = extract_frame(video_path, frame_index)

    state = {"drawing": False, "start": None, "box": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["box"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["box"] = (*state["start"], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["box"] = (*state["start"], x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        display = frame.copy()
        if state["box"] is not None:
            x1, y1, x2, y2 = state["box"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("calibration cancelled")
        if key == ord("r"):
            state["box"] = None
        if key == ord("s"):
            if state["box"] is None:
                print("no box drawn yet")
                continue
            cv2.destroyAllWindows()
            return save_config(video_path, frame_index, width, height, state["box"])


def noninteractive_calibrate(video_path: Path, frame_index: int | None, bbox_str: str):
    _frame, frame_index, width, height = extract_frame(video_path, frame_index)
    x1, y1, x2, y2 = (int(v) for v in bbox_str.split(","))
    return save_config(video_path, frame_index, width, height, (x1, y1, x2, y2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame-index", type=int, default=None,
                         help="frame to calibrate on (default: middle frame)")
    parser.add_argument("--bbox", type=str, default=None,
                         help="x1,y1,x2,y2 in pixel coords; skips the GUI")
    args = parser.parse_args()

    if args.bbox:
        noninteractive_calibrate(args.video, args.frame_index, args.bbox)
    else:
        interactive_calibrate(args.video, args.frame_index)


if __name__ == "__main__":
    main()
