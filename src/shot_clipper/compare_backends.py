"""Measure the ONNX backend against the torch one on real footage.

The point of the ONNX path is a ~350MB install that runs on any GPU instead
of a ~3GB NVIDIA-only one (docs/GPU_SETUP.md). That's only worth having if it
detects the same balls, and "looks about right" is not a standard this
pipeline can be checked against by eye: a subtle preprocessing mismatch
degrades ball recall while every intermediate value still looks plausible.

So this decodes each frame ONCE and feeds the identical pixels to both
backends. Any difference in the output is therefore inference, not decode,
sampling, or crop geometry - the three things that would otherwise be
confounded with it.

Three levels of comparison, weakest signal last:

  per-frame detection agreement  - did both see a ball at all
  centre distance                - when both did, how far apart
  resulting makes                - what find_makes() does with each track

Makes are reported last on purpose. They're the number that matters, but
they're also the least sensitive: a geometric rule can absorb a lot of
per-frame disagreement, or amplify a little of it, so a matching make count
is weak evidence on its own.

Usage:
    shot-clipper-compare-backends <video> [--max-frames 600] [--json out.json]
"""
import argparse
import json
import time
from pathlib import Path

from . import detect_shots, inference, validate, video_source
from .device_config import get_device, get_optimal_batch_size

BACKENDS = ("torch", "onnx")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[idx]


def compare(video: Path, config_path: Path, fps: float, max_frames: int | None,
            device: str | None = None) -> dict:
    import cv2

    hoop_bbox_norm = detect_shots.load_config(config_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    roi = detect_shots.compute_roi(hoop_bbox_norm, frame_w, frame_h)
    rx1, ry1, rx2, ry2 = roi

    device = device or get_device()
    detectors = {}
    for name in BACKENDS:
        detectors[name] = inference.load_detector(device=device, backend=name)
        print(f"{name:>6}: {detectors[name].describe()}")
    print(f"  roi: {rx2 - rx1}x{ry2 - ry1} of {frame_w}x{frame_h}, "
          f"imgsz {((max(rx2 - rx1, ry2 - ry1)) + 31) // 32 * 32}")
    print(f"decode: {video_source.describe()}")

    batch_size = get_optimal_batch_size(device)
    tracks = {name: [] for name in BACKENDS}
    elapsed = {name: 0.0 for name in BACKENDS}
    batch_frames, batch_times = [], []
    n_frames = 0

    def flush():
        if not batch_frames:
            return
        for name, detector in detectors.items():
            start = time.perf_counter()
            centers = detect_shots.detect_ball_centers_batch(
                detector, batch_frames, device=device,
                roi_offset=(rx1, ry1), full_size=(frame_w, frame_h))
            elapsed[name] += time.perf_counter() - start
            for t, c in zip(batch_times, centers):
                tracks[name].append((t, c[0], c[1]) if c is not None else (t, None, None))
        batch_frames.clear()
        batch_times.clear()

    for t, frame in detect_shots.iter_sampled_frames(video, fps, detect_shots.TARGET_HEIGHT, roi=roi):
        batch_frames.append(frame)
        batch_times.append(t)
        n_frames += 1
        if len(batch_frames) >= batch_size:
            flush()
            print(f"\r  {n_frames} frames...", end="", flush=True)
        if max_frames and n_frames >= max_frames:
            break
    flush()
    print(f"\r  {n_frames} frames scanned by both backends")

    return _summarize(tracks, elapsed, hoop_bbox_norm, n_frames,
                      frame_size=(frame_w, frame_h), video=video)


def _summarize(tracks, elapsed, hoop_bbox_norm, n_frames, frame_size, video: Path) -> dict:
    ref, alt = tracks["torch"], tracks["onnx"]
    frame_w, frame_h = frame_size

    both = only_ref = only_alt = neither = 0
    distances_px = []
    for (_, rx, ry), (_, ax, ay) in zip(ref, alt):
        if rx is not None and ax is not None:
            both += 1
            # centres are normalized against the full source frame, so scaling
            # by the frame size gives source pixels - the unit a person can
            # actually judge ("under a ball width" vs "across the rim")
            dx = (rx - ax) * frame_w
            dy = (ry - ay) * frame_h
            distances_px.append((dx * dx + dy * dy) ** 0.5)
        elif rx is not None:
            only_ref += 1
        elif ax is not None:
            only_alt += 1
        else:
            neither += 1

    makes = {name: detect_shots.find_makes(track, hoop_bbox_norm)
             for name, track in tracks.items()}
    matched, extra, missed = validate.match(makes["onnx"], makes["torch"], tolerance=1.0)

    return {
        "video": video.name,
        "n_frames": n_frames,
        "detections": {name: sum(1 for _, cx, _ in track if cx is not None)
                       for name, track in tracks.items()},
        "agreement": {"both": both, "only_torch": only_ref,
                      "only_onnx": only_alt, "neither": neither},
        "centre_distance_px": {
            "median": _percentile(distances_px, 0.5),
            "p95": _percentile(distances_px, 0.95),
            "max": max(distances_px) if distances_px else float("nan"),
        },
        "makes": {name: [round(m, 2) for m in ms] for name, ms in makes.items()},
        "makes_agreement": {"matched": len(matched), "onnx_only": len(extra),
                            "torch_only": len(missed)},
        "seconds": dict(elapsed),
    }


def print_report(report: dict) -> None:
    det = report["detections"]
    agree = report["agreement"]
    dist = report["centre_distance_px"]
    n = report["n_frames"] or 1

    print()
    print("=" * 64)
    print(f"{report['video']}: {report['n_frames']} sampled frames")
    print()
    print("ball detected per frame")
    for name in BACKENDS:
        print(f"  {name:<6} {det[name]:>6}  ({det[name] / n:.1%} of frames)")
    print(f"  both   {agree['both']:>6}")
    print(f"  torch only {agree['only_torch']:>2}, onnx only {agree['only_onnx']:>2}, "
          f"neither {agree['neither']}")
    print()
    print("centre distance where both detected (source pixels)")
    print(f"  median {dist['median']:.2f}   p95 {dist['p95']:.2f}   max {dist['max']:.2f}")
    print()
    print("makes from find_makes()")
    for name in BACKENDS:
        print(f"  {name:<6} {len(report['makes'][name]):>3}  {report['makes'][name]}")
    ma = report["makes_agreement"]
    print(f"  matched {ma['matched']}, onnx-only {ma['onnx_only']}, torch-only {ma['torch_only']}")
    print()
    print("inference wall time (decode excluded, shared)")
    for name in BACKENDS:
        secs = report["seconds"][name]
        print(f"  {name:<6} {secs:>7.1f}s  ({secs / n * 1000:.1f}ms/frame)")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None,
                         help="default: data/configs/<video_stem>.json")
    parser.add_argument("--fps", type=float, default=detect_shots.TARGET_FPS)
    parser.add_argument("--max-frames", type=int, default=None,
                         help="stop after this many sampled frames (a whole video is slow, "
                              "and disagreement shows up long before the end)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None,
                         help="also write the full report here")
    args = parser.parse_args()

    config_path = args.config or Path("data/configs") / f"{args.video.stem}.json"
    if not config_path.is_file():
        raise SystemExit(f"no hoop calibration at {config_path} - run shot-clipper-calibrate first")

    report = compare(args.video, config_path, args.fps, args.max_frames, device=args.device)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"report -> {args.json}")


if __name__ == "__main__":
    main()
