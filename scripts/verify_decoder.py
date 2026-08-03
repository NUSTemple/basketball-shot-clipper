#!/usr/bin/env python3
"""Prove hardware decode gives the detector the same pixels software decode did.

Hardware decode is enabled automatically wherever ffmpeg reports it, which
means a difference in how the decoder tags or converts colour would silently
move detections rather than fail loudly - the frames would still be the right
size and the histogram would still look plausible. This is the check that
catches that, and it is the reason to run it before trusting a new platform.

The bar it enforces is the one NVDEC already cleared on Windows: hardware vs
ffmpeg-CPU must be bit-identical, every frame, max difference 0.

    python scripts/verify_decoder.py <video> [--config data/configs/<stem>.json]
                                    [--frames 40]

Exits non-zero if hardware and software decode disagree.
"""
import argparse
import json
import itertools
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shot_clipper import detect_shots as ds  # noqa: E402
from shot_clipper import video_source  # noqa: E402
from shot_clipper.device_config import device_summary, get_device  # noqa: E402


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"<failed: {e}>"


def environment():
    print("=" * 74)
    print("ENVIRONMENT")
    print("=" * 74)
    ver = run(["ffmpeg", "-version"]).splitlines()
    print(f"ffmpeg      : {ver[0] if ver else '<not found>'}")
    print(f"machine     : {platform.machine()} / {platform.platform()}")
    accels = run(["ffmpeg", "-v", "quiet", "-hwaccels"]).split()
    print(f"hwaccels    : {' '.join(a for a in accels if a != 'methods:') or '<none>'}")
    try:
        import torch
        mps = getattr(torch.backends, "mps", None)
        print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}  "
              f"mps={mps.is_available() if mps else 'n/a'}")
    except ImportError:
        print("torch       : <not installed>")
    print(f"device      : {device_summary(get_device())}")
    print(f"decoder     : {video_source.describe()}  (auto-selected)")


def source_format(video):
    """Colour tagging and bit depth are exactly where a hardware/software
    mismatch would show up, so record them before comparing anything."""
    print()
    print("=" * 74)
    print("SOURCE FORMAT")
    print("=" * 74)
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries",
               "stream=codec_name,pix_fmt,width,height,color_range,color_space,"
               "color_primaries,color_transfer",
               "-of", "json", str(video)])
    try:
        stream = json.loads(out)["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        print(f"  <could not probe: {out[:200]}>")
        return {}
    for k, v in stream.items():
        print(f"  {k:18s} {v}")
    if "10" in str(stream.get("pix_fmt", "")):
        print("  NOTE: 10-bit source. VideoToolbox emits p010le where software emits")
        print("        yuv420p10le; these reach swscale by different paths, so this is")
        print("        the case most likely to differ. Good - this is what to test on.")
    return stream


def collect(video, roi, frame_w, frame_h, step, src_fps, decoder, n):
    os.environ[video_source.DECODER_ENV] = decoder
    # Without this the lru_cache hands every leg the FIRST leg's answer and the
    # whole comparison silently passes against itself.
    video_source.decoder_kind.cache_clear()
    resolved = video_source.decoder_kind()

    t0 = time.perf_counter()
    if resolved is None:
        frames = [f for _, f in itertools.islice(
            ds.iter_sampled_frames(video, src_fps / step, None, roi=roi), n)]
    else:
        frames = [f for _, f in itertools.islice(
            video_source.iter_frames(video, step, src_fps, roi, frame_w, frame_h), n)]
    elapsed = time.perf_counter() - t0
    return resolved, frames, elapsed


def compare(a, b, label):
    if len(a) != len(b):
        print(f"  {label:26s} FRAME COUNT DIFFERS: {len(a)} vs {len(b)}")
        return False
    diffs = [np.abs(x.astype(np.int16) - y.astype(np.int16)) for x, y in zip(a, b)]
    identical = sum(1 for d in diffs if d.max() == 0)
    mx = max(int(d.max()) for d in diffs)
    mean = sum(float(d.mean()) for d in diffs) / len(diffs)
    print(f"  {label:26s} {identical}/{len(a)} identical   max={mx:<4d} mean={mean:.4f}")
    if mx:
        for i, d in enumerate(diffs):
            if d.max():
                y, x, c = np.unravel_index(int(d.argmax()), d.shape)
                print(f"    first differing frame {i} at (y={y}, x={x}, ch={c}): "
                      f"{a[i][y, x]} vs {b[i][y, x]}")
                break
    return mx == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", type=Path, default=None,
                    help="default: data/configs/<video stem>.json")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--fps", type=float, default=ds.TARGET_FPS)
    args = ap.parse_args()

    environment()
    source_format(args.video)

    config = args.config or Path("data/configs") / f"{args.video.stem}.json"
    if not config.is_file():
        sys.exit(f"\nno calibration at {config} - pass --config")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    # the production ROI, so we compare the pixels detection actually sees
    roi = ds.compute_roi(ds.load_config(config), frame_w, frame_h)
    step = max(1, round(src_fps / args.fps))

    print()
    print("=" * 74)
    print(f"DECODING {args.frames} FRAMES  (ROI {roi[2]-roi[0]}x{roi[3]-roi[1]}, "
          f"every {step} source frame{'s' if step > 1 else ''})")
    print("=" * 74)

    hw = "videotoolbox" if sys.platform == "darwin" else "nvdec"
    results = {}
    for decoder in (hw, "cpu", "opencv"):
        try:
            resolved, frames, elapsed = collect(
                args.video, roi, frame_w, frame_h, step, src_fps, decoder, args.frames)
        except Exception as e:  # noqa: BLE001 - report, don't traceback
            print(f"  {decoder:14s} FAILED: {type(e).__name__}: {e}")
            continue
        results[decoder] = frames
        secs = len(frames) * step / src_fps
        print(f"  {decoder:14s} -> {resolved or 'opencv'}: {len(frames)} frames in "
              f"{elapsed:5.2f}s  ({elapsed/max(1,len(frames))*1000:5.1f} ms/frame, "
              f"{secs/elapsed:4.1f}x realtime)")
    os.environ.pop(video_source.DECODER_ENV, None)
    video_source.decoder_kind.cache_clear()

    print()
    print("=" * 74)
    print("PIXEL COMPARISON")
    print("=" * 74)
    ok = True
    if hw in results and "cpu" in results:
        ok = compare(results[hw], results["cpu"], f"{hw} vs ffmpeg-cpu")
    else:
        print("  cannot compare - a decoder leg failed above")
        ok = False
    if "opencv" in results and "cpu" in results:
        compare(results["cpu"], results["opencv"], "ffmpeg-cpu vs opencv")
        print("    (this pair is EXPECTED to differ - different YUV->BGR conversion.")
        print("     On Windows it was max 12, mean 0.62. Shown for comparability only.)")

    print()
    print("=" * 74)
    if ok:
        print(f"PASS - {hw} is bit-identical to software decode.")
        print()
        print("Next, the end-to-end gate (makes_sec must match exactly):")
        print(f"  SHOT_CLIPPER_DECODER=cpu {' ' * len(hw)} shot-clipper-detect {args.video} "
              f"--output /tmp/a.json")
        print(f"  SHOT_CLIPPER_DECODER={hw} shot-clipper-detect {args.video} "
              f"--output /tmp/b.json")
        print("  diff /tmp/a.json /tmp/b.json")
    else:
        print(f"FAIL - {hw} does NOT match software decode.")
        print("Detection output would shift. Roll back with SHOT_CLIPPER_DECODER=cpu")
        print("(ffmpeg, no hwaccel) or =opencv (the original cv2 path), and report the")
        print("colour tagging printed under SOURCE FORMAT above.")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
