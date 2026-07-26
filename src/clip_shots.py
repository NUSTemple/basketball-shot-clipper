"""Cut independent clips around each detected/confirmed shot timestamp.

Reads full original-resolution/framerate video (decision 6/7: independent
files, not a merged highlight reel; 5s before / 2s after each make).

Usage:
    python src/clip_shots.py <video_path> <timestamps_json> [--pre 5] [--post 2]
        [--outdir clips]

<timestamps_json> is either the output of detect_shots.py
({"makes_sec": [...]}) or a plain JSON list of seconds.
"""
import argparse
import json
import subprocess
from pathlib import Path


def load_timestamps(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data["makes_sec"]
    return data


def cut_clip(video_path: Path, start: float, duration: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start):.2f}",
        "-i", str(video_path),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("timestamps", type=Path)
    parser.add_argument("--pre", type=float, default=5.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    outdir = args.outdir or root / "clips" / args.video.stem

    timestamps = load_timestamps(args.timestamps)
    duration = args.pre + args.post

    for i, t in enumerate(timestamps, start=1):
        start = t - args.pre
        out_path = outdir / f"shot_{i:03d}.mp4"
        print(f"[{i}/{len(timestamps)}] make@{t:.2f}s -> {out_path} "
              f"({max(0.0, start):.2f}s .. +{duration:.2f}s)")
        cut_clip(args.video, start, duration, out_path)

    print(f"done: {len(timestamps)} clips in {outdir}")


if __name__ == "__main__":
    main()
