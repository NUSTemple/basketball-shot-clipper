"""Cut independent clips around each detected/confirmed shot timestamp.

Reads full original-resolution/framerate video (decision 6/7: independent
files, not a merged highlight reel; 5s before / 2s after each make).

Usage:
    shot-clipper-clip <video_path> <timestamps_json> [--pre 5] [--post 2]
        [--outdir clips]

Run from the repo root (or pass --outdir) - default output is ./clips/<video_stem>/.

<timestamps_json> is either the output of detect_shots.py
({"makes_sec": [...]}) or a plain JSON list of seconds.
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
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


def cut_all(video_path: Path, timestamps: list[float], outdir: Path,
            pre: float = 5.0, post: float = 2.0, progress_cb=None) -> list[tuple[int, float, Path]]:
    """Cut one clip per timestamp into outdir/shot_NNN.mp4. Returns
    (index, timestamp, out_path) tuples in completion order. progress_cb(i, total),
    if given, is called after each clip finishes.
    """
    duration = pre + post

    def do_one(item):
        i, t = item
        start = t - pre
        out_path = outdir / f"shot_{i:03d}.mp4"
        cut_clip(video_path, start, duration, out_path)
        return i, t, out_path

    results = []
    # ffmpeg cuts don't touch the GPU and barely touch each other's I/O, so
    # cutting several in parallel is a straightforward, zero-risk speedup
    # over doing them one at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, t, out_path in pool.map(do_one, enumerate(timestamps, start=1)):
            results.append((i, t, out_path))
            if progress_cb:
                progress_cb(i, len(timestamps))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("timestamps", type=Path)
    parser.add_argument("--pre", type=float, default=5.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir or Path("clips") / args.video.stem
    timestamps = load_timestamps(args.timestamps)
    duration = args.pre + args.post

    results = cut_all(args.video, timestamps, outdir, args.pre, args.post)
    for i, t, out_path in sorted(results):
        start = max(0.0, t - args.pre)
        print(f"[{i}/{len(timestamps)}] make@{t:.2f}s -> {out_path} "
              f"({start:.2f}s .. +{duration:.2f}s)")

    print(f"done: {len(timestamps)} clips in {outdir}")


if __name__ == "__main__":
    main()
