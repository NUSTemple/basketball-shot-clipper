"""Move per-video calibrations and detection results next to their videos.

These used to live in the checkout, at data/configs/<stem>.json and
data/ground_truth/<stem>_detected.json - one pair per video anyone ever
processed, accumulating in (and getting committed to) the repo. They now
belong beside the footage they describe; see paths.VIDEO_DATA_DIRNAME.

Nothing has to be migrated for the app to keep working: paths.find_config
and paths.find_ground_truth still read the old locations. This just drains
them.

A video's stem is all the old layout recorded, so the source path is
recovered from the job history in data/jobs/*.json. Stems with no job
record are left alone and reported - there is nowhere to put them.

Usage:
    python scripts/migrate_video_data.py            # dry run, prints a plan
    python scripts/migrate_video_data.py --apply    # copy the files
    python scripts/migrate_video_data.py --apply --delete-source
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shot_clipper import paths  # noqa: E402


def source_videos_by_stem() -> dict[str, Path]:
    """stem -> the source video it came from, per the job history.

    A stem that shows up with two different paths is skipped rather than
    guessed at: picking the wrong one would attach a hoop box to the wrong
    footage, which is worse than leaving the file where it is.
    """
    seen: dict[str, set[str]] = {}
    for job_file in glob.glob(str(paths.jobs_dir() / "*.json")):
        try:
            job = json.loads(Path(job_file).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("video", "queue"):
            value = job.get(key)
            for entry in ([value] if isinstance(value, str) else (value or [])):
                seen.setdefault(Path(entry).stem, set()).add(entry)

    resolved = {}
    for stem, candidates in seen.items():
        live = [Path(c) for c in candidates if Path(c).is_file()]
        if len(live) == 1:
            resolved[stem] = live[0]
    return resolved


def plan() -> tuple[list[tuple[Path, Path]], list[Path]]:
    """(moves, orphans) - what can be relocated, and what has no home."""
    by_stem = source_videos_by_stem()
    moves, orphans = [], []

    legacy = [(Path(p), Path(p).stem, paths.config_path_for)
              for p in glob.glob(str(paths.configs_dir() / "*.json"))]
    legacy += [(Path(p), Path(p).stem.removesuffix("_detected"),
                paths.ground_truth_path_for)
               for p in glob.glob(str(paths.ground_truth_dir() / "*_detected.json"))]

    for old_path, stem, destination_for in sorted(legacy):
        video = by_stem.get(stem)
        if video is None:
            orphans.append(old_path)
            continue
        moves.append((old_path, destination_for(video)))
    return moves, orphans


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="actually write; without it this only prints the plan")
    parser.add_argument("--delete-source", action="store_true",
                        help="remove the old file once copied (default: leave it)")
    args = parser.parse_args()

    moves, orphans = plan()
    for old_path, new_path in moves:
        status = "skip (already there)" if new_path.is_file() else "copy"
        print(f"{status}: {old_path} -> {new_path}")
    for orphan in orphans:
        print(f"leave (no source video on record): {orphan}")

    print(f"\n{len(moves)} to relocate, {len(orphans)} staying put")
    if not args.apply:
        print("dry run - pass --apply to write")
        return

    copied = deleted = 0
    for old_path, new_path in moves:
        if not new_path.is_file():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            copied += 1
        if args.delete_source:
            old_path.unlink()
            deleted += 1
    print(f"copied {copied}, deleted {deleted} originals")


if __name__ == "__main__":
    main()
