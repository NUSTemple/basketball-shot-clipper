"""Materialize data/dataset/labels.json into a goal/no_goal folder layout.

Reads the labels written by shot_clipper.label_ui.app and symlinks (default)
or copies each labeled clip into data/dataset/goal/ or data/dataset/no_goal/,
flattening the "<video>/shot_NNN.mp4" clip path into "<video>__shot_NNN.mp4"
so files from different source videos don't collide.

Usage:
    shot-clipper-build-dataset [--clips-dir PATH] [--copy]

--clips-dir defaults to $SHOT_CLIPPER_CLIPS_DIR if set, otherwise a
placeholder that must be overridden explicitly.
"""
import argparse
import os
import shutil
from pathlib import Path

from .dataset_labels import DATASET_DIR, labels_path, load_labels

DEFAULT_CLIPS_DIR = Path(os.environ.get("SHOT_CLIPPER_CLIPS_DIR", "clips"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
                         help=f"default: {DEFAULT_CLIPS_DIR} (or $SHOT_CLIPPER_CLIPS_DIR)")
    parser.add_argument("--copy", action="store_true",
                         help="copy clip files instead of symlinking (uses more disk)")
    args = parser.parse_args()

    labels = load_labels()
    if not labels:
        raise SystemExit(f"no labels found in {labels_path()} - "
                          "label some clips with shot-clipper-label-ui first")

    out_dirs = {"goal": DATASET_DIR / "goal", "no_goal": DATASET_DIR / "no_goal"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    counts = {"goal": 0, "no_goal": 0, "missing": 0}
    for clip_rel, entry in labels.items():
        label = entry["label"]
        src = args.clips_dir / clip_rel
        if not src.is_file():
            print(f"skip (missing on disk): {clip_rel}")
            counts["missing"] += 1
            continue
        flat_name = clip_rel.replace("/", "__")
        dest = out_dirs[label] / flat_name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if args.copy:
            shutil.copy2(src, dest)
        else:
            dest.symlink_to(src.resolve())
        counts[label] += 1

    print(f"goal:     {counts['goal']} -> {out_dirs['goal']}")
    print(f"no_goal:  {counts['no_goal']} -> {out_dirs['no_goal']}")
    if counts["missing"]:
        print(f"missing:  {counts['missing']} (labeled clips no longer on disk)")


if __name__ == "__main__":
    main()
