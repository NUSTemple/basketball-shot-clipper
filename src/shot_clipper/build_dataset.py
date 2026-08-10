"""Materialize data/dataset/labels.json into a goal/no_goal folder layout.

Reads the labels written by shot_clipper.label_ui.app and symlinks (default)
or copies each labeled clip into data/dataset/goal/ or data/dataset/no_goal/,
flattening the "<video>/shot_NNN.mp4" clip path into "<video>__shot_NNN.mp4"
so files from different source videos don't collide. Goal clips that have
been star-rated get the rating in their filename too, e.g.
"<video>__shot_NNN_4star.mp4" - handy for sorting/filtering in a video editor.
Clips tagged with a scorer (Review panel's scorer picker) get the name
appended too, e.g. "<video>__shot_NNN_4star_Alice.mp4".

Usage:
    shot-clipper-build-dataset [--clips-dir PATH] [--copy] [--min-stars N]
        [--group-by-stars] [--player NAME]

--min-stars only affects the goal/ folder - e.g. --min-stars 4 gives you
just your best-rated highlights to pull into a video, leaving lower-rated
and unrated goals out. no_goal is always exported in full.

--player only affects the goal/ folder too - e.g. --player Alice gives you
just Alice's makes, for a per-player highlight reel.

--group-by-stars puts goal clips into goal/5star/, goal/4star/, ...,
goal/unrated/ subfolders instead of one flat folder. Video editors like
CapCut have no concept of custom clip metadata/ratings, so this - plus the
filename suffix - is how a star rating actually carries over on import:
folders typically become bins in the editor's media panel, letting you see
and pick your best-rated clips without re-reviewing every one.

--clips-dir defaults to $SHOT_CLIPPER_CLIPS_DIR if set, otherwise a
placeholder that must be overridden explicitly.
"""
import argparse
import os
import shutil
from pathlib import Path

from . import paths
from .dataset_labels import labels_path, load_labels
from .roster import scorer_slug

DEFAULT_CLIPS_DIR = Path(os.environ.get("SHOT_CLIPPER_CLIPS_DIR", "clips"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
                         help=f"default: {DEFAULT_CLIPS_DIR} (or $SHOT_CLIPPER_CLIPS_DIR)")
    parser.add_argument("--copy", action="store_true",
                         help="copy clip files instead of symlinking (uses more disk)")
    parser.add_argument("--min-stars", type=int, default=0, choices=range(0, 6),
                         help="only include goal clips rated >= this many stars (0 = "
                              "include every goal clip, rated or not)")
    parser.add_argument("--group-by-stars", action="store_true",
                         help="put goal clips into goal/<N>star/ subfolders instead of one "
                              "flat folder - see module docstring")
    parser.add_argument("--player", type=str, default=None,
                         help="only include goal clips tagged with this scorer (exact match, "
                              "see the Review panel's scorer picker) - no_goal is unaffected")
    args = parser.parse_args()

    labels = load_labels()
    if not labels:
        raise SystemExit(f"no labels found in {labels_path()} - "
                          "label some clips with shot-clipper-label-ui first")

    dataset_dir = paths.dataset_dir()
    out_dirs = {"goal": dataset_dir / "goal", "no_goal": dataset_dir / "no_goal"}
    for d in out_dirs.values():
        # fully derived from labels.json - clean rebuild each run (rmtree, not just
        # unlinking each entry, since --group-by-stars runs leave subfolders behind)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    counts = {"goal": 0, "no_goal": 0, "missing": 0, "below_min_stars": 0, "not_this_player": 0}
    stars_breakdown = {n: 0 for n in range(1, 6)}
    stars_breakdown["unrated"] = 0

    for clip_rel, entry in sorted(labels.items()):
        label = entry["label"]
        stars = entry.get("stars") if label == "goal" else None
        scorer = entry.get("scorer") if label == "goal" else None

        if label == "goal":
            stars_breakdown[stars if stars else "unrated"] += 1
            if args.min_stars and (stars is None or stars < args.min_stars):
                counts["below_min_stars"] += 1
                continue
            if args.player and scorer != args.player:
                counts["not_this_player"] += 1
                continue

        src = args.clips_dir / clip_rel
        if not src.is_file():
            print(f"skip (missing on disk): {clip_rel}")
            counts["missing"] += 1
            continue

        base = Path(clip_rel.replace("/", "__")).stem
        suffix = f"_{stars}star" if stars else ""
        suffix += f"_{scorer_slug(scorer)}" if scorer else ""
        flat_name = f"{base}{suffix}.mp4"
        out_dir = out_dirs[label]
        if label == "goal" and args.group_by_stars:
            out_dir = out_dir / (f"{stars}star" if stars else "unrated")
            out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / flat_name
        if args.copy:
            shutil.copy2(src, dest)
        else:
            dest.symlink_to(src.resolve())
        counts[label] += 1

    print(f"goal:     {counts['goal']} -> {out_dirs['goal']}")
    if args.min_stars:
        print(f"          ({counts['below_min_stars']} goal clips excluded: "
              f"unrated or below {args.min_stars} stars)")
    if args.player:
        print(f"          ({counts['not_this_player']} goal clips excluded: "
              f"not scored by {args.player!r})")
    print(f"no_goal:  {counts['no_goal']} -> {out_dirs['no_goal']}")
    if counts["missing"]:
        print(f"missing:  {counts['missing']} (labeled clips no longer on disk)")
    ratings = ", ".join(f"{n}★={stars_breakdown[n]}" for n in range(5, 0, -1))
    print(f"goal ratings: {ratings}, unrated={stars_breakdown['unrated']}")


if __name__ == "__main__":
    main()
