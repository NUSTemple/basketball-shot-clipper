"""Generate a static HTML contact sheet (thumbnail grid) for a folder of clips
- a quick visual sanity check before importing into a video editor, without
re-reviewing every clip in the label UI or scrubbing through Finder one file
at a time.

Usage:
    shot-clipper-contact-sheet <folder> [--output sheet.html] [--cols 6]
        [--at 5.0] [--recursive]

<folder> is typically a shot-clipper-build-dataset output, e.g.
data/dataset/goal/5star/ or, with --recursive, data/dataset/goal/ itself
(grouped into --group-by-stars subfolders -> one section per subfolder,
5star/4star/... ordered before anything else).

Thumbnails are extracted at --at seconds into each clip (default 5.0 -
clips are cut [t-5s, t+2s] around the shot, so 5.0 lands on the shot/make
moment) and written to <output>_thumbnails/ next to the HTML.
"""
import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAR_ORDER = {f"{n}star": n for n in range(5, 0, -1)}
STAR_SUFFIX_RE = re.compile(r"_(\d)star$")


def extract_thumbnail(clip_path: Path, out_path: Path, at: float, width: int = 320) -> bool:
    """Returns False (and leaves no file) if ffmpeg couldn't grab a frame at
    `at` seconds - most commonly because the clip is shorter than that (e.g.
    a custom --pre/--post export). Doesn't fall back to a different time:
    for a quick visual sanity check tool, a thumbnail from the wrong moment
    (e.g. before the shot happens) is misleading in a way a missing
    thumbnail isn't - render_html() skips clips with no thumbnail."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{at:.2f}", "-i", str(clip_path),
        "-frames:v", "1", "-q:v", "2", "-vf", f"scale={width}:-1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def group_key(clip_path: Path, root: Path):
    rel_parent = clip_path.parent.relative_to(root)
    return str(rel_parent) if str(rel_parent) != "." else ""


def sort_groups(groups: list[str]) -> list[str]:
    return sorted(groups, key=lambda g: (STAR_ORDER.get(g, -1) * -1, g))


def star_from_filename(clip_path: Path) -> int | None:
    m = STAR_SUFFIX_RE.search(clip_path.stem)
    return int(m.group(1)) if m else None


def render_html(groups: dict[str, list[tuple[Path, Path]]], cols: int, title: str) -> str:
    sections = []
    for group in sort_groups(list(groups.keys())):
        clips = groups[group]
        heading = f"<h2>{group} <span class='count'>({len(clips)})</span></h2>" if group else ""
        cells = []
        for clip_path, thumb_path in sorted(clips):
            stars = star_from_filename(clip_path)
            star_badge = f"<span class='stars'>{'★' * stars}</span>" if stars else ""
            cells.append(f"""
              <div class="cell">
                <img src="{thumb_path}" loading="lazy">
                <div class="name">{clip_path.name}</div>
                {star_badge}
              </div>""")
        sections.append(f"<section>{heading}<div class='grid'>{''.join(cells)}</div></section>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ margin: 0; padding: 24px; background: #14161a; color: #e8e8e8;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 18px; color: #aaa; font-weight: 600; }}
  h2 {{ font-size: 15px; color: #f5b301; border-bottom: 1px solid #2a2d33; padding-bottom: 6px; }}
  .count {{ color: #888; font-weight: 400; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 12px; margin: 12px 0 28px; }}
  .cell {{ background: #1b1d22; border-radius: 8px; overflow: hidden; position: relative; }}
  .cell img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }}
  .name {{ font-size: 11px; color: #999; padding: 6px 8px; word-break: break-all; }}
  .stars {{ position: absolute; top: 6px; right: 8px; color: #f5b301;
            text-shadow: 0 1px 3px rgba(0,0,0,0.8); font-size: 14px; }}
</style></head>
<body>
<h1>{title}</h1>
{''.join(sections)}
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <folder>/contact_sheet.html")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--at", type=float, default=5.0,
                         help="seconds into each clip to grab the thumbnail (default 5.0, "
                              "matching the shot/make moment in a default clip_shots.py cut)")
    parser.add_argument("--recursive", action="store_true",
                         help="include clips in subfolders too, grouped into one section per "
                              "subfolder (e.g. a shot-clipper-build-dataset --group-by-stars output)")
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"not a folder: {args.folder}")

    clips = sorted(args.folder.rglob("*.mp4") if args.recursive else args.folder.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 clips found in {args.folder}"
                          + ("" if args.recursive else " (pass --recursive to include subfolders)"))

    output = args.output or args.folder / "contact_sheet.html"
    thumb_dir = output.with_name(output.stem + "_thumbnails")

    def do_one(clip_path):
        thumb_path = thumb_dir / f"{clip_path.stem}.jpg"
        ok = extract_thumbnail(clip_path, thumb_path, args.at)
        return clip_path, (thumb_path if ok else None)

    print(f"extracting {len(clips)} thumbnails...")
    pairs = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for clip_path, thumb_path in pool.map(do_one, clips):
            if thumb_path is None:
                print(f"  skip (no frame at {args.at:.1f}s - clip may be shorter): {clip_path}")
                n_failed += 1
                continue
            pairs.append((clip_path, thumb_path))

    groups: dict[str, list[tuple[Path, Path]]] = {}
    for clip_path, thumb_path in pairs:
        key = group_key(clip_path, args.folder)
        groups.setdefault(key, []).append(
            (clip_path.relative_to(args.folder), thumb_path.relative_to(output.parent)))

    html = render_html(groups, args.cols, title=f"Contact sheet: {args.folder}")
    output.write_text(html)
    print(f"{len(pairs)} clips -> {output}"
          + (f" ({n_failed} skipped)" if n_failed else ""))


if __name__ == "__main__":
    main()
