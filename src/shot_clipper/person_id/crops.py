"""The shooter cut out of a goal clip as a small JPEG - the Players tab's
thumbnails, since a whole-court frame doesn't show who anyone is.

shot-clipper-players saves one per goal next to the clips
(<video folder>/CROPS_DIRNAME/<shot>.jpg), so the hosted label UI only has
to send a file: cutting it there meant ffmpeg reading the clip off the GCS
mount, 1-6s each, and the Players tab sat mostly black. Import-light (no
torch) - the label UI uses it too.
"""
import subprocess
from pathlib import Path

CROPS_DIRNAME = "_player_crops"


def crop_path(clip: Path) -> Path:
    return clip.parent / CROPS_DIRNAME / f"{clip.stem}.jpg"


def crop_args(info: dict) -> tuple[float, tuple[float, float, float, float]] | None:
    """(t, (x, y, w, h)) of the padded shooter box from a _players.json
    entry, or None when no shooter was found."""
    try:
        t, (x1, y1, x2, y2) = float(info["t"]), [float(v) for v in info["box"]]
    except (ValueError, KeyError, TypeError):
        return None
    pad_x, pad_y = 0.25 * (x2 - x1), 0.1 * (y2 - y1)
    x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
    return t, (x1, y1, (x2 + pad_x) - x1, (y2 + pad_y) - y1)


def cut_crop(clip: Path, t: float, box: tuple[float, float, float, float], out: Path) -> bool:
    x, y, w, h = box
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1", "-q:v", "3",
         # to 8-bit before cropping: an exact odd-offset crop of 10-bit
         # video (DJI) decodes to magenta
         "-vf", f"format=yuv420p,crop={w:.0f}:{h:.0f}:{x:.0f}:{y:.0f},scale=-2:240", str(out)],
        capture_output=True)
    return result.returncode == 0 and out.is_file()


def save_crops(folder: Path, clips: dict) -> int:
    """Cut the missing crops for one video folder's _players.json clips.
    Returns how many were made."""
    made = 0
    for name, info in clips.items():
        clip, args = folder / name, crop_args(info)
        if args is None or not clip.is_file():
            continue
        out = crop_path(clip)
        if out.is_file() and out.stat().st_mtime >= clip.stat().st_mtime:
            continue
        made += cut_crop(clip, *args, out)
    return made
