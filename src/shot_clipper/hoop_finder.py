"""Propose a hoop box for a new video, from hoops already calibrated.

Calibration is the one step that cannot be automated away by better code
alone - somebody has to say where the hoop is. But it does not have to be
drawn from scratch every time, and there is a good reason not to leave it
hand-drawn: measured across one session where the camera provably did not
move (left edge stable within 8px over six videos), the boxes drawn by hand
for the *same physical hoop* varied 38% in width and 33% in height. Every
threshold in find_makes scales off that box, so that variance silently
changes what counts as a make, video to video, for no reason in the
footage.

Matching a previous hoop into a new frame fixes both halves: the position
comes from where the hoop actually is, and the size is inherited from the
template rather than redrawn, so a run of videos gets one consistent box.

Why template matching rather than a trained detector: the camera is fixed
during a game but moves between them, and the venue changes between
sessions. Matching handles a moved camera on a known hoop, which is the
common case, needs no training, and degrades honestly - a low score means
"draw it yourself" rather than a confident wrong answer. A detector trained
on the accumulated boxes is the answer for a hoop never seen before.
"""
import json
from pathlib import Path

from . import external, paths

TEMPLATES_DIRNAME = "hoop_templates"

# How much context around the rim goes into the template. The rim alone is
# thin, low-contrast and looks like plenty of other gym ironwork; the
# backboard around it is what makes a match unambiguous.
CONTEXT_FRAC = 1.6
# Scales searched, to absorb the camera being closer or further after a
# move. Wider than it looks: 0.6-1.6 covers a hoop appearing at roughly a
# third to two and a half times its previous area.
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
# Below this correlation score the match is not worth showing. Chosen to be
# clearly separated from the scores seen matching a hoop into an unrelated
# gym; a caller should treat anything under it as "no suggestion".
MIN_SCORE = 0.55


def templates_dir() -> Path:
    return paths.data_dir() / TEMPLATES_DIRNAME


def template_path(video_stem: str) -> Path:
    return templates_dir() / f"{video_stem}.png"


def meta_path(video_stem: str) -> Path:
    return templates_dir() / f"{video_stem}.json"


def grab_frame(video: Path, at: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [external.ffmpeg_exe(), "-y", "-ss", f"{at:.2f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out_path)]
    external.run(cmd, check=True, capture_output=True)
    return out_path


def context_box(hoop_bbox_norm, frame_w: int, frame_h: int):
    """Pixel crop around the hoop, and where the hoop sits inside it."""
    hx1, hy1, hx2, hy2 = hoop_bbox_norm
    hw, hh = hx2 - hx1, hy2 - hy1
    x1 = max(0, round((hx1 - hw * CONTEXT_FRAC) * frame_w))
    x2 = min(frame_w, round((hx2 + hw * CONTEXT_FRAC) * frame_w))
    y1 = max(0, round((hy1 - hh * CONTEXT_FRAC) * frame_h))
    y2 = min(frame_h, round((hy2 + hh * CONTEXT_FRAC) * frame_h))
    inner = (round(hx1 * frame_w) - x1, round(hy1 * frame_h) - y1,
             round(hx2 * frame_w) - x1, round(hy2 * frame_h) - y1)
    return (x1, y1, x2, y2), inner


def build_template(video: Path, hoop_bbox_norm, at: float = 30.0) -> Path | None:
    """Save a matchable crop of this video's calibrated hoop."""
    import cv2

    stem = video.stem
    frame_file = templates_dir() / f"{stem}.frame.jpg"
    try:
        grab_frame(video, at, frame_file)
        img = cv2.imread(str(frame_file))
    finally:
        frame_file.unlink(missing_ok=True)
    if img is None:
        return None

    h, w = img.shape[:2]
    (x1, y1, x2, y2), inner = context_box(hoop_bbox_norm, w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    template_path(stem).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(template_path(stem)), crop)
    meta_path(stem).write_text(json.dumps({
        "video": video.name, "source_video": str(video),
        "hoop_bbox_norm": list(hoop_bbox_norm),
        "template_size": [crop.shape[1], crop.shape[0]],
        "inner": list(inner),
        "frame_size": [w, h],
    }))
    return template_path(stem)


def available_templates() -> list[dict]:
    out = []
    for meta_file in sorted(templates_dir().glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        image = template_path(meta_file.stem)
        if image.is_file():
            meta["image"] = str(image)
            meta["stem"] = meta_file.stem
            out.append(meta)
    return out


def _match_one(frame, template, meta):
    """Best (score, hoop_box_norm, scale) for one template over all scales."""
    import cv2

    fh, fw = frame.shape[:2]
    th, tw = template.shape[:2]
    ix1, iy1, ix2, iy2 = meta["inner"]

    best = None
    for scale in SCALES:
        sw, sh = int(round(tw * scale)), int(round(th * scale))
        if sw < 16 or sh < 16 or sw > fw or sh > fh:
            continue
        resized = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if best is not None and score <= best[0]:
            continue
        box = ((loc[0] + ix1 * scale) / fw, (loc[1] + iy1 * scale) / fh,
               (loc[0] + ix2 * scale) / fw, (loc[1] + iy2 * scale) / fh)
        best = (float(score), [round(v, 5) for v in box], scale)
    return best


def suggest(video: Path, at: float = 30.0, exclude_stem: str | None = None) -> dict | None:
    """Best hoop box for `video` from the template library, or None.

    Returns {"hoop_bbox_norm", "score", "scale", "from"} - `from` naming the
    video the box was inherited from, so a caller can say whose hoop this
    is rather than presenting a box out of nowhere.
    """
    import cv2

    templates = [t for t in available_templates() if t["stem"] != exclude_stem]
    if not templates:
        return None

    frame_file = templates_dir() / f"_probe_{video.stem}.jpg"
    try:
        grab_frame(video, at, frame_file)
        frame = cv2.imread(str(frame_file))
    finally:
        frame_file.unlink(missing_ok=True)
    if frame is None:
        return None

    best = None
    for meta in templates:
        template = cv2.imread(meta["image"])
        if template is None:
            continue
        found = _match_one(frame, template, meta)
        if found and (best is None or found[0] > best[0]):
            best = (found[0], found[1], found[2], meta)

    if best is None or best[0] < MIN_SCORE:
        return None
    score, box, scale, meta = best
    return {"hoop_bbox_norm": box, "score": round(score, 4),
            "scale": scale, "from": meta.get("video", meta["stem"])}
