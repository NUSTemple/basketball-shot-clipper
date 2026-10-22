"""Find who took the shot in a goal clip and crop them.

RF-DETR (OpenHoopVision) has a `scorer` class for a player in the shooting
motion. The shooter is the highest-scoring `scorer` box in the 2.8s before
the make, with a bonus when the ball is at its hands in that frame. Checked
against the owner's scorer tags this beat tracing the ball back from the rim
(the trace too often jumped to a defender next to the shooter).

That person is then followed through the window by box overlap, and crops
come from frames where nobody else overlaps them - a crop of two people
mixes their appearance.
"""
from dataclasses import dataclass, field

import numpy as np

PEOPLE = ("player", "holder", "scorer")
WINDOW_BEFORE, WINDOW_AFTER = 3.5, 0.3   # seconds around the make that are scanned
SHOT_BEFORE, SHOT_AFTER = 2.8, 0.15      # where the shooting motion itself can be
STEP_SEC = 1 / 30
MAX_CROPS = 12


@dataclass
class Shooter:
    box: list[float]          # at the anchor frame, pixels
    t: float                  # clip seconds of the anchor frame
    method: str               # "scorer+ball" | "scorer" | "holder"
    crops: list = field(default_factory=list, repr=False)


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ball_at_hands(box, balls) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    for b in balls:
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if box[0] - 0.15 * w <= bx <= box[2] + 0.15 * w and box[1] - 0.15 * h <= by <= box[1] + 0.6 * h:
            return True
    return False


def people(dets) -> list[tuple[str, float, list[float]]]:
    """One frame's person boxes with duplicates merged - the model often
    boxes one person as both `player` and `scorer`; the merged box keeps
    `scorer` (then `holder`) so that label isn't lost to a higher-confidence
    `player` box."""
    rank = {"scorer": 2, "holder": 1, "player": 0}
    out = []
    for cls, conf, box in sorted(dets, key=lambda d: -d[1]):
        if cls not in PEOPLE:
            continue
        dup = next((o for o in out if iou(box, o[2]) >= 0.7), None)
        if dup is None:
            out.append([cls, conf, box])
        elif rank[cls] > rank[dup[0]]:
            dup[0], dup[1] = cls, conf
    return [tuple(o) for o in out]


def pick_shooter(times, dets, make: float):
    """(frame index, box, method) of the shooter, or None. dets per frame
    are (class, conf, box) from the detector."""
    best = None
    for fi, (t, ds) in enumerate(zip(times, dets)):
        if not (make - SHOT_BEFORE <= t <= make - SHOT_AFTER):
            continue
        balls = [b for cls, c, b in ds if cls == "ball" and c > 0.25]
        for cls, conf, box in people(ds):
            if cls != "scorer":
                continue
            at_hands = ball_at_hands(box, balls)
            score = conf + (0.5 if at_hands else 0.0)
            if best is None or score > best[0]:
                best = (score, fi, box, "scorer+ball" if at_hands else "scorer")
    if best:
        return best[1:]
    # no shooting pose seen: the last ball holder before the make
    for fi in range(len(times) - 1, -1, -1):
        if times[fi] > make - 0.3:
            continue
        holders = [(c, b) for cls, c, b in people(dets[fi]) if cls == "holder"]
        if holders:
            return fi, max(holders)[1], "holder"
    return None


def follow(dets, anchor_fi: int, anchor_box) -> dict[int, list[float]]:
    """The anchored person's box per frame, linked by overlap both ways."""
    track = {anchor_fi: anchor_box}
    for step in (-1, 1):
        cur, fi, gap = anchor_box, anchor_fi + step, 0
        while 0 <= fi < len(dets) and gap <= 3:
            cands = [b for _, _, b in people(dets[fi])]
            best = max(cands, key=lambda b: iou(cur, b)) if cands else None
            if best is not None and iou(cur, best) > 0.35:
                track[fi], cur, gap = best, best, 0
            else:
                gap += 1
            fi += step
    return track


def clean_frames(track, dets) -> list[int]:
    """Frames where nobody else overlaps the tracked person."""
    out = []
    for fi, box in sorted(track.items()):
        others = [b for _, _, b in people(dets[fi]) if iou(box, b) < 0.7]
        if all(iou(box, b) < 0.15 for b in others):
            out.append(fi)
    return out


def read_window(clip_path, make: float):
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * STEP_SEC))
    f0, f1 = int(max(0.0, make - WINDOW_BEFORE) * fps), int((make + WINDOW_AFTER) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames, times = [], []
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        if (f - f0) % step == 0:
            frames.append(img)
            times.append(f / fps)
    cap.release()
    return frames, times


def find_shooter(clip_path, make: float, detector, batch: int = 8) -> Shooter | None:
    """The shooter of the make at clip time `make`, with crops of them."""
    import cv2

    frames, times = read_window(clip_path, make)
    if not frames:
        return None
    # detect on <= 1080p frames; boxes are scaled back to the source size
    h, w = frames[0].shape[:2]
    k = max(1.0, w / 1920)
    small = frames if k == 1.0 else [cv2.resize(f, (round(w / k), round(h / k))) for f in frames]
    dets = []
    for i in range(0, len(small), batch):
        for frame_dets in detector.detect_all(small[i:i + batch], conf=0.25):
            dets.append([(c, s, [v * k for v in b]) for c, s, b in frame_dets])
    picked = pick_shooter(times, dets, make)
    if picked is None:
        return None
    fi, box, method = picked
    track = follow(dets, fi, box)
    fis = clean_frames(track, dets)
    if len(fis) < 3:
        fis = sorted(track)
    if len(fis) > MAX_CROPS:
        fis = [fis[int(i)] for i in np.linspace(0, len(fis) - 1, MAX_CROPS)]
    crops = []
    for f in fis:
        x1, y1, x2, y2 = track[f]
        crop = frames[f][max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
        if crop.size:
            crops.append(crop)
    return Shooter(box=[round(v, 1) for v in box], t=round(times[fi], 2), method=method, crops=crops)
