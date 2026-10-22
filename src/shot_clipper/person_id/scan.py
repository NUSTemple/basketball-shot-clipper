"""shot-clipper-players: group a clips folder's goals by who scored.

    shot-clipper-players "<clips dir>"

For every goal clip (a person's goal label, else an AI make score >= 0.6,
else every clip when neither exists): find the shooter, embed them, and
save it to <video folder>/_players.json (cached - re-runs only embed new
clips). Then, per session (the recording date in the video name), group the
goals and suggest a name per group (grouping.py), written back to the same
files - the label UI's Players tab reads them to let a person confirm a
name once per group.

Confirmed names live in labels.json as each clip's `scorer`, like a scorer
tagged by hand in Review. Every scan adds the tagged goals it can see to the
gallery (data/players/gallery.jsonl - local only, it holds people's names),
which is what suggests names for later sessions.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from . import grouping, reid

from . import PLAYERS_FILENAME
GALLERY_PATH = Path(os.environ.get("SHOT_CLIPPER_PLAYERS_GALLERY", "data/players/gallery.jsonl"))
DEFAULT_MAKE_SEC = 5.0  # cut_all puts a clip's first candidate here
MAKE_SCORE_MIN = 0.6    # prelabel.UI_MAKE_FROM


def session_of(video_key: str) -> str:
    m = re.search(r"(20\d{6})", video_key)
    return m.group(1) if m else video_key


def video_folders(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("shot_*.mp4")})


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def goal_clips(folder: Path, labels: dict, key_of) -> list[Path]:
    """A label wins (goal in, anything else out); unlabeled clips go by the
    AI make score; a folder with neither has every clip tried."""
    scores = read_json(folder / "_prelabel_scores.json")
    out = []
    for clip in sorted(folder.glob("shot_*.mp4")):
        label = (labels.get(key_of(clip)) or {}).get("label")
        if label:
            if label == "goal":
                out.append(clip)
        elif not scores or scores.get(clip.name, 0.0) >= MAKE_SCORE_MIN:
            out.append(clip)
    return out


def make_time(folder: Path, clip: Path) -> float:
    shots = read_json(folder / "_highlight_shots.json").get("clips", {})
    return float(shots.get(clip.name, {}).get("make", DEFAULT_MAKE_SEC))


def load_gallery(path: Path = GALLERY_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["key"]] = e
    return out


def save_gallery(entries: dict[str, dict], path: Path = GALLERY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries.values()))
    tmp.replace(path)


def scan(root: Path, labels: dict, key_of=None, detector=None, model=None, device: str = "cpu",
         only: set[str] | None = None, gallery_path: Path = GALLERY_PATH, log=print) -> dict:
    """Embed new goal clips under root, regroup and rename every session
    found there. key_of maps a clip path to its labels.json key (default:
    path relative to root). only: restrict to these clip keys. Returns
    {session: {"goals", "groups", "named", "tagged"}}."""
    from .shooter import find_shooter

    root = Path(root)
    key_of = key_of or (lambda p: str(p.relative_to(root)))
    folders = video_folders(root)
    for folder in folders:
        path = folder / PLAYERS_FILENAME
        data = read_json(path)
        if data.get("method") != reid.METHOD:
            data = {"method": reid.METHOD, "clips": {}}
        todo = [c for c in goal_clips(folder, labels, key_of)
                if c.name not in data["clips"] and (only is None or key_of(c) in only)]
        if not todo:
            continue
        if detector is None:
            from .. import inference
            detector = inference.load_detector(device=device)
        if model is None:
            model = reid.ReidModel(device=device)
        log(f"{folder.name}: finding the shooter in {len(todo)} goals")
        for clip in todo:
            shooter = find_shooter(clip, make_time(folder, clip), detector)
            emb = model.embed(shooter.crops) if shooter else None
            data["clips"][clip.name] = {
                "emb": [round(float(v), 4) for v in emb] if emb is not None else None,
                "t": shooter.t if shooter else None, "box": shooter.box if shooter else None,
                "how": shooter.method if shooter else None}
        path.write_text(json.dumps(data))

    # gallery: every goal a person tagged with a scorer, wherever it was seen
    gallery = load_gallery(gallery_path)
    per_folder = {f: read_json(f / PLAYERS_FILENAME) for f in folders}
    for folder, data in per_folder.items():
        session = session_of(folder.name)
        for name, info in data.get("clips", {}).items():
            key = key_of(folder / name)
            tag = (labels.get(key) or {}).get("scorer")
            gkey = f"{session}|{key}"
            if tag and info.get("emb"):
                gallery[gkey] = {"key": gkey, "session": session, "name": tag, "emb": info["emb"]}
            else:
                gallery.pop(gkey, None)
    save_gallery(gallery, gallery_path)

    # group + name per session
    sessions = defaultdict(list)  # session -> [(folder, clip name, emb, tag)]
    for folder, data in per_folder.items():
        for name, info in data.get("clips", {}).items():
            if info.get("emb"):
                tag = (labels.get(key_of(folder / name)) or {}).get("scorer")
                sessions[session_of(folder.name)].append((folder, name, info["emb"], tag))
    summary = {}
    for session, items in sessions.items():
        idx = grouping.group([e for _, _, e, _ in items])
        members = defaultdict(list)
        for (_, _, emb, tag), g in zip(items, idx):
            members[g].append((emb, tag))
        named = grouping.name_groups(members, list(gallery.values()))
        for (folder, name, _, _), g in zip(items, idx):
            info = per_folder[folder]["clips"][name]
            info["group"] = f"{session}-{g:02d}"
            info["suggested"] = named[g]["name"]
            info["confidence"] = named[g]["confidence"]
            info["suggested_how"] = named[g]["how"]
        summary[session] = {"goals": len(items), "groups": len(members),
                            "named": sum(1 for g in idx if named[g]["name"]),
                            "tagged": sum(1 for g in idx if named[g]["how"] == "tagged")}
        log(f"{session}: {len(items)} goals in {len(members)} groups, "
            f"{summary[session]['named']} with a suggested name")
    for folder, data in per_folder.items():
        if data:
            (folder / PLAYERS_FILENAME).write_text(json.dumps(data))
    return summary


def main() -> None:
    from ..dataset_labels import load_labels
    from ..device_config import get_device

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips_dir", type=Path)
    ap.add_argument("--labels", type=Path, default=None, help="labels.json to read tags from "
                    "(default: the label UI's data/dataset/labels.json)")
    ap.add_argument("--label-prefix", default="", help="prefix of labels keys, e.g. '<account>/' for a "
                    "copy of the hosted app's labels")
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text()) if args.labels else load_labels()
    if args.label_prefix:
        labels = {k[len(args.label_prefix):]: v for k, v in labels.items() if k.startswith(args.label_prefix)}
    summary = scan(args.clips_dir, labels, device=get_device())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
