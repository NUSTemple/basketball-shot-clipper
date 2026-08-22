"""Per-video record of which source timestamp each clip came from.

The old flow threw this away. cut_all() knew every clip's timestamp - it
returns them - but only the .mp4 files survived, so by review time a clip
was just a file with an index in its name. That was fine while the only
question asked of a clip was "goal or not", and became the blocker for
everything the new flow needs:

- putting detected candidates on a timeline the reviewer can scrub,
- letting the reviewer mark a shot detection missed entirely,
- cutting the final clip from the *original* at export time rather than
  shipping whatever the review copy happened to be.

So each clips/<video_stem>/ directory gets a _shots.json alongside the
_filter_scores.json that already lives there.

One rule worth stating outright: shot names are never reused. labels.json
is keyed by "<video_stem>/shot_NNN.mp4", so handing a recycled name to a
new shot would silently inherit the deleted shot's label and star rating.
next_index only ever moves forward, even across deletions.
"""
import json
from pathlib import Path

SHOTS_FILENAME = "_shots.json"

DETECTED = "detected"  # proposed by detect_shots.find_makes
MANUAL = "manual"      # marked by a human in the review view


def manifest_path(video_dir: Path) -> Path:
    return video_dir / SHOTS_FILENAME


def clip_name(index: int) -> str:
    """The filename cut_all gives shot number `index`."""
    return f"shot_{index:03d}.mp4"


def empty(video_name: str, source_video: Path | None = None,
          duration_s: float | None = None) -> dict:
    return {"video": video_name,
            "source_video": str(source_video) if source_video else None,
            "duration_s": duration_s,
            "shots": {},
            "next_index": 1}


def load(video_dir: Path) -> dict | None:
    """The manifest for this clips directory, or None if it predates them.

    Absent is a normal state, not an error: every video processed before
    _shots.json existed has clips but no manifest, and the review view
    simply can't offer a timeline for those.
    """
    path = manifest_path(video_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    data.setdefault("shots", {})
    data.setdefault("next_index", _highest_index(data["shots"]) + 1)
    return data


def save(video_dir: Path, data: dict) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(video_dir).write_text(json.dumps(data, indent=2, sort_keys=True))


def _highest_index(shots: dict) -> int:
    highest = 0
    for name in shots:
        stem = Path(name).stem
        _, _, digits = stem.partition("_")
        if digits.isdigit():
            highest = max(highest, int(digits))
    return highest


def from_cut_results(video_name: str, cut_results, source_video: Path | None = None,
                     duration_s: float | None = None, source: str = DETECTED) -> dict:
    """Build a manifest from cut_all()'s (index, timestamp, path) tuples."""
    data = empty(video_name, source_video, duration_s)
    for index, timestamp, out_path in cut_results:
        data["shots"][out_path.name] = {"t": round(float(timestamp), 2), "source": source}
    data["next_index"] = _highest_index(data["shots"]) + 1
    return data


def add_shot(data: dict, timestamp: float, source: str = MANUAL) -> str:
    """Reserve the next shot name for `timestamp`. Returns the clip name.

    Mutates `data`; the caller saves once the clip is actually on disk, so
    a failed cut doesn't leave a manifest entry pointing at nothing.
    """
    index = data["next_index"]
    name = clip_name(index)
    data["shots"][name] = {"t": round(float(timestamp), 2), "source": source}
    data["next_index"] = index + 1
    return name


def timestamp_for(data: dict | None, name: str) -> float | None:
    if not data:
        return None
    entry = data.get("shots", {}).get(name)
    return entry.get("t") if entry else None


def nearby(data: dict | None, timestamp: float, window: float) -> str | None:
    """Name of an existing shot within `window` seconds of `timestamp`.

    Used to stop a reviewer silently creating a duplicate of a candidate
    that detection already found - at review speed it is genuinely hard to
    tell whether the marker under the playhead is the shot you just watched.
    """
    if not data:
        return None
    best, best_gap = None, window
    for name, entry in data.get("shots", {}).items():
        gap = abs(float(entry["t"]) - timestamp)
        if gap <= best_gap:
            best, best_gap = name, gap
    return best
