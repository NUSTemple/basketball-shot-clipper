"""Who actually played on a given day.

roster.json accumulates every player ever tagged - it only grows, because a
name that scored once is a name you might need again. That makes it the
wrong list to *choose* from: after a season the scorer picker offers
twenty-five people when eight were on court, and the right answer is
somewhere in a list mostly made of people who weren't there.

So a squad is the subset who played on one date, and the picker offers that
instead. Keyed by date rather than by video because the players are a fact
about the session, not the recording - the Library already groups by the
same key (see extractDateKey), and a session is several videos.

Two deliberate softnesses:

- A date with no squad falls back to the whole roster. The feature is opt-in
  and nothing breaks for footage labelled before it existed.
- The squad filters the picker, it does not police it. Someone turning up
  late, a guest, a name typed in a hurry - all still work, and adding a
  player while tagging puts them in the squad too, since otherwise you'd
  add a name and immediately watch it vanish from the list.
"""
import json
import re
from pathlib import Path

from . import paths

# DJI names its files DJI_YYYYMMDDHHMMSS_NNNN_D. Mirrors extractDateKey in
# the front end; anything else has no date and so no squad.
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})\d{6}")


def squads_path() -> Path:
    return paths.dataset_dir() / "squads.json"


def date_key_for(video_name: str) -> str | None:
    match = _DATE_RE.search(str(video_name))
    return f"{match[1]}-{match[2]}-{match[3]}" if match else None


def load_squads() -> dict[str, list[str]]:
    path = squads_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("squads", {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_squads(squads: dict[str, list[str]]) -> None:
    path = squads_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"squads": squads}, indent=2, sort_keys=True,
                               ensure_ascii=False), encoding="utf-8")


def squad_for(date_key: str | None) -> list[str] | None:
    """Who played that day, or None if nobody has said - which callers must
    read as "offer everyone", not "nobody played"."""
    if not date_key:
        return None
    return load_squads().get(date_key)


def squad_for_video(video_name: str) -> list[str] | None:
    return squad_for(date_key_for(video_name))


def set_squad(date_key: str, players: list[str]) -> list[str]:
    """Replace one date's squad. An empty list clears it, which restores the
    fallback rather than leaving a date that offers nobody."""
    cleaned, seen = [], set()
    for name in players:
        name = (name or "").strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            cleaned.append(name)

    squads = load_squads()
    if cleaned:
        squads[date_key] = sorted(cleaned, key=str.casefold)
    else:
        squads.pop(date_key, None)
    save_squads(squads)
    return cleaned


def add_to_squad(date_key: str | None, name: str) -> None:
    """Put a player in a date's squad if a squad exists for it.

    Called when someone is tagged as a scorer: without this, typing a name
    that isn't in the squad would tag the clip and leave the picker still
    not offering them for the next one.
    """
    name = (name or "").strip()
    if not (date_key and name):
        return
    squads = load_squads()
    current = squads.get(date_key)
    if current is None:  # no squad set for that day - nothing to extend
        return
    if not any(name.casefold() == p.casefold() for p in current):
        squads[date_key] = sorted([*current, name], key=str.casefold)
        save_squads(squads)
