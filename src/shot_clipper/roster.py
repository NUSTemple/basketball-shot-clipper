"""Shared read/write for data/dataset/roster.json, the list of known player
names used to tag who scored a goal clip.

Used by shot_clipper.label_ui.app (the Review panel's scorer picker). This
is also the roster an automated jersey-number/face suggester would match
against - see docs/PLAYER_IDENTIFICATION.md - but for now every scorer tag
is entered by hand.
"""
import json
import re
from pathlib import Path

from .dataset_labels import DATASET_DIR


def roster_path(base: Path | None = None) -> Path:
    return (base or DATASET_DIR) / "roster.json"


def load_roster(base: Path | None = None) -> list[str]:
    path = roster_path(base)
    if path.exists():
        return json.loads(path.read_text())["players"]
    return []


def save_roster(players: list[str], base: Path | None = None) -> None:
    path = roster_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"players": players}, indent=2))


def add_player(name: str, base: Path | None = None) -> list[str]:
    name = name.strip()
    if not name:
        raise ValueError("player name must not be empty")
    players = load_roster(base)
    if name not in players:
        players.append(name)
        players.sort(key=str.casefold)
        save_roster(players, base)
    return players


def scorer_slug(name: str) -> str:
    """Filesystem-safe fragment for a scorer's name, for filenames like
    <video>__shot_012_5star_Alice.mp4 (build_dataset.py, api_export_clips).
    Keeps non-ASCII letters (e.g. Chinese names) - only strips whitespace
    and path/filename-unsafe punctuation."""
    slug = re.sub(r'[\s/\\:*?"<>|]+', "", name)
    return slug or "player"
