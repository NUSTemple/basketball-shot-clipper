"""Shared read/write for <data dir>/dataset/labels.json, the goal/no_goal
label store.

Used by both shot_clipper.label_ui.app (writes labels as the user clicks
through clips) and shot_clipper.build_dataset (reads labels to materialize
the dataset folder layout). See paths.py for where the data directory
actually lands - in a checkout it's still ./data.
"""
import json
from pathlib import Path

from . import paths

VALID_LABELS = {"goal", "no_goal"}


def labels_path() -> Path:
    return paths.dataset_dir() / "labels.json"


def load_labels() -> dict:
    path = labels_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_labels(labels: dict) -> None:
    path = labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True))
