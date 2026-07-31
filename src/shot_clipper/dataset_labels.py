"""Shared read/write for data/dataset/labels.json, the goal/no_goal label store.

Used by both shot_clipper.label_ui.app (writes labels as the user clicks
through clips) and shot_clipper.build_dataset (reads labels to materialize
the dataset folder layout). Paths are resolved relative to the current
working directory - run these tools from the repo root.
"""
import json
from pathlib import Path

DATASET_DIR = Path("data/dataset")
VALID_LABELS = {"goal", "no_goal"}


def labels_path() -> Path:
    return DATASET_DIR / "labels.json"


def load_labels() -> dict:
    path = labels_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_labels(labels: dict) -> None:
    path = labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True))
