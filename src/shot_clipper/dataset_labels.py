"""Shared read/write for data/dataset/labels.json, the goal/no_goal label store.

Used by both shot_clipper.label_ui.app (writes labels as the user clicks
through clips) and shot_clipper.build_dataset (reads labels to materialize
the dataset folder layout). Paths are resolved relative to the current
working directory - run these tools from the repo root.
"""
import json
import os
from pathlib import Path

# Overridable for the hosted deployment, where label-ui and worker are
# separate containers with separate ephemeral filesystems - the plain
# "data/dataset" default would put labels.json somewhere invisible to
# whichever of the two containers didn't write it.
DATASET_DIR = Path(os.environ.get("SHOT_CLIPPER_DATASET_DIR", "data/dataset"))
VALID_LABELS = {"goal", "no_goal"}


def labels_path(base: Path | None = None) -> Path:
    return (base or DATASET_DIR) / "labels.json"


def load_labels(base: Path | None = None) -> dict:
    path = labels_path(base)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_labels(labels: dict, base: Path | None = None) -> None:
    path = labels_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True))
