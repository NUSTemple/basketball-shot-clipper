"""Compare detected shot timestamps against a human-recorded ground truth.

Usage:
    shot-clipper-validate <detected_json> <ground_truth_json> [--tolerance 2.0]

Both files are either {"makes_sec": [...]} or a plain JSON list of seconds.
Ground truth is meant to be recorded by a human watching the video once
(decision 9 in PLAN.md) - this script does not fabricate it.
"""
import argparse
import json
from pathlib import Path


def load_timestamps(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data["makes_sec"]
    return data


def match(detected, truth, tolerance):
    remaining_truth = list(truth)
    matched_pairs = []
    false_positives = []

    for t in sorted(detected):
        candidates = [g for g in remaining_truth if abs(g - t) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda g: abs(g - t))
            remaining_truth.remove(best)
            matched_pairs.append((t, best))
        else:
            false_positives.append(t)

    missed = remaining_truth
    return matched_pairs, false_positives, missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detected", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    detected = load_timestamps(args.detected)
    truth = load_timestamps(args.ground_truth)

    matched, false_positives, missed = match(detected, truth, args.tolerance)

    recall = len(matched) / len(truth) if truth else float("nan")
    precision = len(matched) / len(detected) if detected else float("nan")

    print(f"ground truth makes: {len(truth)}")
    print(f"detected makes:     {len(detected)}")
    print(f"matched:            {len(matched)}")
    print(f"missed (recall miss): {len(missed)} -> {[round(m, 2) for m in missed]}")
    print(f"false positives:    {len(false_positives)} -> {[round(f, 2) for f in false_positives]}")
    print(f"recall:    {recall:.1%}")
    print(f"precision: {precision:.1%}")


if __name__ == "__main__":
    main()
