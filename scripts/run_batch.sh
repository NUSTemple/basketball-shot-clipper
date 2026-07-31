#!/bin/bash
# Batch-run detection + clipping over several source videos.
# Run from anywhere; cds to the repo root itself so relative data/models/
# paths resolve. Requires `poetry install --with ml` to have been run.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DIR="/Users/pengtan/Videos/20260725 Basketball Video/DJI_001"
IDS="0002 0003 0006 0007 0008 0009 0011"  # 0001 already done

for id in $IDS; do
  f=$(ls "$DIR"/DJI_*_${id}_D.MP4)
  echo "=== $(date) processing $id: $f ==="
  poetry run shot-clipper-detect "$f"
  detected="data/ground_truth/$(basename "${f%.MP4}")_detected.json"
  poetry run shot-clipper-clip "$f" "$detected"
  echo "=== $(date) done $id ==="
done
echo "=== ALL DONE ==="
