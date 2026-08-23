"""Where the filter's training pixels come from.

Review clips are 720p ultrafast re-encodes of a proxy, and net_motion reads
raw pixel differences - so features have to be extracted from the original
at the shot's timestamp, not from the clip sitting on disk. The manifest is
what makes that possible.

The first version of this looked up the timestamp with the clip's name plus
".mp4" appended, on a name that already ended in .mp4. Every lookup missed,
every clip was skipped, and training failed with an error about zero
cross-validation splits - a long way from the actual mistake.
"""
import json

import pytest

from shot_clipper import shot_manifest, train_filter


@pytest.fixture
def manifest():
    data = shot_manifest.empty("DJI_0001.MP4", source_video=None)
    data["shots"]["shot_001.mp4"] = {"t": 120.5, "source": "detected"}
    data["preview_source"] = "proxy"
    return data


def test_a_real_clip_is_used_as_is(tmp_path):
    """Videos cut at delivery quality have nothing to re-cut."""
    clip = tmp_path / "shot_001.mp4"
    clip.write_bytes(b"")
    with train_filter.feature_source(clip, None, "shot_001.mp4") as source:
        assert source == clip


def test_original_clips_are_used_even_with_a_manifest(tmp_path, manifest):
    clip = tmp_path / "shot_001.mp4"
    clip.write_bytes(b"")
    manifest["preview_source"] = "original"
    with train_filter.feature_source(clip, manifest, "shot_001.mp4") as source:
        assert source == clip


def test_preview_with_no_reachable_original_yields_none(tmp_path, manifest):
    """Better to skip the clip loudly than extract from the wrong pixels."""
    clip = tmp_path / "shot_001.mp4"
    clip.write_bytes(b"")
    manifest["source_video"] = str(tmp_path / "gone.MP4")
    with train_filter.feature_source(clip, manifest, "shot_001.mp4") as source:
        assert source is None


def test_the_clip_name_is_looked_up_verbatim(manifest):
    """The regression: the name already ends in .mp4, so appending another
    one silently misses every shot in the manifest."""
    assert shot_manifest.timestamp_for(manifest, "shot_001.mp4") == 120.5
    assert shot_manifest.timestamp_for(manifest, "shot_001.mp4.mp4") is None


def test_unknown_shot_yields_none(tmp_path, manifest):
    clip = tmp_path / "shot_999.mp4"
    clip.write_bytes(b"")
    manifest["source_video"] = str(clip)  # exists, but the shot doesn't
    with train_filter.feature_source(clip, manifest, "shot_999.mp4") as source:
        assert source is None


def test_excluded_videos_are_dropped(tmp_path, monkeypatch):
    """Practice footage produces candidates at a different rate to a game;
    training on both puts the threshold between two distributions."""
    labels = {"game/shot_001.mp4": {"label": "goal"},
              "practice/shot_001.mp4": {"label": "goal"}}
    seen = []

    def fake_source(clip_path, manifest, name):
        seen.append(clip_path)
        raise StopIteration  # never reached; the clips don't exist on disk

    for rel in labels:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    monkeypatch.setattr(train_filter, "hoop_bbox_for_video",
                        lambda video, clips_dir: [0.1, 0.1, 0.2, 0.2])
    monkeypatch.setattr(train_filter, "extract_features_for_clip",
                        lambda *a, **k: {"has_crossing": 1})
    monkeypatch.setattr(train_filter, "extract_motion_features_for_clip",
                        lambda *a, **k: {"diff_peak": 0.5})

    rows = list(train_filter.build_feature_rows(
        labels, tmp_path, detector=None, device="cpu", fps=15.0,
        exclude_videos=["practice"]))

    assert [r["video"] for r in rows] == ["game"]
