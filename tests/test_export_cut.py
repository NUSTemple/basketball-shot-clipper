"""Export cuts from the original, and has to actually run.

In the review-first flow the clip on disk is a 720p preview, so export is
where the real cut happens: same timestamp, straight from the source. That
path shipped with ThreadPoolExecutor used but never imported - which no
test noticed, because nothing exercised export end to end. It failed with
a 500 the first time somebody pressed the button, days later.

These call the pieces for real rather than mocking them, since the bug was
precisely that the module didn't have what it needed at runtime.
"""
import json
from pathlib import Path

import pytest

from shot_clipper import shot_manifest
from shot_clipper.label_ui import app as label_app


@pytest.fixture
def clips_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CLIPPER_DATA_DIR", str(tmp_path / "data"))
    clips = tmp_path / "clips"
    video_dir = clips / "DJI_0001_D"
    video_dir.mkdir(parents=True)
    (video_dir / "shot_001.mp4").write_bytes(b"preview")
    label_app.app.config["CLIPS_DIR"] = clips
    return clips


def test_cut_for_export_runs_with_no_work(clips_dir):
    """The regression: this used to raise NameError before doing anything,
    because the executor it needs was never imported."""
    assert label_app._cut_for_export([]) == []


def test_cut_for_export_reports_failures_instead_of_raising(clips_dir, tmp_path):
    """A source that can't be cut becomes a reported miss, not a 500."""
    missing = tmp_path / "not-a-video.MP4"
    out = tmp_path / "out.mp4"
    failures = label_app._cut_for_export([((missing, 10.0), "DJI_0001_D/shot_001.mp4", out)])
    assert failures == [("DJI_0001_D/shot_001.mp4", "out.mp4")]


def test_preview_clips_are_marked_for_re_cutting(clips_dir):
    """A manifest saying the clips are proxy previews is what routes export
    through the original instead of copying the preview."""
    data = shot_manifest.empty("DJI_0001_D.MP4", source_video=clips_dir / "src.MP4")
    data["shots"]["shot_001.mp4"] = {"t": 42.0, "source": "detected"}
    data["preview_source"] = "proxy"
    (clips_dir / "src.MP4").write_bytes(b"")
    shot_manifest.save(clips_dir / "DJI_0001_D", data)

    got = label_app._delivery_source(clips_dir, "DJI_0001_D/shot_001.mp4")
    assert got == (clips_dir / "src.MP4", 42.0)


def test_delivery_quality_clips_are_copied_not_re_cut(clips_dir):
    """Videos cut before the review-first flow have real clips on disk;
    export should ship those rather than re-encoding them."""
    data = shot_manifest.empty("DJI_0001_D.MP4", source_video=clips_dir / "src.MP4")
    data["shots"]["shot_001.mp4"] = {"t": 42.0, "source": "detected"}
    data["preview_source"] = "original"
    shot_manifest.save(clips_dir / "DJI_0001_D", data)

    assert label_app._delivery_source(clips_dir, "DJI_0001_D/shot_001.mp4") is None


def test_missing_original_falls_back_to_copying(clips_dir):
    """The source moved or the drive isn't mounted - export should still
    produce something rather than failing the whole batch."""
    data = shot_manifest.empty("DJI_0001_D.MP4", source_video=Path("D:/gone/src.MP4"))
    data["shots"]["shot_001.mp4"] = {"t": 42.0, "source": "detected"}
    data["preview_source"] = "proxy"
    shot_manifest.save(clips_dir / "DJI_0001_D", data)

    assert label_app._delivery_source(clips_dir, "DJI_0001_D/shot_001.mp4") is None


def test_no_manifest_falls_back_to_copying(clips_dir):
    assert label_app._delivery_source(clips_dir, "DJI_0001_D/shot_001.mp4") is None


def test_export_endpoint_is_reachable(clips_dir, tmp_path):
    """Exercises the whole view, which is what would have caught the
    missing import: a bad request should come back as 400, not 500."""
    client = label_app.app.test_client()
    res = client.post("/api/export-clips", json={"clips": [], "dest": str(tmp_path / "out")})
    assert res.status_code == 400  # empty selection, rejected cleanly
    assert "error" in res.get_json()
