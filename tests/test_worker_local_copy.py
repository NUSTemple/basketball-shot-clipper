import pytest

from shot_clipper.label_ui import worker


def _video(tmp_path, size):
    path = tmp_path / "game.mp4"
    path.write_bytes(b"\0" * size)
    return path


def test_no_copy_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "COPY_VIDEO_LOCALLY", False)
    monkeypatch.setattr(worker, "MAX_LOCAL_COPY_GB", 1e-9)
    video = _video(tmp_path, 100)
    with worker._local_video(video) as local:
        assert local == video


def test_copies_video_under_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "COPY_VIDEO_LOCALLY", True)
    monkeypatch.setattr(worker, "MAX_LOCAL_COPY_GB", 1.0)
    video = _video(tmp_path, 100)
    with worker._local_video(video) as local:
        assert local != video
        assert local.read_bytes() == video.read_bytes()
    assert not local.exists()


def test_rejects_video_over_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "COPY_VIDEO_LOCALLY", True)
    monkeypatch.setattr(worker, "MAX_LOCAL_COPY_GB", 50 / 1e9)
    video = _video(tmp_path, 100)
    with pytest.raises(ValueError, match="too large to process: game.mp4"):
        with worker._local_video(video):
            pass


def test_no_limit_when_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "COPY_VIDEO_LOCALLY", True)
    monkeypatch.setattr(worker, "MAX_LOCAL_COPY_GB", 0)
    video = _video(tmp_path, 100)
    with worker._local_video(video) as local:
        assert local.is_file()
