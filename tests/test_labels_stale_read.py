"""A labels.json read that hits ESTALE on the GCS mount is retried with a
fresh open instead of 500ing the click that triggered it."""
import errno
from pathlib import Path

import pytest

from shot_clipper import dataset_labels


def _flaky_read_text(fail_times, err=errno.ESTALE):
    real = Path.read_text
    calls = {"n": 0}

    def read_text(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise OSError(err, "Stale file handle")
        return real(self, *a, **kw)

    return read_text, calls


@pytest.fixture
def labels_dir(tmp_path, monkeypatch):
    (tmp_path / "labels.json").write_text('{"a/b/shot_001.mp4": {"label": "goal"}}')
    monkeypatch.setattr(dataset_labels, "STALE_READ_BACKOFF_SEC", 0)
    return tmp_path


def test_stale_handle_is_retried(labels_dir, monkeypatch):
    read_text, calls = _flaky_read_text(fail_times=1)
    monkeypatch.setattr(Path, "read_text", read_text)
    assert dataset_labels.load_labels(labels_dir) == {"a/b/shot_001.mp4": {"label": "goal"}}
    assert calls["n"] == 2


def test_gives_up_after_the_last_attempt(labels_dir, monkeypatch):
    read_text, _ = _flaky_read_text(fail_times=99)
    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(OSError):
        dataset_labels.load_labels(labels_dir)


def test_other_os_errors_are_not_retried(labels_dir, monkeypatch):
    read_text, calls = _flaky_read_text(fail_times=1, err=errno.EACCES)
    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(OSError):
        dataset_labels.load_labels(labels_dir)
    assert calls["n"] == 1


def test_missing_file_is_empty(tmp_path):
    assert dataset_labels.load_labels(tmp_path) == {}
