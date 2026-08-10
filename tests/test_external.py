"""Finding ffmpeg without relying on PATH.

A packaged install ships its own ffmpeg - not needing a separate winget
install is most of the point - so the bare "ffmpeg" string these commands
used to start with is exactly the thing that can't be trusted once the app
leaves a developer machine.
"""
import sys

import pytest

from shot_clipper import external, paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(external.FFMPEG_ENV, raising=False)
    monkeypatch.delenv(external.FFPROBE_ENV, raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def _bundle(monkeypatch, tmp_path, *tools):
    """Fake a frozen bundle carrying `tools` in its bin/ directory."""
    bin_dir = tmp_path / external.BUNDLED_BIN_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    for tool in tools:
        exe = f"{tool}.exe" if sys.platform == "win32" else tool
        (bin_dir / exe).write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    return bin_dir


def test_bundled_ffmpeg_wins_over_path(tmp_path, monkeypatch):
    bin_dir = _bundle(monkeypatch, tmp_path, "ffmpeg", "ffprobe")
    monkeypatch.setattr(external.shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    assert external.ffmpeg_exe().startswith(str(bin_dir))
    assert external.ffprobe_exe().startswith(str(bin_dir))


def test_env_override_wins_over_the_bundle(tmp_path, monkeypatch):
    _bundle(monkeypatch, tmp_path, "ffmpeg")
    monkeypatch.setenv(external.FFMPEG_ENV, "/opt/custom/ffmpeg")
    assert external.ffmpeg_exe() == "/opt/custom/ffmpeg"


def test_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(external.shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    assert external.ffmpeg_exe() == "/usr/bin/ffmpeg"
    assert external.have_ffmpeg()


def test_missing_everywhere_still_yields_a_usable_argv_entry(monkeypatch):
    """Callers build a command list before anything runs; a None here would
    raise while assembling it instead of failing as the "ffmpeg not found"
    it actually is."""
    monkeypatch.setattr(external.shutil, "which", lambda _n: None)
    assert external.ffmpeg_exe() == "ffmpeg"
    assert not external.have_ffmpeg()


def test_bundled_lookup_is_relative_to_the_resource_dir(tmp_path, monkeypatch):
    """Bundled binaries are read-only payload, so they hang off
    resource_dir(), not the writable app root."""
    _bundle(monkeypatch, tmp_path, "ffmpeg")
    monkeypatch.setattr(external.shutil, "which", lambda _n: None)
    assert paths.resource_dir() == tmp_path
    assert external.ffmpeg_exe().startswith(str(tmp_path))


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows problem")
def test_no_window_flag_on_windows():
    """A --noconsole build has no console for a child to inherit, so every
    ffmpeg call would flash its own black window."""
    assert external.no_window_kwargs()["creationflags"] & 0x08000000


@pytest.mark.skipif(sys.platform == "win32", reason="creationflags is Windows-only")
def test_no_window_is_empty_off_windows():
    assert external.no_window_kwargs() == {}
