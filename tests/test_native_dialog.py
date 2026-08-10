"""Which file picker the app offers depends on what's hosting it.

The desktop window has a real OS dialog; macOS has osascript; Docker and a
plain browser have neither, and must say so rather than erroring in a way
that looks like a bug - the in-app folder browser is the fallback there.
"""
import pytest

from shot_clipper import native_dialog


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch):
    """Every test starts with nothing registered - the module-level provider
    would otherwise leak between tests in registration order."""
    monkeypatch.setattr(native_dialog, "_provider", None)


def test_registered_provider_is_used(monkeypatch):
    monkeypatch.setattr(native_dialog, "_osascript_available", lambda: True)
    native_dialog.register(lambda kind, prompt: {"path": f"/picked/{kind}"})

    assert native_dialog.choose(native_dialog.FILE, "pick") == {"path": "/picked/file"}
    assert native_dialog.available()
    assert native_dialog.describe() == "native window dialog"


def test_provider_wins_over_osascript(monkeypatch):
    """A window's own dialog is the better one when both exist - on macOS,
    osascript would put the prompt behind the app window."""
    called = []
    monkeypatch.setattr(native_dialog, "_osascript_choose",
                        lambda *a: called.append("osascript") or {"path": "/mac"})
    monkeypatch.setattr(native_dialog, "_osascript_available", lambda: True)
    native_dialog.register(lambda kind, prompt: {"path": "/window"})

    assert native_dialog.choose(native_dialog.FOLDER, "pick") == {"path": "/window"}
    assert called == []


def test_falls_back_to_osascript(monkeypatch):
    monkeypatch.setattr(native_dialog, "_osascript_available", lambda: True)
    monkeypatch.setattr(native_dialog, "_osascript_choose",
                        lambda kind, prompt: {"path": "/mac/movie.mov"})

    assert native_dialog.choose(native_dialog.FILE, "pick") == {"path": "/mac/movie.mov"}
    assert native_dialog.describe() == "macOS osascript"


def test_no_picker_raises_something_actionable(monkeypatch):
    monkeypatch.setattr(native_dialog, "_osascript_available", lambda: False)

    assert not native_dialog.available()
    with pytest.raises(RuntimeError, match="type or paste the path"):
        native_dialog.choose(native_dialog.FILE, "pick")


def test_cancelling_is_not_an_error(monkeypatch):
    native_dialog.register(lambda kind, prompt: {"cancelled": True})
    assert native_dialog.choose(native_dialog.FILE, "pick") == {"cancelled": True}
