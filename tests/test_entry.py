"""The frozen build's self-invocation contract.

jobs._spawn_worker runs the worker as a real subprocess so a job survives the
Flask process restarting. Unfrozen it can say `python -m
shot_clipper.label_ui.worker`; a frozen sys.executable is the app's own
launcher and has no -m, so the exe has to be able to start a worker on
request. If either half of that drifts, Detect silently stops working in the
packaged app only - the failure never shows up in a checkout.
"""
import sys

import pytest

from shot_clipper import entry


@pytest.fixture(autouse=True)
def _unfrozen(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)


def test_unfrozen_argv_uses_dash_m():
    argv = entry.worker_argv("data/jobs/abc.json")
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "shot_clipper.label_ui.worker"]
    assert argv[3] == "data/jobs/abc.json"


def test_frozen_argv_asks_the_exe_for_a_worker(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\app\shot-clipper.exe")
    assert entry.worker_argv("j.json") == [r"C:\app\shot-clipper.exe",
                                            entry.WORKER_FLAG, "j.json"]
    assert "-m" not in entry.worker_argv("j.json")


def test_worker_flag_dispatches_to_the_worker(monkeypatch):
    """And strips the flag: the worker must see the argv it would have had
    from `python -m ... <job file>`."""
    seen = {}
    monkeypatch.setattr("shot_clipper.label_ui.worker.main",
                        lambda: seen.update(argv=list(sys.argv)))
    monkeypatch.setattr("shot_clipper.label_ui.app.main",
                        lambda: pytest.fail("started the label UI instead of a worker"))
    monkeypatch.setattr(sys, "argv", ["shot-clipper", entry.WORKER_FLAG, "j.json"])

    entry.main()
    assert seen["argv"] == ["shot-clipper", "j.json"]


def test_bare_invocation_opens_a_window(monkeypatch):
    opened = []
    monkeypatch.setattr("shot_clipper.desktop.available", lambda: True)
    monkeypatch.setattr("shot_clipper.desktop.main", lambda: opened.append(list(sys.argv)))
    monkeypatch.setattr("shot_clipper.label_ui.app.main",
                        lambda: pytest.fail("served to a browser instead of opening a window"))
    monkeypatch.setattr(sys, "argv", ["shot-clipper", "--clips-dir", "X"])

    entry.main()
    assert opened == [["shot-clipper", "--clips-dir", "X"]], "args must pass through untouched"


def test_server_flag_serves_to_a_browser(monkeypatch):
    """Docker's CMD and any headless run need the plain Flask server, and must
    never try to open a window on a machine that has no display."""
    served = []
    monkeypatch.setattr("shot_clipper.label_ui.app.main", lambda: served.append(list(sys.argv)))
    monkeypatch.setattr("shot_clipper.desktop.main",
                        lambda: pytest.fail("opened a window in server mode"))
    monkeypatch.setattr(sys, "argv", ["shot-clipper", "--server", "--port", "5051"])

    entry.main()
    assert served == [["shot-clipper", "--port", "5051"]], "--server is consumed, the rest passes on"


def test_missing_toolkit_degrades_to_the_browser(monkeypatch, capsys):
    """A missing window toolkit is not a reason to fail to start - it's the
    same app either way, only the frame differs."""
    served = []
    monkeypatch.setattr("shot_clipper.desktop.available", lambda: False)
    monkeypatch.setattr("shot_clipper.desktop.main",
                        lambda: pytest.fail("opened a window with no toolkit installed"))
    monkeypatch.setattr("shot_clipper.label_ui.app.main", lambda: served.append(True))
    monkeypatch.setattr(sys, "argv", ["shot-clipper"])

    entry.main()
    assert served == [True]
    assert "browser" in capsys.readouterr().out


def test_spawn_worker_uses_the_shared_argv_builder():
    """jobs.py must not grow its own copy of the command - that's how the two
    halves get out of step."""
    from shot_clipper.label_ui import jobs

    assert jobs.worker_argv is entry.worker_argv
