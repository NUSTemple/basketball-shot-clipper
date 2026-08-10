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


def test_everything_else_starts_the_label_ui(monkeypatch):
    started = []
    monkeypatch.setattr("shot_clipper.label_ui.app.main", lambda: started.append(list(sys.argv)))
    monkeypatch.setattr("shot_clipper.label_ui.worker.main",
                        lambda: pytest.fail("started a worker instead of the label UI"))
    monkeypatch.setattr(sys, "argv", ["shot-clipper", "--port", "5051"])

    entry.main()
    assert started == [["shot-clipper", "--port", "5051"]], "app args must pass through untouched"


def test_spawn_worker_uses_the_shared_argv_builder():
    """jobs.py must not grow its own copy of the command - that's how the two
    halves get out of step."""
    from shot_clipper.label_ui import jobs

    assert jobs.worker_argv is entry.worker_argv
