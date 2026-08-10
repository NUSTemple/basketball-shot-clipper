"""Single entry point for a packaged build, and the reason it has to exist.

jobs._spawn_worker starts the job worker as a real subprocess (not a thread)
so a job survives the Flask process restarting. Unfrozen that's

    [sys.executable, "-m", "shot_clipper.label_ui.worker", <job file>]

which works because sys.executable is a Python interpreter. In a frozen
build it is the app's own launcher, which has no -m: the spawn would fail,
or - worse, if the launcher ignored the unknown arguments - would start a
second copy of the label UI instead of a worker.

So the packaged exe gets one entry point that dispatches on its first
argument, and _spawn_worker asks it for a worker with

    [sys.executable, "--worker", <job file>]

Everything else falls through to the label UI unchanged, which keeps
`shot-clipper --clips-dir ... --port ...` working the same either way.
"""
import sys

WORKER_FLAG = "--worker"


def worker_argv(job_file) -> list[str]:
    """The argv that starts a job worker for `job_file` in this environment.

    Lives here rather than in jobs.py so the two halves of the contract - who
    emits --worker and who honours it - are one file apart from each other,
    not one process apart.
    """
    from . import paths

    if paths.is_frozen():
        return [sys.executable, WORKER_FLAG, str(job_file)]
    return [sys.executable, "-m", "shot_clipper.label_ui.worker", str(job_file)]


def main() -> None:
    if sys.argv[1:2] == [WORKER_FLAG]:
        # Drop the flag so the worker sees the same argv it would have as
        # `python -m shot_clipper.label_ui.worker <job file>`.
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .label_ui.worker import main as worker_main

        worker_main()
        return

    from .label_ui.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
