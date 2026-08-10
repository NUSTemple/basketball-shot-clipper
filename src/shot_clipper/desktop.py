"""The label UI as a desktop window instead of a browser tab.

Same Flask app, unchanged - it just gets served to a WebView2 window that
this process owns rather than to whatever browser the launcher happened to
open. What that buys, beyond looking like an application: a taskbar entry
that isn't Chrome, no stray tab left behind when the app exits, no
"127.0.0.1:5050 refused to connect" if the shortcut is clicked twice, and a
real OS file picker (see native_dialog.py) on every platform rather than
macOS only.

Deliberately still a local HTTP server rather than something bundled into
the window's own protocol: the front end already talks to it that way, job
workers are separate processes that need the same endpoints, and keeping
`shot-clipper --server` able to serve a plain browser is what lets Docker
and headless use carry on unchanged.

Two details that matter for packaging:

- The port is ephemeral, not 5050. A window addresses itself, so there's no
  reason to squat on a fixed port that a second copy - or an unrelated dev
  server - would collide with.
- Flask runs in a daemon thread and the window owns the main thread, because
  every GUI toolkit here requires that. Closing the window ends the process;
  a detection job, being a separate process by design (see jobs.py), keeps
  running and reappears in Job Status on next launch.
"""
import argparse
import socket
import threading
import time
from pathlib import Path

from . import native_dialog, paths

WINDOW_TITLE = "shot-clipper"
WINDOW_SIZE = (1280, 860)
WINDOW_MIN_SIZE = (900, 600)
# how long to wait for Flask to start accepting before opening the window on
# a blank page
SERVER_TIMEOUT_SEC = 15.0


def available() -> bool:
    """Is a window toolkit importable? False means fall back to server mode."""
    import importlib.util

    try:
        return importlib.util.find_spec("webview") is not None
    except (ImportError, ValueError):
        return False


def _free_port() -> int:
    """Ask the OS for an unused port. Small race between closing this socket
    and Flask binding it, which is worth it to avoid a fixed-port collision -
    the alternative is failing to start at all when 5050 is taken."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int, timeout: float = SERVER_TIMEOUT_SEC) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _serve(app, port: int) -> None:
    # threaded=True: a native picker blocks its request thread until the user
    # responds, and without this that would freeze clip loading, labeling and
    # job polling for as long as the dialog is open.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True,
            use_reloader=False)


def _make_picker(window):
    """Bridge native_dialog's contract to pywebview's file dialog."""
    import webview

    def pick(kind: str, prompt: str) -> dict:
        if kind == native_dialog.FOLDER:
            result = window.create_file_dialog(webview.FOLDER_DIALOG)
        else:
            result = window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=False,
                file_types=("Video files (*.mp4;*.mov;*.MP4;*.MOV)", "All files (*.*)"))
        # pywebview returns None when cancelled, otherwise a tuple of paths
        # even for a single selection
        if not result:
            return {"cancelled": True}
        return {"path": result[0] if isinstance(result, (list, tuple)) else result}

    return pick


def run(clips_dir: Path | None = None, port: int | None = None) -> None:
    """Start the server and open the window. Returns when the window closes."""
    import webview

    from .label_ui import app as label_app
    from .label_ui import jobs

    clips_dir = (clips_dir or paths.default_clips_dir()).resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)
    label_app.app.config["CLIPS_DIR"] = clips_dir
    port = port or _free_port()

    print(f"labeling clips from: {clips_dir}")
    resumed = jobs.kick_queue()
    if resumed:
        print(f"resuming queued job: {resumed}")

    thread = threading.Thread(target=_serve, args=(label_app.app, port), daemon=True)
    thread.start()
    if not _wait_for_server(port):
        raise SystemExit(f"the label UI did not start listening on port {port} in "
                          f"{SERVER_TIMEOUT_SEC:.0f}s")

    window = webview.create_window(
        WINDOW_TITLE, f"http://127.0.0.1:{port}",
        width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE, text_select=True)
    native_dialog.register(_make_picker(window))
    print(f"opening window on http://127.0.0.1:{port}")
    webview.start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", type=Path, default=None,
                         help=f"default: {paths.default_clips_dir()} "
                              f"(or ${paths.CLIPS_DIR_ENV})")
    parser.add_argument("--port", type=int, default=None,
                         help="default: an unused port picked at startup")
    args = parser.parse_args()
    run(clips_dir=args.clips_dir, port=args.port)


if __name__ == "__main__":
    main()
