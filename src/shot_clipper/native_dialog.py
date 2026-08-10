"""Native file/folder pickers, from whichever shell is actually hosting the UI.

The label UI is a Flask app, so it has no way to open an OS dialog by itself -
a browser can only offer an <input type=file>, which hands back a file's
*contents*, not the path on disk that the detection pipeline needs. So the
UI asked the server to open the dialog, and the server could only do that on
macOS via osascript. On Windows and in Docker there was no picker at all,
leaving the "type or paste a path" field as the only option - which is how a
copied file:// URL or a typo turns into a confusing "not found".

Running inside a desktop window changes that: the window owns a real toolkit
and can open a real dialog. desktop.py registers one here at startup, and the
Flask endpoints call through this module without knowing which shell they're
in. Nothing registers in Docker or a plain browser session, where the
in-app folder browser (/api/browse-dir) remains the fallback.

Every provider returns the same two shapes, so callers don't branch:
    {"path": "/some/file"}   or   {"cancelled": True}
"""
import shutil

from . import external

FILE = "file"
FOLDER = "folder"

_provider = None


def register(provider) -> None:
    """Install the host shell's picker: provider(kind, prompt) -> dict."""
    global _provider
    _provider = provider


def available() -> bool:
    return _provider is not None or _osascript_available()


def describe() -> str:
    if _provider is not None:
        return "native window dialog"
    if _osascript_available():
        return "macOS osascript"
    return "none (use the in-app folder browser)"


def choose(kind: str, prompt: str) -> dict:
    """Open a picker and return the chosen path. Raises RuntimeError when no
    picker exists, which the caller turns into a 400 telling the user to type
    the path instead."""
    if _provider is not None:
        return _provider(kind, prompt)
    if _osascript_available():
        return _osascript_choose(kind, prompt)
    raise RuntimeError("no native file picker available here - type or paste the path "
                       "instead, or use Browse")


def _osascript_available() -> bool:
    return shutil.which("osascript") is not None


def _osascript_choose(kind: str, prompt: str) -> dict:
    """macOS "choose file"/"choose folder". Blocks this request's thread until
    the user responds; app.run(threaded=True) keeps the rest of the app
    responsive meanwhile."""
    verb = "choose file" if kind == FILE else "choose folder"
    type_clause = ' of type {"public.movie"}' if kind == FILE else ""
    safe_prompt = prompt.replace('"', "")
    script = f'POSIX path of ({verb} with prompt "{safe_prompt}"{type_clause})'
    result = external.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        if "User canceled" in result.stderr:
            return {"cancelled": True}
        raise RuntimeError(f"picker failed: {result.stderr.strip()}")
    return {"path": result.stdout.strip()}
