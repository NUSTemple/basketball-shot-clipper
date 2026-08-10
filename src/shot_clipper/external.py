"""Locating and launching the external programs shot-clipper shells out to.

Two problems, both of which only appear once the app is packaged rather than
run from a checkout:

1. ffmpeg/ffprobe were named as the bare strings "ffmpeg"/"ffprobe" in five
   places, so they had to be on PATH. A self-contained install ships its own
   copy instead - the point of packaging is not needing a separate winget
   install - and PATH is exactly what it can't rely on.
2. A child process launched from a windowed (--noconsole) build gets its own
   console window, because there's no console for it to inherit. Every clip
   cut, thumbnail, and probe would flash a black window on screen. A console
   build is unaffected either way, so the flag is safe to always pass.

Resolution order for both tools: an explicit $SHOT_CLIPPER_FFMPEG /
$SHOT_CLIPPER_FFPROBE override, then a copy bundled next to the app, then
PATH. Deliberately not cached: shutil.which is cheap next to the ffmpeg run
it precedes, and caching would make the override untestable without a
cache_clear dance in every fixture.
"""
import os
import shutil
import subprocess
import sys

from . import paths

FFMPEG_ENV = "SHOT_CLIPPER_FFMPEG"
FFPROBE_ENV = "SHOT_CLIPPER_FFPROBE"

# Where a packaged build keeps its own ffmpeg, relative to resource_dir().
BUNDLED_BIN_DIR = "bin"

# CREATE_NO_WINDOW. Not read off the subprocess module: the constant only
# exists on Windows, so referencing it directly would be an AttributeError
# everywhere else.
_CREATE_NO_WINDOW = 0x08000000


def _locate(tool: str, env_var: str) -> str | None:
    """Absolute path to `tool`, or None if it can't be found anywhere."""
    override = os.environ.get(env_var)
    if override:
        return override
    exe = f"{tool}.exe" if sys.platform == "win32" else tool
    bundled = paths.resource_dir() / BUNDLED_BIN_DIR / exe
    if bundled.is_file():
        return str(bundled)
    return shutil.which(tool)


def ffmpeg_exe() -> str:
    """Path to ffmpeg, falling back to the bare name.

    Never None: callers building an argv want something to put in it, and a
    missing ffmpeg should surface as the same "not found" error it always
    did, at the point of the run, not as a TypeError while assembling the
    command. Use have_ffmpeg() to ask whether it exists.
    """
    return _locate("ffmpeg", FFMPEG_ENV) or "ffmpeg"


def ffprobe_exe() -> str:
    return _locate("ffprobe", FFPROBE_ENV) or "ffprobe"


def have_ffmpeg() -> bool:
    return _locate("ffmpeg", FFMPEG_ENV) is not None


def no_window_kwargs() -> dict:
    """subprocess kwargs that keep a child from opening a console window.

    Merge into the call rather than passing creationflags unconditionally -
    on POSIX there is no such argument.
    """
    if sys.platform == "win32":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run for an external tool: no stray console window."""
    return subprocess.run(cmd, **no_window_kwargs(), **kwargs)


def popen(cmd: list[str], **kwargs) -> subprocess.Popen:
    """subprocess.Popen for an external tool: no stray console window."""
    return subprocess.Popen(cmd, **no_window_kwargs(), **kwargs)
