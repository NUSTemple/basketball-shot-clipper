"""Low-resolution scrubbing proxies of a source video.

The review step plays the *whole* video so a human can confirm detected
candidates and mark the ones detection missed. The originals are 1080p60 at
~40Mbps (2.5-6GB for 9-18 minutes), which a browser can play but cannot
seek through pleasantly: every seek decodes forward from whatever keyframe
the camera happened to write, and the review loop is almost entirely
seeking.

So review plays a proxy instead: same timeline, same timestamps, ~1/15th
the bytes, with a keyframe every PROXY_GOP_SEC seconds so a seek lands
almost immediately. Nothing downstream is cut from it - final clips are
always cut from the original (see clip_shots.cut_clip), because the proxy
exists to be scrubbed, not to be delivered.

Two encoder details that matter:

- h264_nvenc when the machine has it, libx264 otherwise. This is the one
  place hardware *encoding* is unambiguously right: unlike the final clips
  (whose written pixels feed net_motion's features, hence x264 there), a
  proxy is only ever looked at by a human.
- +faststart, so the moov atom is at the front of the file. Without it a
  browser must fetch the end of the file before it can seek at all, which
  on a 400MB proxy is the whole problem again in miniature.
"""
import re
import subprocess
import tempfile
from pathlib import Path

from . import external, video_source

PROXY_FILENAME = "_proxy.mp4"
PROXY_HEIGHT = 720
# Review doesn't need 60fps, and halving the framerate roughly halves the
# encode time and the file.
PROXY_FPS = 30
# Seek granularity. Shorter means faster seeks and a bigger file; 2s is
# about where a marker-to-marker jump stops feeling like a wait.
PROXY_GOP_SEC = 2
PROXY_CRF = 26  # visually fine at 720p for judging whether a ball went in
PROXY_NVENC_CQ = 28


# DJI cameras record a low-resolution copy of every take alongside the real
# one: same timeline, 720p30 h264 in an mp4 container, ".LRF" extension.
# That is precisely what this module otherwise spends minutes producing, so
# when one is sitting next to the source it gets used instead.
CAMERA_PROXY_SUFFIXES = (".LRF", ".lrf")


def proxy_path(clips_video_dir: Path) -> Path:
    return clips_video_dir / PROXY_FILENAME


def find_camera_proxy(source: Path) -> Path | None:
    """A camera-written low-res sibling of `source`, if there is one.

    Measured against DJI's: 1280x720 at 29.97fps versus the original's
    1080p60, a keyframe every 1.0s (better seek granularity than this
    module targets), and a duration within 21ms of the original - well
    under one frame, so timestamps carry over untouched.

    Worth preferring even where encode time is free, because the originals
    here are HEVC and a browser that can't decode HEVC can't show the
    review view at all.
    """
    for suffix in CAMERA_PROXY_SUFFIXES:
        candidate = source.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def is_current(proxy: Path, source: Path) -> bool:
    """A proxy is reusable if it exists and isn't older than its source."""
    try:
        return proxy.is_file() and proxy.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def remux_command(source: Path, out_path: Path) -> list[str]:
    """Rewrap an already-suitable file as mp4 without re-encoding.

    -c copy so this runs at disk speed and loses nothing, +faststart so the
    moov atom leads (DJI writes it at the end) and a browser can seek
    without first fetching the tail.
    """
    return [external.ffmpeg_exe(), "-y", "-i", str(source),
            "-c", "copy", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(out_path)]


_nvenc_cache: bool | None = None


def _have_nvenc() -> bool:
    """Can this machine actually encode with h264_nvenc right now?

    Deliberately not `ffmpeg -encoders | grep h264_nvenc`: that lists what
    the *build* supports, which is a different question from what the
    installed driver supports. Observed here - ffmpeg lists h264_nvenc,
    then fails at runtime with "Driver does not support the required nvenc
    API version. Required: 13.1 Found: 13.0". The only honest probe is a
    real encode, so do one frame of black and see if it survives.

    Cached: a driver does not appear mid-session, and this costs a process
    spawn each time it is asked.
    """
    global _nvenc_cache
    if _nvenc_cache is None:
        try:
            proc = external.run(
                [external.ffmpeg_exe(), "-v", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=256x144:d=0.1", "-c:v", "h264_nvenc",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=30)
            _nvenc_cache = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _nvenc_cache = False
    return _nvenc_cache


def _encoder_args(use_nvenc: bool) -> list[str]:
    gop = str(PROXY_GOP_SEC * PROXY_FPS)
    if use_nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", str(PROXY_NVENC_CQ), "-g", gop]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(PROXY_CRF),
            "-g", gop, "-keyint_min", gop, "-sc_threshold", "0"]


def build_command(source: Path, out_path: Path, use_nvenc: bool | None = None) -> list[str]:
    """Split out from build() so the wiring is testable without ffmpeg."""
    if use_nvenc is None:
        use_nvenc = _have_nvenc()
    cmd = [external.ffmpeg_exe(), "-y"]
    kind = video_source.decoder_kind()
    flag = video_source.HWACCEL_FLAG.get(kind) if kind else None
    if flag:
        cmd += ["-hwaccel", flag]
    cmd += ["-i", str(source)]
    cmd += ["-vf", f"scale=-2:{PROXY_HEIGHT}", "-r", str(PROXY_FPS)]
    cmd += _encoder_args(use_nvenc)
    # Audio is worth keeping: the net swish is often the clearest signal
    # that a shot went in, especially at 720p.
    cmd += ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", "-nostats", str(out_path)]
    return cmd


_OUT_TIME_RE = re.compile(r"out_time_ms=(\d+)")


def _run_encode(cmd: list[str], partial: Path, progress_cb, cancel_check) -> str | None:
    """Run one ffmpeg encode. Returns None on success, else its stderr tail.

    stderr goes to a temp file rather than DEVNULL so a failure can say
    *why*. Reading two pipes from one process needs threads to avoid
    deadlocking on a full buffer, and stdout is already spoken for by
    -progress.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errfile:
        proc = external.popen(cmd, stdout=subprocess.PIPE, stderr=errfile, text=True)
        try:
            for line in proc.stdout:
                match = _OUT_TIME_RE.search(line)
                if match and progress_cb:
                    progress_cb(int(match.group(1)) / 1_000_000)
                if cancel_check and cancel_check():
                    proc.kill()
                    proc.wait()
                    partial.unlink(missing_ok=True)
                    raise KeyboardInterrupt("proxy build cancelled")
        finally:
            if proc.stdout:
                proc.stdout.close()
        if proc.wait() == 0:
            return None
        errfile.seek(0)
        return "".join(errfile.readlines()[-5:]).strip() or "(no stderr)"


def build(source: Path, out_path: Path, progress_cb=None,
          cancel_check=None) -> Path:
    """Transcode `source` into a scrubbing proxy at `out_path`.

    progress_cb(seconds_done) is called as ffmpeg reports progress.
    cancel_check, if given, is polled while encoding; once truthy the
    encode is killed and the partial file removed.

    A hardware encode that fails is retried in software. _have_nvenc()
    already probes for real rather than trusting `-encoders`, but "the
    driver accepted one frame of black" is still a weaker claim than "the
    driver encoded this video", and falling back beats failing the job over
    an optional convenience file.

    Writes to a .part file and renames on success, so an interrupted run
    can never leave a truncated proxy that is_current() would happily
    accept next time.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".part.mp4")

    attempts = [True, False] if _have_nvenc() else [False]
    error = None
    for use_nvenc in attempts:
        cmd = build_command(source, partial, use_nvenc=use_nvenc)
        error = _run_encode(cmd, partial, progress_cb, cancel_check)
        if error is None:
            partial.replace(out_path)
            return out_path
        partial.unlink(missing_ok=True)

    raise RuntimeError(f"ffmpeg failed to build a proxy for {source.name}: {error}")


def ensure(source: Path, out_path: Path, progress_cb=None,
           cancel_check=None) -> tuple[Path, str]:
    """Make sure a scrubbing proxy for `source` exists at `out_path`.

    Returns (path, how) where `how` is one of "cached", "camera" or
    "encoded" - worth reporting, because the three differ by two orders of
    magnitude in cost and a user watching a progress bar deserves to know
    which one they're waiting for.

    Order is cheapest-first: an existing proxy, then rewrapping the
    camera's own low-res take, then a full transcode.
    """
    if is_current(out_path, source):
        return out_path, "cached"

    camera = find_camera_proxy(source)
    if camera is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(".part.mp4")
        error = _run_encode(remux_command(camera, partial), partial,
                            progress_cb, cancel_check)
        if error is None:
            partial.replace(out_path)
            return out_path, "camera"
        # A camera file that won't rewrap is not worth failing over - fall
        # through and encode one from the original instead.
        partial.unlink(missing_ok=True)

    return build(source, out_path, progress_cb, cancel_check), "encoded"
