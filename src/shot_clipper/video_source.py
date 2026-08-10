"""Hardware-decoded frame source for detection.

Detection is decode-bound, not inference-bound. Measured on a 2688x1512
HEVC drone clip with an RTX 4070: OpenCV's CPU decode cost ~24ms per sampled
frame while the YOLO pass over that same (cropped) frame took ~4ms - so ~85%
of detection's wall time was the GPU sitting idle waiting for the CPU, with
the card's dedicated NVDEC engine reading 0% the whole time. ffmpeg decoding
the same file measured 386fps with `-hwaccel cuda` vs 134fps without.

So frames come from an ffmpeg pipe here instead, with the crop applied inside
ffmpeg - only the hoop ROI crosses the pipe (~600KB/frame at 347x576 rather
than 12MB for a full 4K frame).

Sampling is deliberately identical to the OpenCV loop it replaces: the select
filter keeps every step-th SOURCE frame - the same frames grab()/retrieve()
kept - so timestamps stay exactly i*step/src_fps and detection results don't
move. Only the path the pixels take is different.

The same argument applies to Apple Silicon, where the hardware decoder is
VideoToolbox rather than NVDEC - and it matters more there, because MPS
already accelerates inference (the cheap 23%) while decode was still entirely
on the CPU. Which one gets used is decided purely by what `ffmpeg -hwaccels`
reports, never by the CPU architecture: Intel Macs have VideoToolbox but no
MPS, so decode and inference capability are detected independently.

Set SHOT_CLIPPER_DECODER=opencv to force the old cv2.VideoCapture path, or
=cpu to keep the ffmpeg pipe but skip hardware decode. Either one is the
rollback if hardware decode ever produces different pixels than software.
"""
import os
import subprocess
from functools import lru_cache

from . import external

DECODER_ENV = "SHOT_CLIPPER_DECODER"  # auto (default) | nvdec | videotoolbox | cpu | opencv

# ffmpeg's -hwaccel name per decoder kind. "cpu" is absent on purpose: no flag.
HWACCEL_FLAG = {"nvdec": "cuda", "videotoolbox": "videotoolbox"}


@lru_cache(maxsize=1)
def decoder_kind() -> str | None:
    """Which ffmpeg decode path to use: "nvdec", "videotoolbox", "cpu", or
    None for OpenCV."""
    choice = os.environ.get(DECODER_ENV, "auto").lower()
    if choice == "opencv":
        return None
    if not external.have_ffmpeg():
        return None
    # an explicit choice skips the probe entirely: someone forcing a decoder
    # shouldn't have the result depend on how their ffmpeg self-reports
    if choice in ("nvdec", "videotoolbox", "cpu"):
        return choice
    try:
        out = external.run([external.ffmpeg_exe(), "-v", "quiet", "-hwaccels"],
                           capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    accels = set(out.split())
    # cuda first, so this returns exactly what it did before on any NVIDIA host
    if "cuda" in accels:
        return "nvdec"
    # Not gated on platform.machine(): Intel Macs have VideoToolbox but no MPS.
    # Decode capability comes from this probe, inference capability from
    # torch.backends.mps - keeping them independent is what makes Intel Macs
    # get hardware decode without pretending they can run Metal inference.
    if "videotoolbox" in accels:
        return "videotoolbox"
    return "cpu"


def describe() -> str:
    kind = decoder_kind()
    # .get, not [], so an unrecognised kind degrades to a label instead of
    # taking down whatever is logging it
    return {"nvdec": "ffmpeg/NVDEC (GPU)",
            "videotoolbox": "ffmpeg/VideoToolbox (GPU)",
            "cpu": "ffmpeg (CPU)",
            None: "OpenCV (CPU)"}.get(kind, f"ffmpeg/{kind}")


def even_crop(roi, frame_w: int, frame_h: int):
    """Grow roi outward to even offsets and even size, and report where the
    original roi sits inside it.

    Not cosmetic. The source is chroma-subsampled (yuv420), so ffmpeg snaps an
    odd crop onto the chroma grid: asking for x=517 w=347 silently yields
    w=346. Reading w*h*3 bytes per frame from a stream of (w-1)*h*3 byte
    frames shears every frame progressively - the pixel histogram still looks
    right, so it corrupts detection without looking broken. Feeding ffmpeg
    only even geometry removes the rounding entirely; the caller slices the
    exact roi back out afterwards.

    Returns (x, y, w, h, off_x, off_y).
    """
    rx1, ry1, rx2, ry2 = roi
    x = max(0, rx1 - (rx1 % 2))
    y = max(0, ry1 - (ry1 % 2))
    x2 = min(frame_w, rx2 + (rx2 % 2))
    y2 = min(frame_h, ry2 + (ry2 % 2))
    return x, y, x2 - x, y2 - y, rx1 - x, ry1 - y


def _build_cmd(kind: str, video_path, vf: str) -> list[str]:
    """The ffmpeg argv for one decode kind. Split out from iter_frames so the
    hwaccel wiring can be tested without the hardware it names."""
    cmd = [external.ffmpeg_exe(), "-nostdin", "-v", "error"]
    flag = HWACCEL_FLAG.get(kind)
    if flag:
        # has to precede -i; placed after the input ffmpeg silently ignores it
        cmd += ["-hwaccel", flag]
    cmd += ["-i", str(video_path), "-vf", vf,
            # without this, the rawvideo muxer's CFR default would duplicate
            # frames back up to the source rate and undo the select
            "-fps_mode", "passthrough",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    return cmd


def iter_frames(video_path, step: int, src_fps: float, roi, frame_w: int, frame_h: int):
    """Yield (timestamp_sec, BGR ndarray) for every step-th source frame,
    cropped to roi. Raises RuntimeError if ffmpeg produced nothing or emitted
    an unexpected frame size, so the caller can fall back rather than feed the
    detector garbage."""
    import numpy as np

    kind = decoder_kind()
    if kind is None:
        raise RuntimeError("no ffmpeg decoder available")

    rx1, ry1, rx2, ry2 = roi
    want_w, want_h = rx2 - rx1, ry2 - ry1
    cx, cy, w, h, off_x, off_y = even_crop(roi, frame_w, frame_h)
    # the comma inside mod() has to be escaped or ffmpeg reads it as a filter
    # separator
    vf = f"select=not(mod(n\\,{step})),crop={w}:{h}:{cx}:{cy}"

    cmd = _build_cmd(kind, video_path, vf)

    nbytes = w * h * 3
    proc = external.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          bufsize=nbytes)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                # A partial trailing frame means our idea of the frame size
                # disagrees with ffmpeg's, and every frame we already yielded
                # was sheared. Never let that pass as a normal end-of-stream.
                if buf:
                    raise RuntimeError(
                        f"ffmpeg emitted {n * nbytes + len(buf)} bytes, not a multiple of the "
                        f"expected {w}x{h}x3 frame - refusing to trust these frames")
                break
            # .copy() is not optional: frombuffer over `bytes` is read-only,
            # and callers hold a whole batch of these at once, so they each
            # need to own their memory
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield (n * step) / src_fps, frame[off_y:off_y + want_h, off_x:off_x + want_w].copy()
            n += 1
        if n == 0:
            err = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg produced no frames: {err or 'no error output'}")
    finally:
        # the consumer can stop early (features.py reads part of a clip), and
        # an abandoned ffmpeg would keep a handle on the video file
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
