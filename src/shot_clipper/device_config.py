"""Which compute device inference runs on, and the knobs that go with it.

Device selection is CUDA > MPS > CPU. Note this only covers *inference*, which
profiling showed is the cheap end of detection - decode is ~77% of the wall
time and is selected separately in video_source.py. On Apple Silicon in
particular, MPS here plus VideoToolbox there are what make a Mac fast; either
alone leaves most of the time on the table.

SHOT_CLIPPER_DEVICE (cuda|mps|cpu) and SHOT_CLIPPER_BATCH_SIZE override the
automatic choices - the escape hatches for when a backend misbehaves, matching
SHOT_CLIPPER_DECODER in video_source.py.
"""
import os
import subprocess
import sys
from functools import lru_cache

DEVICE_ENV = "SHOT_CLIPPER_DEVICE"
BATCH_SIZE_ENV = "SHOT_CLIPPER_BATCH_SIZE"
VALID_DEVICES = ("cuda", "mps", "cpu")


def _autodetect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # getattr: torch builds before 1.12 have no backends.mps at all, and
        # an AttributeError here would escape the ImportError guard
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_device() -> str:
    """Best available device: cuda > mps > cpu, or $SHOT_CLIPPER_DEVICE.

    A forced device is validated rather than obeyed blindly - asking for cuda
    on a Mac would otherwise fail deep inside torch with an opaque error, long
    after the point where we could say something useful. An unrecognised value
    is ignored in favour of autodetection, matching video_source.decoder_kind.
    """
    forced = os.environ.get(DEVICE_ENV, "").strip().lower()
    detected = _autodetect_device()
    if forced in VALID_DEVICES:
        if forced == "cpu" or forced == detected:
            return forced
        print(f"{DEVICE_ENV}={forced}: GPU requested but unavailable - falling back to CPU")
        return "cpu"
    if forced:
        print(f"ignoring {DEVICE_ENV}={forced!r} (expected one of {', '.join(VALID_DEVICES)})")
    return detected


def get_optimal_batch_size(device: str | None = None) -> int:
    """Frames per inference call. $SHOT_CLIPPER_BATCH_SIZE overrides.

    Lower it if the GPU runs out of memory (an 8GB M1 or a small NVIDIA card);
    the defaults assume roughly 12GB. Note this is a speed/memory knob, not a
    free one - see detect_ball_centers_batch.
    """
    override = os.environ.get(BATCH_SIZE_ENV)
    if override:
        try:
            n = int(override)
        except ValueError:
            n = 0
        if n > 0:
            return n
        print(f"ignoring {BATCH_SIZE_ENV}={override!r} (not a positive integer)")

    if device is None:
        device = get_device()
    if device == "cuda":
        return 16
    if device == "mps":
        return 8
    return 4


def warmup_device(model, device: str, imgsz: int | None = None):
    """Run one dummy inference so the first real batch doesn't pay for setup.

    Worth more on MPS than CUDA: Metal compiles shaders on first dispatch,
    where CUDA kernels largely ship precompiled in the wheel.

    The dummy is (H, W, C) - Ultralytics' predict() rejects a batched NCHW
    array outright, which meant the original (1, 3, 640, 640) default made this
    whole function a silent no-op behind its own except clause.

    imgsz should match what production will use (detect_ball_centers_batch
    derives it from the ROI), since both backends specialise per input shape
    and warming up the wrong one warms up nothing.
    """
    if device not in ("cuda", "mps"):
        return

    try:
        import numpy as np
        import torch

        side = imgsz or 640
        dummy = np.zeros((side, side, 3), dtype=np.uint8)
        kwargs = {"device": device, "verbose": False}
        if imgsz:
            kwargs["imgsz"] = imgsz
        with torch.no_grad():
            model.predict(dummy, **kwargs)
        backend = torch.cuda if device == "cuda" else torch.mps
        backend.synchronize()
        backend.empty_cache()
    except Exception as e:  # noqa: BLE001 - warmup is an optimisation, never a hard failure
        print(f"Warmup failed (non-fatal): {e}")


@lru_cache(maxsize=1)
def _mac_chip_name() -> str:
    """"Apple M3 Max", or "" if we can't tell. platform.processor() just says
    "arm" on macOS, which is useless for telling two machines apart."""
    if sys.platform != "darwin":
        return ""
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def device_summary(device: str | None = None) -> str:
    """Human-readable device label, e.g. "CUDA (NVIDIA GeForce RTX 4070)" or
    "MPS (Apple M3 Max)". Called once per job from label_ui/pipeline, so it
    degrades rather than raising or blocking."""
    if device is None:
        device = get_device()

    if device == "cuda":
        try:
            import torch
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001 - a label is never worth failing a job over
            return "CUDA (device unknown)"
    if device == "mps":
        chip = _mac_chip_name()
        return f"MPS ({chip})" if chip else "MPS (Apple Metal)"
    return "CPU"
