"""even_crop() guards against a bug that does not look like a bug.

The source is chroma-subsampled, so ffmpeg silently snaps an odd crop onto the
chroma grid - ask for width 347 and you get 346. Reading w*h*3 bytes per frame
out of a stream of (w-1)*h*3 byte frames shears every frame progressively.
The pixel histogram still looks normal, so detection quietly degrades instead
of failing. These cases pin the geometry that keeps ffmpeg from rounding.
"""
import subprocess

import pytest

from shot_clipper import external, video_source
from shot_clipper.video_source import even_crop


@pytest.mark.parametrize("roi,frame,expected", [
    # the real case that exposed the bug: odd x, odd width
    ((517, 0, 864, 576), (2688, 1512), (516, 0, 348, 576, 1, 0)),
    # already even - must be left exactly alone
    ((4, 2, 10, 8), (100, 100), (4, 2, 6, 6, 0, 0)),
    # odd on every side
    ((1, 1, 9, 9), (100, 100), (0, 0, 10, 10, 1, 1)),
    # growing outward must not run past the frame
    ((1, 1, 99, 99), (100, 100), (0, 0, 100, 100, 1, 1)),
    # full frame, nothing to do
    ((0, 0, 100, 100), (100, 100), (0, 0, 100, 100, 0, 0)),
])
def test_even_crop_geometry(roi, frame, expected):
    assert even_crop(roi, *frame) == expected


@pytest.mark.parametrize("roi", [
    (517, 0, 864, 576), (1, 1, 9, 9), (3, 7, 101, 203), (0, 0, 1, 1), (99, 99, 100, 100),
])
def test_even_crop_always_even_and_covers_roi(roi):
    x, y, w, h, off_x, off_y = even_crop(roi, 2688, 1512)
    assert x % 2 == 0 and y % 2 == 0, "ffmpeg rounds odd offsets onto the chroma grid"
    assert w % 2 == 0 and h % 2 == 0, "ffmpeg rounds odd sizes onto the chroma grid"
    rx1, ry1, rx2, ry2 = roi
    # the slice the caller takes back out has to land inside what we asked for
    assert x + off_x == rx1 and y + off_y == ry1
    assert off_x + (rx2 - rx1) <= w and off_y + (ry2 - ry1) <= h


# --------------------------------------------------------------------------
# decoder selection
#
# Which hardware decoder gets used decides ~77% of detection's wall time, and
# it is chosen from a string match on `ffmpeg -hwaccels`. These fake that probe
# so the wiring is testable on a machine that has neither an NVIDIA card nor a
# Mac.
# --------------------------------------------------------------------------

CUDA_PROBE = "Hardware acceleration methods:\ncuda\ndxva2\nd3d11va\nqsv\n"
VT_PROBE = "Hardware acceleration methods:\nvideotoolbox\n"
BOTH_PROBE = "Hardware acceleration methods:\ncuda\nvideotoolbox\n"
NEITHER_PROBE = "Hardware acceleration methods:\nvaapi\n"


def _clear(fn):
    """cache_clear if it's still the lru_cached original - a test may have
    swapped in a plain stub, which monkeypatch only restores after this runs."""
    getattr(fn, "cache_clear", lambda: None)()


@pytest.fixture(autouse=True)
def _fresh_decoder(monkeypatch):
    """decoder_kind is lru_cached, so without this every case after the first
    would silently read the first one's answer."""
    _clear(video_source.decoder_kind)
    monkeypatch.delenv(video_source.DECODER_ENV, raising=False)
    # external._locate consults these before PATH, so a developer who has one
    # set would otherwise see "ffmpeg is missing" tests find one anyway.
    monkeypatch.delenv(external.FFMPEG_ENV, raising=False)
    monkeypatch.delenv(external.FFPROBE_ENV, raising=False)
    yield
    _clear(video_source.decoder_kind)


def _fake_ffmpeg(monkeypatch, probe_stdout=None, *, found=True, raises=None):
    """Pretend ffmpeg exists and reports probe_stdout. Returns a list that
    records probe invocations, so a test can assert it was never run."""
    calls = []
    # external is where the lookup lives now - video_source only asks it
    # whether ffmpeg exists (see external.have_ffmpeg).
    monkeypatch.setattr(external.shutil, "which",
                        lambda _name: "/usr/bin/ffmpeg" if found else None)

    def fake_run(*args, **kwargs):
        calls.append(args)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(args, 0, stdout=probe_stdout, stderr="")

    monkeypatch.setattr(video_source.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("probe,expected", [
    (CUDA_PROBE, "nvdec"),          # NVIDIA regression guard
    (VT_PROBE, "videotoolbox"),
    (BOTH_PROBE, "nvdec"),          # pins precedence - cuda must win
    (NEITHER_PROBE, "cpu"),
    ("", "cpu"),
])
def test_decoder_autodetect(monkeypatch, probe, expected):
    _fake_ffmpeg(monkeypatch, probe)
    assert video_source.decoder_kind() == expected


@pytest.mark.parametrize("boom", [OSError("nope"),
                                  subprocess.TimeoutExpired("ffmpeg", 15)])
def test_probe_failure_falls_back_to_cpu(monkeypatch, boom):
    _fake_ffmpeg(monkeypatch, raises=boom)
    assert video_source.decoder_kind() == "cpu"


def test_no_ffmpeg_means_opencv(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, CUDA_PROBE, found=False)
    assert video_source.decoder_kind() is None
    assert calls == [], "should not probe when ffmpeg isn't installed"


def test_opencv_override_wins_over_available_cuda(monkeypatch):
    _fake_ffmpeg(monkeypatch, CUDA_PROBE)
    monkeypatch.setenv(video_source.DECODER_ENV, "opencv")
    assert video_source.decoder_kind() is None


@pytest.mark.parametrize("value", ["videotoolbox", "VideoToolbox", "NVDEC", "cpu"])
def test_explicit_choice_skips_the_probe(monkeypatch, value):
    """Forcing a decoder must not depend on how the local ffmpeg self-reports."""
    calls = _fake_ffmpeg(monkeypatch, NEITHER_PROBE)
    monkeypatch.setenv(video_source.DECODER_ENV, value)
    assert video_source.decoder_kind() == value.lower()
    assert calls == [], "explicit choice should not run the probe"


def test_unrecognised_value_falls_through_to_autodetect(monkeypatch):
    _fake_ffmpeg(monkeypatch, CUDA_PROBE)
    monkeypatch.setenv(video_source.DECODER_ENV, "videotoolbx")  # typo
    assert video_source.decoder_kind() == "nvdec"


@pytest.mark.parametrize("probe,expected", [
    (CUDA_PROBE, "ffmpeg/NVDEC (GPU)"),
    (VT_PROBE, "ffmpeg/VideoToolbox (GPU)"),
    (NEITHER_PROBE, "ffmpeg (CPU)"),
])
def test_describe(monkeypatch, probe, expected):
    _fake_ffmpeg(monkeypatch, probe)
    assert video_source.describe() == expected


def test_describe_without_ffmpeg(monkeypatch):
    _fake_ffmpeg(monkeypatch, found=False)
    assert video_source.describe() == "OpenCV (CPU)"


def test_describe_tolerates_an_unknown_kind(monkeypatch):
    """describe() had no callers until the decoder was surfaced in output; a
    KeyError here would take down whatever is logging it."""
    monkeypatch.setattr(video_source, "decoder_kind", lambda: "some_future_codec")
    assert "some_future_codec" in video_source.describe()


# --------------------------------------------------------------------------
# ffmpeg argv
# --------------------------------------------------------------------------

VF = r"select=not(mod(n\,2)),crop=348:576:516:0"


@pytest.mark.parametrize("kind,flag", [("nvdec", "cuda"),
                                       ("videotoolbox", "videotoolbox")])
def test_hwaccel_flag_precedes_input(kind, flag):
    """-hwaccel after -i is silently ignored by ffmpeg, so ordering is load
    bearing and easy to break."""
    cmd = video_source._build_cmd(kind, "V.MP4", VF)
    assert cmd[cmd.index("-hwaccel") + 1] == flag
    assert cmd.index("-hwaccel") < cmd.index("-i")


def test_cpu_kind_passes_no_hwaccel():
    assert "-hwaccel" not in video_source._build_cmd("cpu", "V.MP4", VF)


@pytest.mark.parametrize("kind", ["nvdec", "videotoolbox", "cpu"])
def test_output_format_is_identical_across_kinds(kind):
    """Only the decode flag may vary - the raw stream this reads back is
    parsed as tightly packed w*h*3 bgr24, and -fps_mode passthrough is what
    stops the muxer duplicating frames back up to the source rate."""
    cmd = video_source._build_cmd(kind, "V.MP4", VF)
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[-1] == "-"
    assert cmd[cmd.index("-vf") + 1] == VF
