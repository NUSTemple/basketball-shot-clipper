"""The two constants that carry detection, and why they are what they are.

Measured on DJI_20260822170858_0002_D against 14 hand-labelled makes, after
a camera angle change dropped the ball to ~37px across:

    scale 1.0, conf 0.10 (the old defaults)   3/14 makes   recall 0.21
    scale 2.0, conf 0.02 (these defaults)    14/14 makes   recall 1.00

These tests don't re-derive that - it needs a GPU and the source footage.
They pin the wiring the result depends on, which is what would silently
regress: that the scale actually reaches the inference size, and that
nothing resets the constants to values the measurement disowns.
"""
import numpy as np
import pytest

from shot_clipper import detect_shots


def frame(width=357, height=536):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_scale_multiplies_the_inference_size():
    """The whole fix is that the crop reaches YOLO enlarged."""
    native = detect_shots.batch_imgsz(frame(), scale=1.0)
    doubled = detect_shots.batch_imgsz(frame(), scale=2.0)
    assert doubled == pytest.approx(native * 2, abs=32)


def test_imgsz_stays_on_the_stride():
    """YOLO wants a multiple of 32; an off-stride size gets silently
    rounded somewhere less visible."""
    for scale in (1.0, 1.5, 2.0, 3.0):
        assert detect_shots.batch_imgsz(frame(), scale=scale) % 32 == 0


def test_imgsz_follows_the_longest_side():
    assert detect_shots.batch_imgsz(frame(357, 536)) == \
           detect_shots.batch_imgsz(frame(536, 357))


def test_defaults_are_the_measured_ones():
    """A regression here is a silent 5x loss of recall, which nothing else
    in the suite would notice."""
    assert detect_shots.BALL_IMGSZ_SCALE == 2.0
    assert detect_shots.BALL_CONF_THRESHOLD == 0.02


def test_run_detection_exposes_both_knobs():
    """compare_backends and any future sweep tune these per call rather
    than by reaching in and mutating module state."""
    import inspect

    params = inspect.signature(detect_shots.run_detection).parameters
    assert params["imgsz_scale"].default == detect_shots.BALL_IMGSZ_SCALE
    assert params["ball_conf"].default == detect_shots.BALL_CONF_THRESHOLD


def test_detect_batch_receives_the_scaled_size(monkeypatch):
    """The end-to-end wiring: a scale passed to the batch helper has to
    arrive at the detector as a bigger imgsz, not be dropped en route."""
    seen = {}

    class FakeDetector:
        def detect_batch(self, frames, classes, conf, imgsz):
            seen["imgsz"] = imgsz
            seen["conf"] = conf
            return [[] for _ in frames]

    detect_shots.detect_ball_centers_batch(
        FakeDetector(), [frame()], imgsz_scale=2.0, conf=0.02,
        full_size=(1920, 1080))

    assert seen["imgsz"] == detect_shots.batch_imgsz(frame(), 2.0)
    assert seen["conf"] == 0.02
