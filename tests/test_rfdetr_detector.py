"""RF-DETR detector wiring and the static-spot filter, without loading the
real 128MB checkpoint (the end-to-end check needs weights and footage, so it
runs as `shot-clipper-detect` on a real video, not here)."""
from types import SimpleNamespace

import numpy as np
import pytest

from shot_clipper import detect_shots, inference
from shot_clipper.detect_shots import detect_ball_candidates_batch, suppress_static
from shot_clipper.inference import Detection, RFDETRDetector


def test_pth_is_rfdetr_and_pt_is_yolo():
    assert inference.is_rfdetr_weights("models/rf-detr-basketball.pth")
    assert not inference.is_rfdetr_weights("models/yolov8l.pt")


def test_default_detector_is_rfdetr_and_filter_stays_on_yolo():
    """The trained filter was fit on YOLO trajectories - swapping its feature
    extractor along with the detector would silently invalidate it."""
    assert inference.is_rfdetr_weights(inference.RFDETR_WEIGHTS)
    assert not inference.is_rfdetr_weights(inference.FILTER_FEATURE_WEIGHTS)


def test_missing_non_default_weights_raise_instead_of_downloading(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.ensure_rfdetr_weights(tmp_path / "someone-elses.pth")


def test_existing_weights_are_used_as_is(tmp_path):
    path = tmp_path / "rf-detr-basketball.pth"
    path.write_bytes(b"x")
    assert inference.ensure_rfdetr_weights(path) == path


class _FakePreds:
    def __init__(self, rows):
        self.xyxy = np.array([r[:4] for r in rows], dtype=np.float32).reshape(-1, 4)
        self.confidence = np.array([r[4] for r in rows], dtype=np.float32)
        self.class_id = np.array([r[5] for r in rows])


def _fake_rfdetr(per_frame_rows, multiple=32):
    det = RFDETRDetector.__new__(RFDETRDetector)
    det.device = "cpu"
    det.ball_class = 0
    det._multiple = multiple
    calls = []

    def predict(images, threshold, shape):
        calls.append(shape)
        return [_FakePreds(rows) for rows in per_frame_rows[:len(images)]]

    det.model = SimpleNamespace(predict=predict)
    return det, calls


def test_rfdetr_detect_batch_keeps_only_requested_classes_as_plain_floats():
    rows = [[(10, 20, 30, 40, 0.9, 0), (0, 0, 5, 5, 0.8, 2)]]  # ball, rim
    det, _ = _fake_rfdetr(rows)
    out = det.detect_batch([np.zeros((100, 100, 3), np.uint8)], classes=(0,), conf=0.3, imgsz=0)
    assert out == [[Detection(pytest.approx(0.9), 10.0, 20.0, 30.0, 40.0)]]
    assert all(type(v) is float for v in out[0][0])  # json-serializable downstream


def test_rfdetr_shape_rounds_up_to_the_backbone_multiple():
    det, calls = _fake_rfdetr([[]])
    det.detect_batch([np.zeros((455, 343, 3), np.uint8)], classes=(0,), conf=0.3, imgsz=0)
    assert calls == [(480, 352)]


def test_candidates_use_the_detectors_own_ball_class_and_threshold():
    seen = {}

    class Fake:
        ball_class = 0
        ball_conf = 0.3

        def detect_batch(self, frames, classes, conf, imgsz):
            seen.update(classes=classes, conf=conf)
            return [[Detection(0.4, 0, 0, 10, 10), Detection(0.9, 50, 50, 60, 60)]]

    out = detect_ball_candidates_batch(Fake(), [np.zeros((100, 100, 3), np.uint8)])
    assert seen == {"classes": (0,), "conf": 0.3}
    assert [c[0] for c in out[0]] == [0.9, 0.4]  # best first
    assert out[0][0][1:] == pytest.approx((0.55, 0.55))


def test_suppress_static_skips_a_spot_that_never_moves():
    frame_w, frame_h = 1000, 1000
    net = (0.12, 0.5, 0.5)  # same place every frame
    candidates = [[net] for _ in range(100)]
    candidates[40] = [net, (0.9, 0.3, 0.2)]  # a real ball, scored below the net spot
    centers = suppress_static(candidates, frame_w, frame_h)
    assert centers[40] == (0.3, 0.2)
    assert centers[0] is None


def test_suppress_static_keeps_a_ball_seen_in_a_few_frames():
    candidates = [[] for _ in range(100)]
    candidates[10] = [(0.8, 0.4, 0.4)]
    candidates[11] = [(0.8, 0.4, 0.4)]  # 2% of frames - not over the limit
    assert suppress_static(candidates, 1000, 1000)[10] == (0.4, 0.4)
    assert detect_shots.STATIC_MAX_FRAC * len(candidates) == 2


def test_detector_kind_reports_rfdetr_for_the_default_weights():
    """The job status shows this; it used to say "torch" while RF-DETR ran."""
    assert inference.detector_kind(inference.RFDETR_WEIGHTS) == "rfdetr"
    assert inference.detector_kind(inference.FILTER_FEATURE_WEIGHTS) in inference.VALID_BACKENDS
