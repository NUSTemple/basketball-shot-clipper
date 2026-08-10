"""The ONNX backend's pre/post-processing, which is where it can silently
disagree with torch.

The weights are the same network - what surrounds them isn't. Letterbox
geometry, channel order, normalisation and the box rescale each have to match
what ultralytics does, and getting one subtly wrong degrades ball recall
while every intermediate value still looks plausible. That's the same failure
mode video_source.even_crop exists to prevent, and it's why these test
geometry directly rather than only checking that inference runs.

The end-to-end agreement check against a real torch model lives in
shot_clipper.compare_backends (needs weights and footage, so not a unit test).
"""
import numpy as np
import pytest

from shot_clipper import inference
from shot_clipper.inference import Detection, _decode, _letterbox


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(inference.BACKEND_ENV, raising=False)


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [("torch", "torch"), ("onnx", "onnx"),
                                             ("ONNX", "onnx"), ("  torch ", "torch")])
def test_env_selects_backend(monkeypatch, value, expected):
    monkeypatch.setenv(inference.BACKEND_ENV, value)
    assert inference.resolve_backend() == expected


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv(inference.BACKEND_ENV, "onnx")
    assert inference.resolve_backend("torch") == "torch"


def test_nonsense_backend_falls_through_to_autodetect(monkeypatch, capsys):
    monkeypatch.setenv(inference.BACKEND_ENV, "tensorflow")
    assert inference.resolve_backend() in inference.VALID_BACKENDS
    assert "ignoring" in capsys.readouterr().out


def test_onnx_path_sits_beside_the_pt():
    assert inference.onnx_path_for("/models/yolov8l.pt").name == "yolov8l.onnx"


# --------------------------------------------------------------------------
# letterbox
# --------------------------------------------------------------------------

def test_letterbox_pads_to_stride_not_to_a_square():
    """Squaring a tall narrow hoop ROI would scale the ball down and cost
    recall. ultralytics pads to the next stride multiple for a batch of
    same-shaped frames; this has to do the same."""
    frame = np.zeros((576, 347, 3), dtype=np.uint8)
    tensor, (ratio, pad_x, pad_y) = _letterbox(frame, 576)

    assert ratio == 1.0, "a frame already at imgsz must not be rescaled"
    assert tensor.shape[1] == 576, "height is already a stride multiple"
    assert tensor.shape[2] == 352, "width padded 347 -> 352, not to 576"
    assert tensor.shape[2] % inference.STRIDE == 0


def test_letterbox_output_is_chw_rgb_and_normalized():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR
    tensor, _ = _letterbox(frame, 64)

    assert tensor.shape == (3, 64, 64), "channels first"
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0, "must be scaled to 0..1"
    # BGR blue must land in the LAST channel once converted to RGB
    assert tensor[2].max() == pytest.approx(1.0)
    assert tensor[0].max() == pytest.approx(0.0)


def test_letterbox_fill_is_ultralytics_grey():
    """The pad value is part of what the network sees; black padding is a
    different input than grey padding."""
    frame = np.zeros((576, 347, 3), dtype=np.uint8)  # 347 -> 352 needs 5px of pad
    tensor, (_, pad_x, _) = _letterbox(frame, 576)
    assert pad_x > 0, "this frame width must actually need padding"
    padded_column = tensor[:, :, 0]
    assert padded_column.min() == pytest.approx(inference.LETTERBOX_COLOR / 255.0, abs=1e-6)
    assert tensor[:, :, pad_x + 1].max() == pytest.approx(0.0), "content is still black"


def test_letterbox_downscales_a_frame_larger_than_imgsz():
    frame = np.zeros((1280, 640, 3), dtype=np.uint8)
    tensor, (ratio, _, _) = _letterbox(frame, 640)
    assert ratio == pytest.approx(0.5)
    assert tensor.shape[1] == 640


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------

def _fake_output(boxes_scores, n_classes=80, n_anchors=None):
    """Build a (1, 4+n_classes, anchors) YOLOv8-shaped output.

    boxes_scores: list of ((cx, cy, w, h), {class_index: score}).
    """
    n_anchors = n_anchors or len(boxes_scores)
    out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
    for i, (box, scores) in enumerate(boxes_scores):
        out[0, :4, i] = box
        for cls, score in scores.items():
            out[0, 4 + cls, i] = score
    return out


def test_decode_maps_a_box_back_through_the_letterbox():
    """A box the network reports in padded input space has to come back in
    source-frame pixels, or every centre is offset by the padding."""
    ratio, pad_x, pad_y = 0.5, 10.0, 4.0
    out = _fake_output([((110.0, 54.0, 20.0, 20.0), {32: 0.9})])

    dets = _decode(out, (ratio, pad_x, pad_y), classes=(32,), conf=0.1,
                   source_shape=(1000, 1000))[0]

    assert len(dets) == 1
    # centre 110 -> (110 - 10)/0.5 = 200 in source x; 54 -> (54 - 4)/0.5 = 100
    assert (dets[0].x1 + dets[0].x2) / 2 == pytest.approx(200.0)
    assert (dets[0].y1 + dets[0].y2) / 2 == pytest.approx(100.0)
    assert dets[0].x2 - dets[0].x1 == pytest.approx(40.0), "width scales too"


def test_decode_filters_by_class_and_confidence():
    out = _fake_output([
        ((10.0, 10.0, 4.0, 4.0), {32: 0.9}),   # wanted class, over threshold
        ((20.0, 20.0, 4.0, 4.0), {32: 0.05}),  # wanted class, under threshold
        ((30.0, 30.0, 4.0, 4.0), {0: 0.99}),   # a person, not a ball
    ])
    dets = _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1,
                   source_shape=(100, 100))[0]

    assert [round(d.conf, 2) for d in dets] == [0.9]


def test_decode_keeps_every_box_so_the_caller_can_take_the_max():
    """No NMS here by design (see inference.py) - correctness depends on the
    caller getting the top-scoring box, which NMS would never have removed."""
    out = _fake_output([
        ((10.0, 10.0, 4.0, 4.0), {32: 0.4}),
        ((11.0, 11.0, 4.0, 4.0), {32: 0.8}),  # heavy overlap, higher score
    ])
    dets = _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1,
                   source_shape=(100, 100))[0]

    assert len(dets) == 2
    assert max(d.conf for d in dets) == pytest.approx(0.8)


def test_decode_clips_to_the_source_frame():
    """A ball half outside the ROI would otherwise report a centre off-frame."""
    out = _fake_output([((5.0, 5.0, 40.0, 40.0), {32: 0.9})])
    dets = _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1,
                   source_shape=(50, 50))[0]

    assert dets[0].x1 >= 0 and dets[0].y1 >= 0
    assert dets[0].x2 <= 50 and dets[0].y2 <= 50


def test_decode_handles_a_frame_with_nothing_in_it():
    out = _fake_output([((10.0, 10.0, 4.0, 4.0), {32: 0.01})])
    assert _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1,
                   source_shape=(100, 100)) == [[]]


def test_decode_rejects_an_out_of_range_class():
    """Asking for COCO class 32 from a model that wasn't trained on 80
    classes must say so, not read a neighbouring channel as a score."""
    out = np.zeros((1, 4 + 10, 5), dtype=np.float32)
    with pytest.raises(RuntimeError, match="out of range"):
        _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1, source_shape=(100, 100))


def test_decode_rejects_an_unexpected_output_rank():
    with pytest.raises(RuntimeError, match="unexpected ONNX output shape"):
        _decode(np.zeros((84, 100), dtype=np.float32), (1.0, 0.0, 0.0),
                classes=(32,), conf=0.1, source_shape=(100, 100))


def test_decode_returns_one_entry_per_frame_in_the_batch():
    out = np.zeros((3, 84, 5), dtype=np.float32)
    out[1, 4 + 32, 0] = 0.9  # only the middle frame sees a ball
    out[1, :4, 0] = (10.0, 10.0, 4.0, 4.0)

    per_frame = _decode(out, (1.0, 0.0, 0.0), classes=(32,), conf=0.1,
                        source_shape=(100, 100))
    assert [len(f) for f in per_frame] == [0, 1, 0]


def test_detection_is_ordered_conf_first():
    """detect_ball_centers_batch compares det.conf and unpacks the corners;
    swapping the field order would silently reorder every box."""
    det = Detection(0.5, 1.0, 2.0, 3.0, 4.0)
    assert det.conf == 0.5
    assert (det.x1, det.y1, det.x2, det.y2) == (1.0, 2.0, 3.0, 4.0)
