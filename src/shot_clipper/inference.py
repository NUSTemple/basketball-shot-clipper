"""Ball detection backends: ultralytics/torch, or ONNX Runtime.

Why two. The torch path is the one every detection result in this repo was
produced and validated with, so it stays the reference. The ONNX path exists
because a packaged build carrying CUDA torch is roughly 3GB and only helps
NVIDIA owners, against ~120MB for onnxruntime-directml, which runs on any
DirectX 12 GPU - see docs/PACKAGING.md. Both are selectable at runtime via
$SHOT_CLIPPER_INFERENCE so the two can be measured against each other on real
footage (shot-clipper-compare-backends) instead of swapped on faith.

Both return the same thing: per frame, a list of Detection in the *input
frame's* pixel coordinates, already filtered to the requested classes and
confidence. Callers do their own geometry on top (see
detect_shots.detect_ball_centers_batch).

No NMS in the ONNX path, deliberately. Every consumer in this codebase keeps
only the single highest-confidence box per frame, and NMS never removes the
top-scoring box - it only drops lower-scoring overlaps of the same object. So
running it would cost time and change nothing. If a caller ever needs the
full box set, that assumption has to be revisited here.
"""
import os
from pathlib import Path
from typing import NamedTuple

from . import paths

BACKEND_ENV = "SHOT_CLIPPER_INFERENCE"  # auto (default) | torch | onnx
VALID_BACKENDS = ("torch", "onnx")

# ultralytics' letterbox fill and stride - matched exactly so the two backends
# see the same pixels, not merely similar ones
LETTERBOX_COLOR = 114
STRIDE = 32


class Detection(NamedTuple):
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def resolve_backend(explicit: str | None = None) -> str:
    """Which backend to use. Explicit argument > $SHOT_CLIPPER_INFERENCE >
    whatever is installed, preferring torch because it's the validated one.

    An unrecognised value is ignored rather than obeyed, matching
    device_config.get_device and video_source.decoder_kind.
    """
    choice = (explicit or os.environ.get(BACKEND_ENV, "auto")).strip().lower()
    if choice in VALID_BACKENDS:
        return choice
    if choice and choice != "auto":
        print(f"ignoring {BACKEND_ENV}={choice!r} (expected one of {', '.join(VALID_BACKENDS)})")
    if _module_available("ultralytics"):
        return "torch"
    if _module_available("onnxruntime"):
        return "onnx"
    return "torch"  # let the import error name the missing dependency


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def onnx_path_for(weights) -> Path:
    """The .onnx sitting alongside a .pt of the same name."""
    return Path(weights).with_suffix(".onnx")


def load_detector(weights=None, device: str | None = None, backend: str | None = None):
    """Build the detector for `backend`, defaulting to the resolved one.

    weights is a bare filename ("yolov8l.pt"), an explicit path, or None for
    the project default; the ONNX backend swaps the suffix to .onnx and
    expects the exported file next to it (shot-clipper-export-onnx).
    """
    from .device_config import get_device

    kind = resolve_backend(backend)
    device = device or get_device()
    pt_path = paths.find_model(weights) if weights else paths.detect_weights()
    if kind == "onnx":
        return OnnxDetector(onnx_path_for(pt_path), device=device)
    return TorchDetector(pt_path, device=device)


class TorchDetector:
    """ultralytics YOLO - the reference implementation."""

    kind = "torch"

    def __init__(self, weights, device: str = "cpu"):
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

    def warmup(self, imgsz: int | None = None) -> None:
        from .device_config import warmup_device

        warmup_device(self.model, self.device, imgsz=imgsz)

    def detect_batch(self, frames, classes, conf: float, imgsz: int) -> list[list[Detection]]:
        results = self.model.predict(frames, classes=list(classes), conf=conf,
                                      imgsz=imgsz, device=self.device, verbose=False)
        out = []
        for r in results:
            out.append([Detection(float(box.conf[0]), *[float(v) for v in box.xyxy[0].tolist()])
                        for box in r.boxes])
        return out

    def describe(self) -> str:
        from .device_config import device_summary

        return f"torch/{device_summary(self.device)}"


class OnnxDetector:
    """ONNX Runtime, with the pre/post-processing ultralytics would have done.

    The weights are the same network; everything that could make this disagree
    with TorchDetector lives in the four steps around it - letterbox geometry,
    channel order, normalisation, and mapping boxes back to source pixels. Each
    is matched to ultralytics deliberately rather than approximately, because a
    subtle mismatch here degrades ball recall without ever looking broken (the
    same failure mode video_source.even_crop guards against).
    """

    kind = "onnx"

    def __init__(self, weights, device: str = "cpu"):
        import onnxruntime as ort

        self.weights = weights
        self.device = device
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"no ONNX weights at {weights} - export them first with:\n"
                f"    shot-clipper-export-onnx")
        self.session = ort.InferenceSession(str(weights), providers=self._providers())
        self.input_name = self.session.get_inputs()[0].name
        self._provider = self.session.get_providers()[0]

    @staticmethod
    def _providers() -> list[str]:
        """Best available execution provider, most specific first.

        DirectML is the point of this backend on Windows: it reaches AMD and
        Intel GPUs that the CUDA build can't, with no toolkit install.
        onnxruntime silently falls through to the next entry when one isn't
        registered in the installed wheel, so listing all of them is safe.
        """
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                     "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def warmup(self, imgsz: int | None = None) -> None:
        import numpy as np

        side = imgsz or 640
        dummy = np.zeros((1, 3, side, side), dtype=np.float32)
        try:
            self.session.run(None, {self.input_name: dummy})
        except Exception as e:  # noqa: BLE001 - warmup is an optimisation, never fatal
            print(f"Warmup failed (non-fatal): {e}")

    def detect_batch(self, frames, classes, conf: float, imgsz: int) -> list[list[Detection]]:
        import numpy as np

        if not frames:
            return []
        batch, meta = [], None
        for frame in frames:
            tensor, meta = _letterbox(frame, imgsz)
            batch.append(tensor)
        blob = np.stack(batch).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        return _decode(outputs, meta, classes, conf,
                       source_shape=(frames[0].shape[0], frames[0].shape[1]))

    def describe(self) -> str:
        label = {"DmlExecutionProvider": "DirectML",
                 "CUDAExecutionProvider": "CUDA",
                 "CoreMLExecutionProvider": "CoreML",
                 "CPUExecutionProvider": "CPU"}.get(self._provider, self._provider)
        return f"onnx/{label}"


def _letterbox(frame, imgsz: int):
    """Resize-and-pad one BGR frame to the network's input, ultralytics-style.

    Returns (CHW float32 RGB tensor in 0..1, (ratio, pad_x, pad_y)).

    Padding is minimal - grown only to the next multiple of the stride, not to
    a full imgsz square. That's what ultralytics does for a batch of
    same-shaped images (LetterBox(auto=True)), and since detection runs on a
    tall narrow hoop ROI, squaring it instead would scale the ball down and
    quietly cost recall.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_w, pad_h = (imgsz - new_w) % STRIDE, (imgsz - new_h) % STRIDE
    pad_w /= 2
    pad_h /= 2

    if (w, h) != (new_w, new_h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(LETTERBOX_COLOR,) * 3)

    rgb = padded[:, :, ::-1]  # BGR -> RGB
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, (ratio, left, top)


def _decode(outputs, meta, classes, conf: float, source_shape) -> list[list[Detection]]:
    """Turn raw YOLOv8 output into per-frame Detections in source pixels.

    Output is (batch, 4 + n_classes, n_anchors): four box values (cx, cy, w, h
    in letterboxed input pixels) followed by one score per class. v8 has no
    objectness channel - the class score is the confidence.
    """
    import numpy as np

    ratio, pad_x, pad_y = meta
    src_h, src_w = source_shape
    if outputs.ndim != 3:
        raise RuntimeError(f"unexpected ONNX output shape {outputs.shape} - expected "
                            f"(batch, 4+classes, anchors)")
    preds = np.transpose(outputs, (0, 2, 1))  # -> (batch, anchors, 4+classes)
    n_channels = preds.shape[2]
    wanted = list(classes)
    if max(wanted) + 4 >= n_channels:
        raise RuntimeError(f"class {max(wanted)} is out of range for a model with "
                            f"{n_channels - 4} classes")

    per_frame = []
    for frame_preds in preds:
        scores = frame_preds[:, [4 + c for c in wanted]].max(axis=1)
        keep = scores >= conf
        if not keep.any():
            per_frame.append([])
            continue
        boxes = frame_preds[keep, :4]
        kept_scores = scores[keep]

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - pad_x) / ratio
        y1 = (cy - bh / 2 - pad_y) / ratio
        x2 = (cx + bw / 2 - pad_x) / ratio
        y2 = (cy + bh / 2 - pad_y) / ratio
        # ultralytics clips to the source frame after rescaling; a ball
        # half-outside the ROI would otherwise shift its centre off-frame
        x1, x2 = np.clip(x1, 0, src_w), np.clip(x2, 0, src_w)
        y1, y2 = np.clip(y1, 0, src_h), np.clip(y2, 0, src_h)

        per_frame.append([Detection(float(s), float(a), float(b), float(c), float(d))
                          for s, a, b, c, d in zip(kept_scores, x1, y1, x2, y2)])
    return per_frame


def export_onnx(weights=None, imgsz: int | None = None, dynamic: bool = True):
    """Export a .pt to .onnx next to it. Needs torch/ultralytics installed -
    this is a build step, not something the packaged app ever runs.

    dynamic keeps the input size variable, which is not optional here:
    detection runs at the hoop ROI's native resolution (a different shape per
    video), and a fixed-size export would silently rescale every frame.
    """
    from ultralytics import YOLO

    pt_path = paths.find_model(weights) if weights else paths.detect_weights()
    if not Path(pt_path).is_file():
        raise FileNotFoundError(f"no weights at {pt_path}")
    model = YOLO(pt_path)
    kwargs = {"format": "onnx", "dynamic": dynamic, "simplify": True}
    if imgsz:
        kwargs["imgsz"] = imgsz
    produced = model.export(**kwargs)
    return Path(produced)


def main() -> None:
    """shot-clipper-export-onnx"""
    import argparse

    parser = argparse.ArgumentParser(description=export_onnx.__doc__)
    parser.add_argument("--weights", type=str, default=None,
                         help=f"default: {paths.DEFAULT_DETECT_WEIGHTS}")
    parser.add_argument("--imgsz", type=int, default=None,
                         help="nominal export size; the graph stays dynamic regardless")
    parser.add_argument("--static", action="store_true",
                         help="export a fixed input size (not recommended - detection runs "
                              "at each video's own hoop-ROI resolution)")
    args = parser.parse_args()

    out = export_onnx(args.weights, imgsz=args.imgsz, dynamic=not args.static)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"exported -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
