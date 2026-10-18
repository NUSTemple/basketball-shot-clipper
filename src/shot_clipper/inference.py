"""Ball detection backends: RF-DETR (the default), ultralytics/torch, or ONNX Runtime.

The detector is OpenHoopVision's basketball-trained RF-DETR Small
(RFDETRDetector below, Apache-2.0, github.com/feifeifei85/OpenHoopVision).
It replaced COCO YOLO's generic "sports ball" class, which a friend's 1080p
side-on camera exposed: YOLO saw the ball in under 1% of sampled frames there
and find_makes never fired, while this model sees it in 10-13% - and more
often than YOLO on the original 4K DJI footage too, without losing any
labeled make.

The two YOLO backends stay for one reason: the trained shot filter
(features.py / train_filter.py) was fit on trajectories YOLO produced, so
scoring clips with it has to keep extracting features with the same YOLO
weights (FILTER_FEATURE_WEIGHTS) until it's retrained on RF-DETR tracks.

Why two YOLO backends. The torch path is the one every detection result in this repo was
produced and validated with, so it stays the reference. The ONNX path exists
because a packaged build carrying CUDA torch is roughly 3GB and only helps
NVIDIA owners, against ~120MB for onnxruntime-directml, which runs on any
DirectX 12 GPU - see docs/GPU_SETUP.md. Both are selectable at runtime via
$SHOT_CLIPPER_INFERENCE so the two can be measured against each other on real
footage (shot-clipper-compare-backends) instead of swapped on faith.

Both return the same thing: per frame, a list of Detection in the *input
frame's* pixel coordinates, already filtered to the requested classes and
confidence. Callers do their own geometry on top (see
detect_shots.detect_ball_centers_batch).

No NMS in the ONNX path, deliberately. Every consumer in this codebase keeps
only the single highest-confidence box per frame, and NMS never removes the
top-scoring box - it only drops lower-scoring overlaps. The assumption is
load-bearing, so it is written down where someone changing it will look.
"""
from pathlib import Path
from typing import NamedTuple
import os

BACKEND_ENV = "SHOT_CLIPPER_INFERENCE"  # auto (default) | torch | onnx - YOLO weights only
VALID_BACKENDS = ("torch", "onnx")

# Pinned to a commit, and checked against the LFS pointer's sha256, so the
# weights a run uses can't change underneath it.
RFDETR_WEIGHTS = "models/rf-detr-basketball.pth"
RFDETR_WEIGHTS_URL = ("https://media.githubusercontent.com/media/feifeifei85/OpenHoopVision/"
                      "9c000659a7d34ded894db90161c27cc0b55bb745/weights/rf-detr-basketball.pth")
RFDETR_WEIGHTS_SHA256 = "43932f5e4425c176d967f938dd134db3dbf9271176ecdd4888a4b0e702c8a977"

# Overridable so the hosted deployment can point this at the GCS-mounted
# volume (weights uploaded there once) instead of the container's own
# ephemeral filesystem - without this, a fresh Cloud Run instance has no
# local copy and would re-download ~128MB on every single cold start.
DEFAULT_WEIGHTS = os.environ.get("SHOT_CLIPPER_MODEL_PATH", RFDETR_WEIGHTS)
# What the trained shot filter's features were extracted with - see the
# module docstring. Not the detector. Overridable for the same cold-start
# reason as DEFAULT_WEIGHTS.
FILTER_FEATURE_WEIGHTS = os.environ.get("SHOT_CLIPPER_FILTER_FEATURE_WEIGHTS", "models/yolov8l.pt")

COCO_SPORTS_BALL_CLASS = 32

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


def is_rfdetr_weights(weights) -> bool:
    """RF-DETR checkpoints are .pth; ultralytics YOLO weights are .pt."""
    return Path(weights).suffix == ".pth"


def ensure_rfdetr_weights(weights) -> Path:
    """Return the path to the RF-DETR checkpoint, downloading the pinned one
    first if it's the default file and missing - the installer and a fresh
    checkout both rely on this rather than a separate setup step.

    Any other missing path is an error rather than a silent download, since
    it means someone pointed the app at weights that aren't there.
    """
    path = Path(weights)
    if path.is_file():
        return path
    if path.name != Path(RFDETR_WEIGHTS).name:
        raise FileNotFoundError(f"no RF-DETR weights at {path}")
    import hashlib
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"downloading detector weights (~128MB) -> {path}", flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(RFDETR_WEIGHTS_URL, timeout=60) as resp, open(partial, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != RFDETR_WEIGHTS_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"detector weights from {RFDETR_WEIGHTS_URL} failed the sha256 check")
    partial.replace(path)
    return path


def load_detector(weights=None, device: str | None = None, backend: str | None = None):
    """Build the detector for `weights`, defaulting to DEFAULT_WEIGHTS.

    A .pth is RF-DETR, whatever `backend` says. Otherwise weights is a YOLO
    .pt (torch backend) or, for the onnx backend, the .pt whose .onnx sibling
    should be loaded instead.
    """
    from .device_config import get_device

    device = device or get_device()
    pt_path = weights or DEFAULT_WEIGHTS
    if is_rfdetr_weights(pt_path):
        return RFDETRDetector(pt_path, device=device)
    kind = resolve_backend(backend)
    if kind == "onnx":
        return OnnxDetector(onnx_path_for(pt_path), device=device)
    return TorchDetector(pt_path, device=device)


class RFDETRDetector:
    """OpenHoopVision's basketball RF-DETR Small - the default detector.

    Classes are ball, holder, rim, scorer, player, referee; ball_class is
    looked up by name rather than assumed. Unlike YOLO it runs at each
    frame's own size (rounded up to what the backbone accepts), so detect_batch
    ignores imgsz - on the hoop ROI crop that's the same native-resolution
    choice the YOLO path makes, and upscaling 2x made no difference in testing.
    """

    kind = "rfdetr"
    # Below ~0.3 this model labels fixed spots near the rim (the net, a pole)
    # "ball" in most frames - median conf ~0.12 there vs ~0.8 on a real ball.
    # detect_shots.suppress_static drops whatever such spots survive this.
    ball_conf = 0.3

    def __init__(self, weights, device: str = "cpu"):
        from rfdetr import from_checkpoint

        self.weights = ensure_rfdetr_weights(weights)
        self.device = device
        self.model = from_checkpoint(str(self.weights), device=device)
        names = self.model.class_names
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_ids = {str(name).lower().strip(): int(i) for i, name in entries}
        self.ball_class = self.class_ids["ball"]
        cfg = self.model.model_config
        self._multiple = cfg.patch_size * cfg.num_windows

    def _shape(self, h: int, w: int) -> tuple[int, int]:
        m = self._multiple
        return max(m, -(-h // m) * m), max(m, -(-w // m) * m)

    def warmup(self, imgsz: int | None = None) -> None:
        import numpy as np

        side = imgsz or 640
        try:
            self.model.predict(np.zeros((side, side, 3), dtype=np.uint8), threshold=0.5,
                               shape=self._shape(side, side))
        except Exception as e:  # noqa: BLE001 - warmup is an optimisation, never fatal
            print(f"Warmup failed (non-fatal): {e}")

    def detect_batch(self, frames, classes, conf: float, imgsz: int) -> list[list[Detection]]:
        import cv2

        if not frames:
            return []
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        preds = self.model.predict(rgb, threshold=conf, shape=self._shape(*frames[0].shape[:2]))
        if not isinstance(preds, list):
            preds = [preds]
        wanted = set(classes)
        return [[Detection(float(c), *(float(v) for v in box))
                 for box, c, k in zip(p.xyxy, p.confidence, p.class_id) if int(k) in wanted]
                for p in preds]

    def describe(self) -> str:
        from .device_config import device_summary

        return f"rfdetr/{device_summary(self.device)}"


class TorchDetector:
    """ultralytics YOLO - kept for the trained filter's features."""

    kind = "torch"
    ball_class = COCO_SPORTS_BALL_CLASS
    # recall-first (decision 5): low bar, filtered by trajectory logic
    ball_conf = 0.1

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
    ball_class = COCO_SPORTS_BALL_CLASS
    ball_conf = TorchDetector.ball_conf

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
    this is a build step, not something a CPU/DirectML-only install ever runs.

    dynamic keeps the input size variable, which is not optional here:
    detection runs at the hoop ROI's native resolution (a different shape per
    video), and a fixed-size export would silently rescale every frame.
    """
    from ultralytics import YOLO

    pt_path = weights or FILTER_FEATURE_WEIGHTS
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
                         help=f"YOLO .pt to export (default: {FILTER_FEATURE_WEIGHTS})")
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
