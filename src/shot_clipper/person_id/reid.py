"""Person re-identification embeddings: OSNet-AIN trained on MSMT17.

Picked against the owner's scorer tags (2026-10, three sessions): a re-ID
model trained to match one person across cameras beat general image
embeddings by a wide margin on "is the nearest other goal by the same
player" (~50% vs CLIP ~35%, DINOv2 ~31%, colour histograms ~15%).
Fine-tuning it on the tags themselves didn't help yet - too few tagged
weeks - so the stock MSMT17 weights are used.
"""
from pathlib import Path

import numpy as np

WEIGHTS = Path("models/osnet_ain_x1_0_msmt17.pt")
# torchreid model zoo (MIT): osnet_ain_x1_0 trained on MSMT17
WEIGHTS_URL = ("https://drive.usercontent.google.com/download?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal"
               "&export=download&confirm=t")
WEIGHTS_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
METHOD = "osnet-ain-msmt17"
INPUT_SIZE = (128, 256)  # width, height the network was trained at
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_weights(path: Path = WEIGHTS) -> Path:
    """The weights file, downloading the pinned one on first use."""
    path = Path(path)
    if path.is_file():
        return path
    import hashlib
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"downloading person re-ID weights (~17MB) -> {path}", flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as resp, open(partial, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"re-ID weights from {WEIGHTS_URL} failed the sha256 check")
    partial.replace(path)
    return path


class ReidModel:
    def __init__(self, device: str = "cpu", weights: Path = WEIGHTS):
        import torch

        from .osnet_ain import osnet_ain_x1_0

        self.torch = torch
        self.device = device
        model = osnet_ain_x1_0(num_classes=4101)  # MSMT17's identities; the classifier isn't used
        state = torch.load(ensure_weights(weights), map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.get("state_dict", state).items()
                 if not k.startswith("classifier")}
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()

    def embed(self, crops: list) -> np.ndarray | None:
        """One unit vector for a person from several BGR crops of them (the
        mean of the per-crop embeddings), or None without usable crops."""
        import cv2

        crops = [c for c in crops if c is not None and c.size and c.shape[0] >= 20 and c.shape[1] >= 8]
        if not crops:
            return None
        x = np.stack([(cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), INPUT_SIZE).astype(np.float32) / 255.0
                       - _MEAN) / _STD for c in crops])
        with self.torch.no_grad():
            e = self.model(self.torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device))
            e = self.torch.nn.functional.normalize(e, dim=-1).mean(0)
        v = e.cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-9)
