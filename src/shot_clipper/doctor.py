"""shot-clipper-doctor: is this machine ready to run the whole pipeline
(/make-highlights) locally? Checks every piece and says how to fix what's
missing. Exit code 1 if something required is missing.

    poetry run shot-clipper-doctor
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_gpu():
    try:
        import torch
    except ImportError:
        return FAIL, "torch not installed", "poetry install --with ml"
    has_nvidia = bool(shutil.which("nvidia-smi"))
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
        return OK, f"CUDA: {name} (sm {cap}), torch {torch.__version__}", ""
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return OK, f"Apple GPU (MPS), torch {torch.__version__}", ""
    if has_nvidia:
        # RTX 50xx (Blackwell) needs a CUDA 12.8+ torch build; PyPI's Windows torch is CPU-only
        return FAIL, f"NVIDIA GPU present but torch {torch.__version__} can't use it", (
            "poetry run python -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 "
            "--index-url https://download.pytorch.org/whl/cu132   (see docs/GPU_SETUP.md)")
    return WARN, "no GPU - detection will be very slow on CPU", "see docs/GPU_SETUP.md"


def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return FAIL, "ffmpeg/ffprobe not on PATH", (
            "Windows: winget install Gyan.FFmpeg | Linux: apt install ffmpeg | Mac: brew install ffmpeg")
    accels = _run(["ffmpeg", "-hide_banner", "-hwaccels"]).split()
    hw = [a for a in ("cuda", "videotoolbox", "d3d11va", "vaapi") if a in accels]
    return OK, "ffmpeg found" + (f", hardware decode: {', '.join(hw)}" if hw else ""), ""


def check_node():
    node, npx = shutil.which("node"), shutil.which("npx")
    if not node or not npx:
        return FAIL, "Node.js not found (reels are rendered with Remotion)", (
            "install Node.js 22 LTS: Windows winget install OpenJS.NodeJS.LTS | Mac brew install node | "
            "Linux nodejs.org")
    version = _run([node, "--version"])
    if not (REPO / "remotion-reel" / "node_modules").exists():
        return WARN, f"Node {version}; Remotion packages not installed yet", (
            "cd remotion-reel && npm ci   (also happens automatically on the first render)")
    return OK, f"Node {version}, Remotion packages installed", ""


def check_weights():
    from . import inference
    from .person_id import reid

    missing = [p for p in (inference.RFDETR_WEIGHTS, str(reid.WEIGHTS)) if not Path(p).is_file()]
    if missing:
        return WARN, f"not downloaded yet: {', '.join(missing)}", "downloaded automatically on first use"
    return OK, "detector + re-ID weights present", ""


def check_make_model():
    from . import prelabel

    for path in (prelabel.MODEL_DIR / prelabel.GLOBAL_MODEL_NAME,
                 Path("data/prelabel/models") / prelabel.GLOBAL_MODEL_NAME):
        if prelabel.load_model(path) is not None:
            return OK, f"AI make model: {path}", ""
    return FAIL, "no AI make model (models/prelabel/global.joblib)", "git pull - it ships with the repo"


def check_vlm():
    from . import highlight_vlm, highlights

    backend = highlights.vlm_backend()
    if backend == "mlx":
        return OK, f"AI stars: MLX ({highlights.VLM_PYTHON})", ""
    if backend == "ollama":
        return OK, f"AI stars: Ollama, model {highlight_vlm.OLLAMA_MODEL}", ""
    return WARN, "no vision model - AI stars fall back to the make score (much rougher)", (
        f"install Ollama (ollama.com), then: ollama pull {highlight_vlm.OLLAMA_MODEL}   "
        "(~20GB, needs a 24GB+ GPU; qwen3-vl:8b for smaller GPUs via SHOT_CLIPPER_OLLAMA_MODEL)")


def check_disk():
    free = shutil.disk_usage(Path.home()).free / 2**30
    if free < 50:
        return WARN, f"{free:.0f} GB free", "a session's clips + reels take ~3GB; models ~25GB"
    return OK, f"{free:.0f} GB free", ""


CHECKS = [("GPU", check_gpu), ("ffmpeg", check_ffmpeg), ("Node/Remotion", check_node),
          ("model weights", check_weights), ("AI make model", check_make_model),
          ("AI stars (VLM)", check_vlm), ("disk", check_disk)]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # names/marks on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(REPO)  # model paths are repo-relative
    print(f"{platform.system()} {platform.release()}, Python {sys.version.split()[0]}\n")
    failed = False
    for name, fn in CHECKS:
        try:
            status, detail, fix = fn()
        except Exception as e:  # noqa: BLE001 - a broken check is itself the finding
            status, detail, fix = FAIL, f"{type(e).__name__}: {e}", ""
        failed |= status == FAIL
        print(f"[{status}] {name:15s} {detail}")
        if fix and status != OK:
            print(f"       fix: {fix}")
    if sys.platform != "darwin":
        print("\nnote: the Photos-album step is macOS only and is skipped here.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
