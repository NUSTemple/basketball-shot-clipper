"""Centralized device configuration for GPU/CPU/MPS selection.

Handles automatic detection of available compute devices (CUDA, MPS, CPU)
and device-specific optimizations like batch size and precision mode.
"""
from typing import Literal


def get_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu.

    Returns:
        "cuda" (NVIDIA GPU), "mps" (Apple Metal), or "cpu"
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_optimal_batch_size(device: str | None = None) -> int:
    """Get recommended batch size based on device.

    Batch sizes tuned for typical hardware:
    - CUDA (NVIDIA): 16-24 (tested on 4070 with 8GB+ VRAM)
    - MPS (Apple): 8-12 (conservative for mixed hardware)
    - CPU: 4-8 (memory-bound on typical systems)

    Args:
        device: "cuda", "mps", "cpu", or None for auto-detect

    Returns:
        Recommended batch size for inference
    """
    if device is None:
        device = get_device()

    if device == "cuda":
        return 16
    elif device == "mps":
        return 8
    else:
        return 4


def get_optimal_precision(device: str | None = None) -> Literal["fp32", "fp16"]:
    """Get recommended precision mode based on device.

    FP16 (half precision) gives ~2x speedup on NVIDIA GPUs with minimal
    accuracy loss; not recommended for CPU or MPS.

    Args:
        device: "cuda", "mps", "cpu", or None for auto-detect

    Returns:
        "fp32" (full precision) or "fp16" (half precision)
    """
    if device is None:
        device = get_device()

    return "fp16" if device == "cuda" else "fp32"


def warmup_device(model, device: str, batch_shape: tuple[int, ...] = (1, 3, 640, 640)):
    """Warmup GPU by running a dummy inference pass.

    Reduces latency of first real inference by compiling kernels/allocating memory.
    Safe to call even on CPU (just a no-op).

    Args:
        model: YOLO model instance
        device: "cuda", "mps", or "cpu"
        batch_shape: Shape of dummy batch (B, C, H, W) for image tensors
    """
    if device != "cuda":
        return

    try:
        import torch
        import numpy as np
        dummy_batch = np.zeros(batch_shape, dtype=np.uint8)
        with torch.no_grad():
            model.predict(dummy_batch, device=device, verbose=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")


def device_summary(device: str | None = None) -> str:
    """Human-readable device configuration summary.

    Returns a string like "CUDA (NVIDIA RTX 4070)" or "CPU (Intel i9)"
    for logging/debugging.
    """
    if device is None:
        device = get_device()

    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            return f"CUDA ({name})"
        except Exception:
            return "CUDA (device unknown)"
    elif device == "mps":
        return "MPS (Apple Metal)"
    else:
        return "CPU"
