# GPU Setup Guide

This guide covers GPU acceleration setup for basketball-shot-clipper on Windows with NVIDIA GPUs.

## Native Windows Setup (Recommended for Best Performance)

### Prerequisites
- NVIDIA GPU (RTX 4070 or better recommended)
- NVIDIA CUDA Toolkit 12.1+ ([download](https://developer.nvidia.com/cuda-downloads))
- NVIDIA cuDNN ([download](https://developer.nvidia.com/cudnn))
- Python 3.11+ with pip/Poetry

### Installation

1. **Verify NVIDIA GPU is detected:**
   ```powershell
   nvidia-smi
   ```
   Should show your GPU details (memory, compute capability, etc.)

2. **Install dependencies:**
   ```bash
   poetry install --with ml
   ```

3. **Verify CUDA is available to PyTorch:**
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`. If `False`, check CUDA/cuDNN installation.

4. **Run detection with GPU:**
   ```bash
   poetry run shot-clipper-detect path/to/video.MP4
   ```
   
   On first run, this will print `device: CUDA (NVIDIA RTX 4070)` or similar, confirming GPU is in use.

### Troubleshooting

**GPU not detected (shows "device: CPU")**
- Ensure NVIDIA CUDA Toolkit 12.1+ is installed
- Verify `nvidia-smi` shows your GPU
- Check PyTorch CUDA availability:
  ```python
  import torch
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"CUDA version: {torch.version.cuda}")
  ```
- If `CUDA available: False`, install a CUDA build of PyTorch. Pick the index
  matching the CUDA version `nvidia-smi` reports (`cu132` for CUDA 13.2,
  `cu124` for 12.4, ...), and install it into **Poetry's own venv** - plain
  `pip install` may land in a different interpreter:

  ```powershell
  $venv = poetry env info -p
  & "$venv\Scripts\pip.exe" install torch==2.13.0 torchvision==0.28.0 `
      --index-url https://download.pytorch.org/whl/cu132
  ```

  Verified working on Windows 11 + RTX 4070 with driver 595.95 / CUDA 13.2.
  Notes from getting there:
  - The cu132 index only carries torch 2.12-2.13, and publishes no
    `torchaudio` at all. This project doesn't use torchaudio, so skip it -
    asking for it just fails the whole command.
  - **`poetry install` will quietly put the `+cpu` build back**, because
    `pyproject.toml` pins no CUDA-specific source. After running it, re-check
    `torch.cuda.is_available()` and reinstall as above if needed. Install
    incidental tools (pytest and friends) with the venv's own pip for the
    same reason.
  - Windows Application Control can block torch's DLLs on first import
    (`WinError 4551` loading `shm.dll`). Restarting the terminal cleared it.

**Out of memory errors**
- Reduce batch size (see Environment Variables below)
- The default batch size of 16 requires ~3GB VRAM on 4K footage
- For 8GB GPUs, try `SHOT_CLIPPER_BATCH_SIZE=8`

**Slower than CPU**
- First run may be slow due to CUDA kernel compilation
- Check that model weights are at `models/yolov8m.pt` (skip CPU serialization overhead)
- GPU should show 3-5x speedup over CPU once warmed up

### Environment Variables

Control GPU behavior with environment variables:

```bash
# Force specific device (cuda, mps, cpu)
set SHOT_CLIPPER_DEVICE=cuda

# Override batch size (default: 16 for CUDA, 8 for MPS, 4 for CPU)
set SHOT_CLIPPER_BATCH_SIZE=24

# For training filter models
set SHOT_CLIPPER_DEVICE=cuda poetry run shot-clipper-train-filter --clips-dir /path/to/clips
```

## Docker Setup with NVIDIA GPU

### Prerequisites
- NVIDIA GPU
- Docker Desktop for Windows
- NVIDIA Container Runtime for Windows ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Installation

1. **Install NVIDIA Container Runtime:**
   Follow the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for Docker on Windows.

2. **Verify NVIDIA Docker:**
   ```bash
   docker run --rm --runtime=nvidia nvcr.io/nvidia/cuda:12.3-runtime-ubuntu22.04 nvidia-smi
   ```
   Should show your GPU details.

3. **Run with Docker Compose:**
   ```bash
   CLIPS_DIR=/path/to/clips docker compose up --build
   ```
   
   This will automatically use GPU if available. Inside the container, you'll see:
   ```
   device: CUDA (NVIDIA RTX 4070)
   ```

4. **Optional: Control which GPUs are used:**
   
   In `docker-compose.yml`, add to both services:
   ```yaml
   environment:
     - NVIDIA_VISIBLE_DEVICES=0  # Use only first GPU
   ```

### Troubleshooting

**"could not select device driver" error**
- Ensure NVIDIA Container Runtime is installed
- Verify `docker run --runtime=nvidia ...` works
- Restart Docker daemon after installing runtime

**GPU not available in container**
- Check `docker run --rm --runtime=nvidia ... nvidia-smi`
- If that fails, runtime installation is incomplete
- Try explicit GPU specification:
  ```yaml
  runtime: nvidia
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  ```

**Container uses CPU instead of GPU**
- Check that `runtime: nvidia` is set in `docker-compose.yml`
- Restart containers: `docker compose down && docker compose up`

## Performance Expectations

Measured on an RTX 4070 against 2688x1512 HEVC drone footage, sampling at
15fps with a 347x576 hoop ROI:

| stage | per sampled frame | share of wall time |
|---|---|---|
| video decode (OpenCV, CPU) | 17.8 ms | 77% |
| YOLO inference (CUDA) | 5.1 ms | 23% |
| video decode (ffmpeg + NVDEC) | **6.1 ms** | - |

**Detection is decode-bound, not inference-bound.** Moving the model to CUDA
alone changes little: YOLO was never the bottleneck. What matters is that
frames stop being decoded on the CPU. With `nvidia-smi dmon` you can watch
this directly - on the OpenCV path the GPU's SM sits at 0-4% and its `dec`
(NVDEC) engine at 0% while a scan runs.

End-to-end throughput on that footage:

| frame source | throughput |
|---|---|
| OpenCV CPU decode | 2.88x realtime |
| ffmpeg + NVDEC | **5.97x realtime** (2.08x faster) |

Decode and inference are now roughly balanced (1.83s vs 1.52s over 300
frames), so they are still run one after the other; overlapping them with a
prefetch thread is the remaining ~1.8x and is not implemented yet.

**First run warmup:** the first batch pays CUDA kernel compilation, so a very
short video can look slower than expected. Subsequent runs are fast.

## Choosing the decoder

`SHOT_CLIPPER_DECODER` selects the frame source:

| value | meaning |
|---|---|
| `auto` (default) | NVDEC if this ffmpeg build lists `cuda`, else ffmpeg CPU, else OpenCV |
| `nvdec` | force the ffmpeg CUDA path |
| `cpu` | ffmpeg pipe without hardware decode |
| `opencv` | the original `cv2.VideoCapture` loop |

Hardware decode was verified **bit-identical** to ffmpeg's own CPU decode
(40/40 frames, max difference 0). Note that ffmpeg and OpenCV do differ
slightly from each other in YUV->BGR conversion (max 12, mean 0.62 per
channel). Ball centres still agree to ~1e-4 in normalised units - sub-pixel -
but because `BALL_CONF_THRESHOLD` is deliberately low (0.1), a handful of
marginal detections can appear or disappear. Set `SHOT_CLIPPER_DECODER=opencv`
if you need to reproduce timestamps from a previous run exactly.

## Native vs Docker Performance

| Metric | Native | Docker |
|--------|--------|--------|
| GPU Support | Full CUDA | Full CUDA (if runtime installed) |
| Speed | ~1-1.5 sec/frame | ~1-1.5 sec/frame |
| Setup Complexity | Moderate (CUDA + cuDNN) | Simple (Docker + NVIDIA Runtime) |
| Best for | Production pipelines | Development / batch jobs |

## Advanced: Custom CUDA Versions

If you need a specific CUDA version:

1. Edit `Dockerfile`:
   ```dockerfile
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04  # Change version here
   ```

2. Rebuild:
   ```bash
   docker compose build --no-cache
   ```

Supported versions: 11.8, 12.1, 12.3, 12.4 (check [NVIDIA Docker Hub](https://hub.docker.com/r/nvidia/cuda) for latest).

## Fallback to CPU

If GPU setup fails, the system automatically falls back to CPU with a warning:
```
GPU requested but unavailable - falling back to CPU
```

All detection/training commands work identically on CPU, just slower. No code changes needed.
