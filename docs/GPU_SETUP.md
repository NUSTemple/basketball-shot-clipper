# GPU Setup Guide

GPU setup for basketball-shot-clipper, on **Windows/Linux with an NVIDIA GPU**
and on **macOS with Apple Silicon**.

**Read this first, because it determines what is worth configuring:**
detection is *decode*-bound, not inference-bound. On an RTX 4070 against
2688x1512 HEVC, decoding a sampled frame on the CPU cost 17.8ms while running
YOLO over it cost 5.1ms - so ~77% of the time was the GPU idle, waiting.

That means there are **two independent accelerators to enable**, and the one
everybody thinks of first is the smaller half:

| what | picked by | env override |
|---|---|---|
| **video decode** (~77%) | `ffmpeg -hwaccels` -> NVDEC or VideoToolbox | `SHOT_CLIPPER_DECODER` |
| model inference (~23%) | `torch` -> CUDA or MPS | `SHOT_CLIPPER_DEVICE` |

They are detected separately and on purpose. An Intel Mac has VideoToolbox but
no MPS; a Linux box could have CUDA but an ffmpeg built without it. Every run
prints both, so you can always see which you actually got:

```
device:  MPS (Apple M3 Max)
decoder: ffmpeg/VideoToolbox (GPU)
```

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

**Slower than expected**
- The first batch pays kernel/shader compilation, so a very short video can
  look worse than it is
- Check the `decoder:` line. If it says `ffmpeg (CPU)` you have the inference
  half only, which is the *smaller* half — see "Choosing the decoder" below

## macOS (Apple Silicon)

A Mac needs both halves too: **MPS** for inference and **VideoToolbox** for
decode. MPS alone leaves ~77% of the time on the CPU.

```bash
brew install ffmpeg          # Homebrew builds include VideoToolbox
ffmpeg -hwaccels             # must list: videotoolbox
poetry install --with ml     # torch has macOS arm64 wheels; no special index
```

Verify, then run — both accelerators are picked up automatically:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # True
poetry run shot-clipper-detect path/to/video.MP4
```

Expected startup:

```
device:  MPS (Apple M3 Max)
decoder: ffmpeg/VideoToolbox (GPU)
```

**Before trusting hardware decode on a new machine**, run the check that
proves it feeds the detector the same pixels software decode did:

```bash
python scripts/verify_decoder.py path/to/video.MP4
```

It must report **bit-identical, max=0**. That is the bar NVDEC cleared on
Windows (40/40 frames). If it fails, hardware decode would silently *move*
detections rather than fail loudly — roll back with `SHOT_CLIPPER_DECODER=cpu`
and report the colour tagging it prints.

### Intel Macs

VideoToolbox yes, MPS no — they are unrelated capabilities. So this is the
**correct** output on an Intel Mac, not a misconfiguration:

```
device:  CPU
decoder: ffmpeg/VideoToolbox (GPU)
```

You still get the larger (decode) half of the speedup.

### macOS troubleshooting

- **`videotoolbox` missing from `ffmpeg -hwaccels`** — conda-forge and some
  static builds ship without it. Use Homebrew's.
- **`NotImplementedError` for an MPS operator** — run with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU per-op. Set it in the
  shell; it must be set before `import torch`, so the app cannot set it for you.
- **MPS out of memory** — `SHOT_CLIPPER_BATCH_SIZE=4`.
- **MPS producing odd results** — `SHOT_CLIPPER_DEVICE=cpu` is the escape
  hatch, exactly like `SHOT_CLIPPER_DECODER=opencv` is for decode.

### Environment Variables

```bash
# Force the inference device: cuda | mps | cpu
# An unavailable choice warns and falls back to CPU rather than failing
# deep inside torch later.
SHOT_CLIPPER_DEVICE=cpu

# Frames per inference call. Default 16 (CUDA) / 8 (MPS) / 4 (CPU).
# Lower it on an 8GB GPU. Note this is a speed/memory knob, not a free one:
# batch composition can nudge a marginal detection at BALL_CONF_THRESHOLD=0.1.
SHOT_CLIPPER_BATCH_SIZE=8

# Force the decode path: nvdec | videotoolbox | cpu | opencv
SHOT_CLIPPER_DECODER=opencv
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

3. **Run with Docker Compose, adding the NVIDIA overlay:**
   ```bash
   CLIPS_DIR=/path/to/clips docker compose \
       -f docker-compose.yml -f docker-compose.nvidia.yml up --build
   ```

   Inside the container you should see `device: CUDA (NVIDIA GeForce RTX 4070)`.

   The GPU bits live in a separate overlay file because `runtime: nvidia` is a
   hard failure ("could not select device driver") on any host without the
   NVIDIA container runtime. Plain `docker compose up --build` therefore still
   works everywhere, CPU-only.

   The base image is `python:3.12-slim-bookworm`, not an `nvidia/cuda` one, and
   that is fine for CUDA: the linux/amd64 PyPI `torch` wheel bundles the CUDA
   runtime through its `nvidia-*` dependencies and only needs the host driver,
   which the container runtime injects. It also builds on Apple Silicon. To pin
   a CUDA base anyway:
   `docker build --build-arg BASE_IMAGE=nvidia/cuda:12.3-runtime-ubuntu22.04 .`

4. **Optional: control which GPUs are used** — edit `NVIDIA_VISIBLE_DEVICES`
   in `docker-compose.nvidia.yml`.

**Docker on macOS gets no acceleration at all** — there is no GPU passthrough,
so no MPS, and the container's Linux ffmpeg will not report `videotoolbox`
either. `decoder_kind()` correctly falls back to the ffmpeg CPU path. Since
native macOS now gets *both* accelerators, the native-vs-Docker gap on a Mac is
much wider than it used to be: run long videos natively.

### Troubleshooting

**"could not select device driver" error**
- You passed `-f docker-compose.nvidia.yml` on a host without the NVIDIA
  container runtime. Either install it, or just drop the overlay and run
  `docker compose up --build` CPU-only.
- Restart the Docker daemon after installing the runtime

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

**First run warmup:** the first batch pays kernel/shader compilation, so a very
short video can look slower than expected. Subsequent runs are fast.

**macOS: not yet measured.** No Apple numbers are published here because none
have been taken — run `scripts/verify_decoder.py` on your Mac and fill this in
rather than assuming the NVIDIA ratios carry over. What is known is that the
*shape* of the problem is the same: MPS accelerates the ~23% and VideoToolbox
the ~77%.

## Choosing the decoder

`SHOT_CLIPPER_DECODER` selects the frame source:

| value | meaning |
|---|---|
| `auto` (default) | NVDEC if `ffmpeg -hwaccels` lists `cuda`; else VideoToolbox if it lists `videotoolbox`; else ffmpeg CPU; else OpenCV |
| `nvdec` | force the ffmpeg CUDA path |
| `videotoolbox` | force the ffmpeg Apple VideoToolbox path |
| `cpu` | ffmpeg pipe without hardware decode |
| `opencv` | the original `cv2.VideoCapture` loop |

An explicitly named decoder skips the probe entirely — forcing one shouldn't
depend on how your local ffmpeg self-reports. Selection is never gated on CPU
architecture, only on what ffmpeg reports, which is what lets an Intel Mac get
VideoToolbox without pretending it can run Metal inference.

**Verification status.** NVDEC is verified **bit-identical** to ffmpeg's own
CPU decode — 40/40 frames, max difference 0, on 10-bit `yuv420p10le` HEVC.
VideoToolbox is **not yet measured**; run `scripts/verify_decoder.py` on your
Mac and record the result here.

ffmpeg and OpenCV do differ slightly from *each other* in YUV->BGR conversion
(max 12, mean 0.62 per channel — measured). Ball centres still agree to ~1e-4
in normalised units, i.e. sub-pixel, but because `BALL_CONF_THRESHOLD` is
deliberately low (0.1) a handful of marginal detections can appear or
disappear. `SHOT_CLIPPER_DECODER=opencv` reproduces timestamps from a
pre-ffmpeg run exactly.

## Native vs Docker Performance

| | Native | Docker |
|---|---|---|
| **Windows/Linux + NVIDIA** | CUDA inference + NVDEC decode | same, with the `docker-compose.nvidia.yml` overlay |
| **macOS (Apple Silicon)** | MPS inference + VideoToolbox decode | **neither** — no GPU passthrough, and the container's Linux ffmpeg has no VideoToolbox |

On a Mac the gap is now large enough to matter: Docker loses *both*
accelerators, not just inference. Use native for anything long.

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
