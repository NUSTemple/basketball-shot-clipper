---
name: setup-local
description: First-time setup of this repo on a new computer so the whole basketball pipeline (/make-highlights - clips, AI scores, AI stars, player groups, highlight reels) runs locally. Windows with an NVIDIA GPU (incl. RTX 50-series) is the main target; also covers Linux and macOS. Use when someone has just cloned the repo, asks to set up / install / deploy it locally, or shot-clipper-doctor reports failures.
---

# Set up this PC to run the whole pipeline locally

Goal: at the end, `poetry run shot-clipper-doctor` shows no FAIL, and the
user knows the one sentence they'll use from now on:
`/make-highlights <video folder>`. Talk to the user in their language; keep
each message short and say what's happening and how long it takes.

## 1. Look before installing

Run these (Claude Code on Windows uses Git Bash; `powershell -Command` works
from there) and summarise what's already present:

```bash
uname -a; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
which python py poetry ffmpeg node npm ollama git
df -h .   # or: powershell -Command "Get-PSDrive -PSProvider FileSystem"
```

- No NVIDIA GPU (or `nvidia-smi` missing) on a PC that should have one: ask
  them to install/update the NVIDIA driver first (nvidia.com/drivers) and
  restart - nothing else works well without it.
- Needs ~40 GB free: Python packages incl. CUDA torch ~6 GB, models ~1 GB,
  the vision model ~20 GB, plus room for videos and clips.

Then tell the user, in one message, what will be installed and ask once for
the go-ahead (it installs system software via winget): Python 3.12, Poetry,
ffmpeg, Node.js LTS, Ollama + the `qwen3-vl:30b` vision model (~20 GB,
optional - without it AI stars are rougher). GPU memory below 24 GB: offer
`qwen3-vl:8b` instead.

## 2. Install

**Windows** - one script does everything and is safe to re-run:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File installer/setup-local-pipeline.ps1
#   add -SkipOllama to skip the vision model, or -OllamaModel qwen3-vl:8b
```

It takes 20-60 minutes (mostly the CUDA torch and vision-model downloads):
run it in the background and check its output every few minutes; relay
progress briefly. It ends by running `shot-clipper-doctor`.

**Linux / macOS**: follow `docs/LOCAL_SETUP.md` section 1 (package manager
installs, `poetry install --with ml`, `cd remotion-reel && npm ci`,
`ollama pull qwen3-vl:30b` on Linux; Macs use MLX instead of Ollama).

## 3. Restart, then check

Tools installed during this session are NOT on this Claude Code session's
PATH (Windows only reads PATH at process start). So after the script
finishes, tell the user exactly:

1. type `/exit` (or close the terminal),
2. open a NEW terminal in the repo folder and run `claude` again,
3. say: `run shot-clipper-doctor` (or `/setup-local` again).

After the restart run `poetry run shot-clipper-doctor` and fix any FAIL
from its `fix:` line. Common ones:

| doctor says | do |
|---|---|
| GPU: NVIDIA present but torch can't use it | `poetry run python -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132` |
| ffmpeg / Node not found | re-run the setup script, then restart the terminal |
| AI stars WARN | Ollama app not running (start "Ollama" from the Start menu) or model not pulled (`ollama pull qwen3-vl:30b`) |
| AI make model FAIL | `git pull` (it ships with the repo) |

## 4. Hand over

When doctor shows no FAIL, tell the user:
- From now on, put a session's videos in a folder and say
  `/make-highlights <that folder>` - it cuts every shot into clips, scores
  them, rates them, groups goals by player and renders three highlight reels
  (details and the optional review step are in that skill).
- Keep the PC plugged in and awake during a run (it also asks Windows not to
  sleep while running).
- Optional: offer to try it now on a short video they have.
