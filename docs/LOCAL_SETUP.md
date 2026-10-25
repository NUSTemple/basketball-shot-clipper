# Running the whole pipeline locally (Windows / Linux / macOS)

Raw session videos in, then cut shot clips, AI make scores, AI star ratings,
goals grouped by player, and three highlight reels out - all on your own
machine, no cloud account needed. With Claude Code it's one sentence: open
this repo and say `/make-highlights <your video folder>`.

## 中文快速开始（Windows + NVIDIA 显卡 + Claude Code）

**需要：** Windows 10/11，NVIDIA 显卡并已装好最新驱动（RTX 50 系列可以），约 40GB 空闲硬盘。

1. **装 Git 和 Claude Code**（在 PowerShell 里执行）：
   ```powershell
   winget install --id Git.Git -e
   irm https://claude.ai/install.ps1 | iex
   ```
   装完关掉 PowerShell 再打开一个新的。第一次运行 `claude` 时按提示登录 Claude 账号。
2. **下载项目并打开 Claude Code**：
   ```powershell
   cd $HOME
   git clone https://github.com/NUSTemple/basketball-shot-clipper.git
   cd basketball-shot-clipper
   claude
   ```
3. **在 Claude Code 里说：** `/setup-local`
   Claude 会检查电脑、列出要安装的东西（Python、Poetry、ffmpeg、Node.js、Ollama 和
   约 20GB 的 AI 打星模型）并征求你同意，然后自动安装（20–60 分钟）。
4. **按提示重启 Claude Code**（输入 `/exit`，重新打开 PowerShell，进入项目文件夹，再运行 `claude`），
   让它再检查一次（`/setup-local`），直到全部 OK。
5. **以后每次打完球：** 把视频放进一个文件夹，在 Claude Code 里说
   `/make-highlights D:\Videos\20261024 Basketball Video\DJI_001`。
   跑完后得到：所有投篮片段（`clips`）、AI 星级、按球员分组，以及三个集锦视频（`reels`）。
6. （可选）在本地标注界面复核进球、打星、确认每组球员名字，再让 Claude 重跑一次
   `/make-highlights`，集锦会按你的标注重新挑片段（约 2 分钟）。

不用 Claude Code 也可以：在项目目录运行
`powershell -NoProfile -ExecutionPolicy Bypass -File installer\setup-local-pipeline.ps1`，
然后 `poetry run shot-clipper-highlights "<视频文件夹>"`。

## 1. Install the tools

| Tool | Windows | Linux (Ubuntu) | macOS |
|---|---|---|---|
| everything below | `installer\setup-local-pipeline.ps1` (or `/setup-local` in Claude Code) | | |
| Python 3.12+ and Poetry | `installer\setup-deps.ps1` does both (plus ffmpeg and CUDA torch) | `apt install python3 pipx && pipx install poetry` | `brew install python poetry` |
| ffmpeg | (setup-deps) or `winget install Gyan.FFmpeg` | `apt install ffmpeg` | `brew install ffmpeg` |
| Node.js 22 LTS | `winget install OpenJS.NodeJS.LTS` | from nodejs.org / nvm | `brew install node` |
| Ollama (AI stars) | `winget install Ollama.Ollama` | `curl -fsSL https://ollama.com/install.sh \| sh` | not needed - Macs use MLX |

Then, in the repo:

```bash
poetry install --with ml
poetry run shot-clipper-doctor
```

### NVIDIA GPUs (incl. RTX 5090)

On Windows, Poetry installs CPU-only torch from PyPI. RTX 50-series (Blackwell)
cards need a CUDA 12.8+ build, e.g.:

```bash
poetry run python -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
```

`installer\setup-deps.ps1` does this for you; `shot-clipper-doctor` checks it
(`GPU ... CUDA: NVIDIA GeForce RTX 5090`). More in [GPU_SETUP.md](GPU_SETUP.md).

### AI stars: the vision model

```bash
ollama pull qwen3-vl:30b
```

About 20GB, needs ~24GB+ of GPU memory (a 5090's 32GB is plenty). For a
smaller GPU set `SHOT_CLIPPER_OLLAMA_MODEL=qwen3-vl:8b` (faster, rougher).
Without any vision model the pipeline still runs and ranks goals by make
confidence alone.

## 2. Run it

```bash
poetry run shot-clipper-highlights "D:\Videos\20261024 Basketball Video\DJI_001"
```

Output goes next to the video folder in `auto_highlights\` (`--out <dir>` to
choose, e.g. the session folder so clips land in `<session>\clips`):

- `clips\<video>\shot_NNN.mp4` - every detected shot, with AI make scores
- `reels\top_plays.mp4`, `countdown.mp4`, `fast_cut.mp4` (+ contact sheets)
- `ai_stars.csv` - goals ranked, 1-5 AI stars, one-line description each
- `summary.json` - counts, star breakdown, goals by player

Every step resumes: re-running skips what's done. Keep the computer awake
for long runs (Windows: Power settings -> never sleep while plugged in).

## 3. Review and re-run (optional)

```bash
poetry run shot-clipper-label-ui --clips-dir "<out>\clips"
```

In **Review** mark goals/misses and stars; in **Players** confirm each
group's name once. Labels are saved in `data\dataset\labels.json`, and the
next `shot-clipper-highlights` run uses them: your misses never appear in a
reel and your 5-star goals come first. That re-run only re-renders (~2 min).

## What runs where

| Step | Model | Notes |
|---|---|---|
| hoop calibration, shot detection, shooter | RF-DETR (OpenHoopVision, downloaded on first use) | GPU |
| make / miss | `models/prelabel/global.joblib` (ships with the repo) | trained on several cameras |
| AI stars | Qwen3-VL via MLX (Mac) or Ollama | optional |
| who scored | OSNet-AIN person re-ID (downloaded on first use) | groups per session; names are suggestions you confirm |
| reels | Remotion (`remotion-reel/`, Node) | free for individuals / teams up to 3 |
