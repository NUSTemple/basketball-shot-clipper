# Running the whole pipeline locally (Windows / Linux / macOS)

Raw session videos in, then cut shot clips, AI make scores, AI star ratings,
goals grouped by player, and three highlight reels out - all on your own
machine, no cloud account needed. With Claude Code it's one sentence: open
this repo and say `/make-highlights <your video folder>`.

## 中文快速开始

1. 安装：Git、Python 3.12+、Poetry、ffmpeg、Node.js 22 LTS、（可选）Ollama。
   Windows 上可直接运行 `installer\setup-deps.ps1`（会装好 CUDA 版 torch）。
2. 在仓库目录运行 `poetry install --with ml`，然后 `poetry run shot-clipper-doctor`，
   按提示把 FAIL 项修好。RTX 50 系列显卡需要 CUDA 12.8+ 的 torch（doctor 会给出命令）。
3. AI 打星需要视觉模型：装好 Ollama 后运行 `ollama pull qwen3-vl:30b`（约 20GB，
   5090 的 32GB 显存足够）。不装也能跑，只是星级会粗糙很多。
4. 在 Claude Code 里打开本仓库，说：`/make-highlights D:\Videos\20261024 Basketball Video\DJI_001`。
   跑完后片段在 `clips/`，三个集锦视频在 `reels/`。
5. （可选）用本地标注界面复核进球、打星、确认每组球员名字，然后再让 Claude 重跑一次
   `/make-highlights`——集锦会按你的标注重新挑片段（约 2 分钟）。

## 1. Install the tools

| Tool | Windows | Linux (Ubuntu) | macOS |
|---|---|---|---|
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
