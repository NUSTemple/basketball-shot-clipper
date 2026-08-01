# shot-clipper

[English](README.md)

检测固定机位篮球视频中的命中投篮，并将每次命中剪辑成独立片段；附带一个本地网页界面，用于手动将这些片段标注为"进球"或"未进"，以构建训练数据集。

处理流程：为每个视频标定一次篮筐区域 -> 用 YOLO 球体检测 + 轨迹穿过篮筐的几何规则找出候选命中 -> 用 `ffmpeg` 把每个候选剪成独立片段 -> 可选地在标注界面中复核片段并导出 goal/no_goal 数据集。

完整的设计理由见 [docs/PLAN.md](docs/PLAN.md)。

**使用 Claude Code？** `.claude/skills/process-videos` 会自动完成下文描述的完整流程——"新视频进 -> 已评分精彩片段出"：指向一个存放新视频的文件夹，它会帮你跑检测/剪辑/过滤，然后交给标注界面复核，并帮你导出评分最高的片段。

## 安装

需要安装 [Poetry](https://python-poetry.org/) 和 `ffmpeg`（并确保在 `PATH` 中）。

```bash
# 仅标注界面（轻量 - 只需要 Flask）
poetry install

# 检测/剪辑流程（会加装 ultralytics/opencv/numpy）
poetry install --with ml

# + 测试工具
poetry install --with ml,dev
```

以下命令都假设在仓库根目录运行，因为默认路径（`data/`、`models/`、`clips/`）是相对当前目录解析的。

## 检测流程

标注界面的 "+ Process new video" 面板（见下文）已经把下面 1-3 步封装好了。如果你想写脚本批量处理，或者完全不想装标注界面，可以直接用这些命令。

```bash
# 1. 每个视频一次性标定篮筐 -> data/configs/<name>.json
poetry run shot-clipper-calibrate path/to/video.MP4

# 2. 检测候选命中 -> data/ground_truth/<name>_detected.json
poetry run shot-clipper-detect path/to/video.MP4

# 3. 把每个候选剪成独立片段 -> clips/<name>/shot_NNN.mp4
poetry run shot-clipper-clip path/to/video.MP4 data/ground_truth/<name>_detected.json

# 可选：把检测出的时间点与人工记录的列表做比对
poetry run shot-clipper-validate detected.json ground_truth.json
```

模型权重（`yolov8m.pt`）需要放在 `models/yolov8m.pt`；如果没有请从 Ultralytics 下载。`scripts/run_batch.sh` 会对一批视频批量跑第 2-3 步。`shot-clipper-train-filter`（见下文）另外需要 `models/yolov8l.pt`。

## 使用标注界面

一个本地网页应用，用来逐个查看候选片段并标注为 `goal` 或 `no_goal`，边看边生成 `data/dataset/labels.json`。这是日常使用本项目的主要方式——大多数人根本用不到上面那些原始 CLI 命令。

### 1. 启动

```bash
poetry install --with ml     # "+ Process new video" 面板需要；如果只标注已有片段可以跳过
poetry run shot-clipper-label-ui --clips-dir /path/to/clips
# 打开 http://127.0.0.1:5050
```

`--clips-dir` 指向存放片段的文件夹，其中每个源视频对应一个子文件夹（例如 `<clips-dir>/DJI_0010/shot_001.mp4`, ...）。不传的话会用 `$SHOT_CLIPPER_CLIPS_DIR` 或内置默认值。

也可以用 Docker 运行（只需要 Flask，没有 ML 依赖，所以跑不了下面的第 2 步，只能标注已经存在的片段）：

```bash
CLIPS_DIR=/path/to/clips docker compose up --build
```

### 2.（可选）把原始视频转成候选片段

如果 `--clips-dir` 里已经有片段，可以直接跳到第 3 步——应用会自动加载它们。如果想从新的源视频生成片段，且不想碰 CLI，可以：

1. 打开页面顶部的 **"+ Process new video"** 面板。"clips folder" 输入框默认是启动时的 `--clips-dir`——可以直接改成别的路径（比如换一个空文件夹开始新项目），应用会切换到浏览/标注那个文件夹，这样新剪出来的片段总是落在你能立刻看到并标注的地方。
2. 粘贴视频文件的完整路径，点击 **"Detect & cut clips"**。
3. 观察状态行——它会先跑球体检测（一段约 100 秒的 4K 视频大概需要几分钟），然后把每个候选命中剪成独立文件。页面会自动轮询进度；处理期间你可以继续标注其他片段。
4. 完成后，新片段会加入下方列表，处于未标注状态，可以开始标注。

这需要先给该视频做一次性篮筐标定（`poetry run shot-clipper-calibrate path/to/video.MP4`，会弹出一个 OpenCV 窗口，拖框圈出篮筐）——如果还没标定，面板会直接告诉你该跑哪条命令。

### 3. 标注片段——并为进球打分

当前片段的视频会自动播放并循环。用以下按键逐个处理：

| 按键 | 操作 |
|---|---|
| `G` | 标为 **goal**（会停在这个片段上，方便你打分） |
| `N` 或 `X` | 标为 **no goal**（自动跳到下一个） |
| `1`-`5` | 给这个片段打 N 星——如果还没标为 goal 会顺带标上，然后自动前进 |
| `0` | 清除星级评分（保留 goal 标签） |
| `Backspace` | 完全清除标签（以及评分） |
| `←` / `→` | 上一个 / 下一个片段 |
| `M` | 切换静音 |
| `R` | 从头重播 |
| `[` / `]` | 调慢 / 调快播放速度（0.25x-3x，页面顶部也有下拉框，设置会在切换片段时保留） |

星级（1-5）用来衡量这个进球有多精彩/值不值得放进剪辑——真正想放进视频的片段就打高分。一个片段只有在标为 `no_goal`，或者标为 `goal` 且已打分之后，才算"完成"——所以 `no_goal` 仍然会立刻自动前进，但标为 `goal` 后会停在原地，直到你按下数字键。"jump to next incomplete"（默认开启）会直接跳到还需要标签或评分的片段；顶部会显示 goal / no-goal / 待评分 / 未标注 的计数和进度条。每次操作都会立即保存到 `data/dataset/labels.json`——关掉标签页也不会丢，随时可以回来继续。

如果片段是用 `--filter-model` 剪出来的（见下文），每个片段都带着训练出的过滤器给出的置信度分数——会显示在标签徽章旁边，也可以勾选 **"sort by confidence"** 按分数排序，这样你会先看到最可能是真进球的片段，而不是按文件顺序一个个刷掉一长串误报。这只是辅助你排优先级，不是自动过滤——几何检测器仍然按设计会过量生成候选，最终判断 goal/no_goal 的还是你。

### 4. 导出数据集（或者只导出你的精彩片段）

标注了一些片段之后，可以把它们整理成扁平的 `goal/` / `no_goal/` 文件夹结构（默认用符号链接，所以是瞬间完成，也不会复制视频文件）。`goal` 片段的文件名会带上星级，例如 `DJI_0001_D__shot_012_5star.mp4`：

```bash
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips
# -> data/dataset/goal/, data/dataset/no_goal/

# 只导出评分最高的 goal 片段（比如用来剪视频）——no_goal 依然会完整导出
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips --min-stars 4

# 按星级分到 goal/5star/、goal/4star/ ... 子文件夹，而不是一个扁平文件夹
poetry run shot-clipper-build-dataset --clips-dir /path/to/clips --group-by-stars
```

像 CapCut 这样的视频剪辑软件并没有"自定义片段元数据/评分"这个概念，所以文件名后缀和 `--group-by-stars` 生成的子文件夹，就是评分信息能够"带进"剪辑软件的实际方式——大多数剪辑软件（包括 CapCut）导入文件夹时，会把子文件夹变成媒体面板里的独立分类/素材夹，这样你不用重新看一遍所有片段，就能直接找到 5 星片段。可以和 `--min-stars` 搭配使用，把完全不想要的低分片段直接排除在导出之外。

### 5. 导入前快速核查

`shot-clipper-contact-sheet` 会生成一个静态 HTML 页面——按文件夹分组的缩略图网格，方便你在导入剪辑软件之前快速扫一眼整批导出结果，不用打开标注界面，也不用在 Finder 里一个个翻：

```bash
poetry run shot-clipper-contact-sheet data/dataset/goal --recursive --cols 6
# -> data/dataset/goal/contact_sheet.html（缩略图放在旁边的文件夹里）
```

`--recursive` 会把 `--group-by-stars` 生成的子文件夹也纳入，每个子文件夹单独一个分区（5star、4star... 排在最前，其他的排在后面）。默认在每个片段的第 5 秒抓取缩略图（`--at`），正好对应默认 `[t-5s, t+2s]` 剪辑方式下投篮/命中发生的那一刻。

## 用训练出的过滤器提升精确率

`find_makes()`（判断球体轨迹是否像一次命中的几何规则）刻意做成召回优先（`docs/PLAN.md` 决策 5）——它宁可多生成候选，也不愿漏掉真实命中，这意味着它标出来的很多东西其实并不是进球。等标注了足够多的片段后，可以训练一个小分类器，把这些误报重新过滤掉：

```bash
poetry install --with ml
poetry run shot-clipper-train-filter --clips-dir /path/to/clips
```

这一步会为每个已标注片段提取两组互补的特征——轨迹特征（下落速度、在篮筐框内的停留时间、反弹信号等，见 `src/shot_clipper/features.py`）和净空运动特征（候选事件发生后篮网区域的像素运动——运动脉冲的形状、光流方向一致性，见 `src/shot_clipper/net_motion.py`，不需要球体检测），用留一视频交叉验证（leave-one-video-out）评估这个组合，并挑选一个能在保持召回率 >= 98%（可用 `--min-recall` 修改）的同时精确率最高的置信度阈值。它会打印每个视频的前后精确率对比报告，并把模型存到 `models/shot_filter.joblib`。

**在 307 个已标注片段的数据集上（9 个视频）的结果**：候选片段的基线精确率是 34.2%。要真正取得改善，有两件事很关键：
- 球体检测质量：默认的 `yolov8m` 检测器对特征提取来说太稀疏——19% 的真实进球完全没有可用的"篮筐上方 -> 穿过篮筐"轨迹，这就限制了任何过滤器能安全做到的上限。`shot-clipper-train-filter` 专门把特征提取的默认模型换成更大的 `models/yolov8l.pt`（真实进球中能拿到可用轨迹的比例从 81% 提升到 100%；但尚未验证它是否适合作为 `shot-clipper-detect` 本身的主检测模型）。
- 单独的净空运动特征几乎没有信号（34.3%，基本等于基线），但明显能和轨迹特征互补——两者结合后，在同样的召回率目标下，过滤器能安全丢弃的误报数量比只用轨迹特征时几乎翻倍。

在默认的 98% 召回率阈值下，目前部署的模型能把精确率从 **34.2% 提升到 41.7%**，丢弃 307 个候选中的 60 个，代价是 105 个真实进球里有 2 个存在被误删的风险。如果你愿意用更多召回率换精确率，能达到的上限还要高得多——用 `--min-recall <x>` 重新训练即可在这条曲线上选别的点；报告表格会精确列出每种设置下会丢弃多少候选、多少真实进球会有风险。

训练完成后，它会自动生效：
- `shot-clipper-clip --filter-model models/shot_filter.joblib` 会在剪出每个片段后立刻给它打分（净空运动特征需要实际像素，所以过滤是在剪辑之后而不是检测阶段进行的），并删除低于阈值的片段；被丢弃的片段及其分数会打印到标准输出。
- 只要 `models/shot_filter.joblib` 存在，标注界面的 "process new video" 面板就会自动应用它（在 `/api/process-video` 的请求体里传 `"use_filter": false` 可以针对某一次运行关闭它）。

## 测试

```bash
poetry install --with ml,dev
poetry run pytest
```

## 项目结构

```
src/shot_clipper/      可安装的包（CLI 工具 + label_ui Flask 应用）
data/configs/          每个视频的篮筐标定（已纳入版本控制）
data/ground_truth/      检测器输出 / 人工记录的命中时间点（已纳入版本控制）
data/dataset/          labels.json + 整理好的 goal/no_goal 文件夹
models/                 YOLO 权重（未纳入版本控制 - 见"安装"）
clips/                  生成的投篮片段（未纳入版本控制）
scripts/                批处理辅助脚本
docs/                   设计文档
tests/                  单元测试
.claude/skills/         Claude Code 技能（已纳入版本控制 - 见上文 process-videos）
```
