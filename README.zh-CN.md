# shot-clipper

[English](README.md)

检测固定机位篮球视频中的命中投篮，并将每次命中剪辑成独立片段；附带一个本地网页界面，用于手动将这些片段标注为"进球"或"未进"，以构建训练数据集。

处理流程：为每个视频标定一次篮筐区域 -> 用 YOLO 球体检测 + 轨迹穿过篮筐的几何规则找出候选命中 -> 用 `ffmpeg` 把每个候选剪成独立片段 -> 可选地在标注界面中复核片段并导出 goal/no_goal 数据集。

完整的设计理由见 [docs/PLAN.md](docs/PLAN.md)。

**使用 Claude Code？** `.claude/skills/process-videos` 会自动完成下文描述的完整流程——"新视频进 -> 已评分精彩片段出"：指向一个存放新视频的文件夹，它会帮你跑检测/剪辑/过滤，然后交给标注界面复核，并帮你导出评分最高的片段。

## 界面截图

**检测（Detect）** - 把原始视频变成候选片段，可在速度/召回率之间取舍：

![Detect 标签页](docs/screenshots/detect.png)

**任务状态（Job Status）** - 实时显示任务进度，并保留历史记录，方便用相同设置重新运行任意一次任务：

![Job Status 标签页](docs/screenshots/job_status.png)

**复核（Review）** - 对照 5 星评分指南给每个候选片段打分，同时做进球/未进的判定：

![Review 标签页](docs/screenshots/review.jpg)

**素材库（Library）** - 所有片段的缩略图时间线，按拍摄日期再按视频分组，方便快速检查或批量导出：

![Library 标签页](docs/screenshots/library.jpg)

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

标注界面的 **Detect** 页签（见下文）已经把下面 1-3 步封装好了。如果你想写脚本批量处理，或者完全不想装标注界面，可以直接用这些命令。

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
poetry install --with ml     # Detect 页签需要；如果只标注已有片段可以跳过
poetry run shot-clipper-label-ui --clips-dir /path/to/clips
# 打开 http://127.0.0.1:5050
```

`--clips-dir` 指向存放片段的文件夹，其中每个源视频对应一个子文件夹（例如 `<clips-dir>/DJI_0010/shot_001.mp4`, ...）。不传的话会用 `$SHOT_CLIPPER_CLIPS_DIR` 或内置默认值。

也可以用 Docker 运行：

```bash
CLIPS_DIR=/path/to/clips docker compose up --build
```

这会启动**两个**容器：`label-ui`（浏览器打开的网页应用）和 `worker`（专门跑检测任务）。之所以拆成两个，是为了让重启/重建网页容器不会误杀正在跑的检测任务——`worker` 是完全独立的容器和进程树，不只是 `label-ui` 内部的一个后台线程。`label-ui` 只负责把任务写进共享的 `data/jobs/` 卷里排队；`worker` 负责取出来跑。两个容器都装了 `ml` 依赖（torch/ultralytics/opencv），所以检测在这里是真的能跑起来的——不过是**纯 CPU**：macOS 上的 Docker 没有 GPU/MPS 直通，所以处理一段较长的 4K 视频会比原生运行明显慢很多（如果速度比精度更重要，可以看下文的"检测速度"选项）。

应用的侧边栏分四个区域：**Detect**（把原始视频转成候选片段）、**Job Status**（查看正在跑的任务进度）、**Review**（标注片段，见下文）、**Library**（按时间线展示所有片段——先按拍摄日期分组，再按视频分组——点击一个可跳转到复核页）。Review 导航项上会有一个徽章，显示还有多少片段没处理完；只要有任务在跑，Job Status 旁边就会出现一个小圆点（蓝色=进行中，绿色=完成，红色=出错），不用切过去也能一眼看出状态。侧边栏底部有一个语言切换（EN / 中文），覆盖整个界面，不只是标签文字——切换后会记在浏览器里，下次打开还是你选的那个语言。

### 2.（可选）把原始视频转成候选片段

如果 `--clips-dir` 里已经有片段，可以直接跳到第 3 步——应用会自动加载它们。如果想从新的源视频生成片段，且不想碰 CLI，切到 **Detect** 页签就行。"clips folder" 输入框默认是启动时的 `--clips-dir`——可以直接改成别的路径（比如换一个空文件夹开始新项目，或者点 **Browse…** 用原生文件夹选择框），应用会切换到浏览/标注那个文件夹，这样新剪出来的片段总是落在你能立刻看到并标注的地方。

这个页签有两种模式：
- **Single video**：输入单个视频文件的完整路径（可以直接打字、粘贴，也可以点 **Browse…** 打开原生文件选择框），然后点击 **"Detect & cut clips"**。
- **Batch folder**：输入一个存放多个新视频的文件夹路径，点击 **"Process all videos"**。每个已经标定过篮筐的视频都会被排队、**依次**处理（同时跑多个 YOLO 检测在个人电脑上只会互相抢资源）；还没标定的视频不会排队，而是在返回结果里报告为跳过。

不管哪种方式，一提交就会自动切到 **Job Status** 页签，显示实时状态消息和进度条——批量模式下还会有一张表格，实时列出每个视频的 makes/kept/dropped 数量，以及哪些视频因为没有标定被跳过了。**不用等整批处理完**：每个视频一处理完，它的片段就会落地并出现在 Review 里，你可以立刻切过去开始复核这一个，同时队列里剩下的视频继续在后台处理。

处理一个视频需要先给它做一次性篮筐标定。对于单个视频，直接点 Detect 页签里的 **🎯 标定篮筐** 按钮就行——它会用 `ffmpeg` 从视频里截一帧（所以在 Docker 里也能用），然后你可以直接在浏览器里拖框圈出篮筐，不用再单独跑一步命令。老的 CLI 方式（`poetry run shot-clipper-calibrate path/to/video.MP4`，弹出一个 OpenCV 窗口）依然可用，但必须**在 Mac 本机直接跑**（先 `poetry install --with ml`），不能在 Docker 容器里跑，因为它需要真正的显示环境。两种方式最终生成的都是同一种小文件——`data/configs/<video>.json`——所以在 Docker 里用应用内标定保存的结果，原生运行的实例立刻就能看到，反过来也一样。批量模式下，没标定的视频会被直接跳过，并在返回结果里列出名字——可以到单视频页签把它们逐个标定完，再重新跑一次批量。

**检测速度**：默认按 15fps 采样，这是唯一经过实际验证召回率的设置（见下文"提升精确率"）——一段真实的 10 分钟视频，处理时间完全可能超过视频本身的时长，在 Docker 里纯 CPU 跑的话更明显。Detect 页签里的"检测速度"卡片可以改成 8fps 或 5fps 采样，用一定的召回率换取实打实的速度（一次只在一两帧里露出球进网瞬间的进球，可能会被直接采样跳过）。检测开始后，Job Status 会实时显示进度百分比、已用时和预计剩余时间，还有一个**停止**按钮可以中途取消任务——可以放心用，它要么直接终止（还在排队的任务），要么在下一批帧处理完时干净地停下来（正在跑的任务），不会留下写到一半的片段文件。

**原生路径选择框**：因为这是本地应用，**Browse…** 按钮会弹出真正的 macOS 文件/文件夹选择框（通过 `osascript`），不用再手动打字或粘贴路径——顺带解决了一个实际会遇到的坑：从浏览器地址栏复制或从 Finder 拖拽得到的路径，可能会带着 `file://` 前缀、空格也变成 `%20`，现在这两种形式应用都能正常识别。在 `osascript` 完全跑不了的地方（比如 Docker 里没有 macOS 可调用），**Browse…** 会自动改用应用内置的文件夹浏览器，而不是直接报错。

这个内置浏览器会固定从一个"素材根目录"开始浏览，并且不会跳出这个目录——默认是 `/Users/pengtan/Videos`，这样就不会在 Docker 的 `/root`、`/etc` 这类无关的系统目录里瞎逛才能找到自己的视频。**把 `SHOT_CLIPPER_MEDIA_ROOT` 设成你自己视频实际存放的路径**（新拍的源视频和剪出来的片段一般都在这同一个目录下的不同子文件夹里）：

```bash
SHOT_CLIPPER_MEDIA_ROOT=/path/to/your/videos poetry run shot-clipper-label-ui --clips-dir /path/to/clips
# Docker 下则和 CLIPS_DIR 一起传：
SHOT_CLIPPER_MEDIA_ROOT=/path/to/your/videos CLIPS_DIR=/path/to/clips docker compose up --build
```

如果用 Docker，记得同步修改 `docker-compose.yml` 里的视频文件夹挂载（`VIDEO_DIR`，只读挂载）到同一个路径，不然浏览器里会什么都看不到。

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
| `[` / `]` | 调慢 / 调快播放速度（0.25x-3x） |

星级（1-5）用来衡量这个进球有多精彩/值不值得放进剪辑——真正想放进视频的片段就打高分。一个片段只有在标为 `no_goal`，或者标为 `goal` 且已打分之后，才算"完成"——所以 `no_goal` 仍然会立刻自动前进，但标为 `goal` 后会停在原地，直到你按下数字键。顶部的分段进度条能让你一眼看出 goal / 待评分 / no-goal 的比例，具体数字则显示在上方的小圆点标签里。齿轮图标（⚙）打开的设置面板里有 **jump to next incomplete**（默认开启——直接跳到还需要标签或评分的片段）、**sort by confidence**（见下文）和播放速度。每次操作都会立即保存到 `data/dataset/labels.json`——关掉标签页也不会丢，随时可以回来继续。

如果片段是用 `--filter-model` 剪出来的（见下文），每个片段都带着训练出的过滤器给出的置信度分数——会显示在标签徽章旁边，也可以在设置里勾选 **"sort by confidence"** 按分数排序，这样你会先看到最可能是真进球的片段，而不是按文件顺序一个个刷掉一长串误报。这只是辅助你排优先级，不是自动过滤——几何检测器仍然按设计会过量生成候选，最终判断 goal/no_goal 的还是你。

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
- 只要 `models/shot_filter.joblib` 存在，标注界面的 Detect 页签就会自动应用它（在 `/api/process-video` 的请求体里传 `"use_filter": false` 可以针对某一次运行关闭它）。

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
