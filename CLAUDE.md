# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository & Remote

| 项目 | 值 |
| --- | --- |
| **仓库地址** | `https://github.com/wjk-121/yolov8-ultralytics-8.4.56` |
| **远程名称** | `origin` |
| **默认分支** | `main` |
| **所有者** | `wjk-121` |
| **项目根目录** | `d:/yolov8-ultralytics-8.4.56` (Windows) |

**常见 Git 操作：**

```bash
git add -A && git commit -m "message" && git push origin main
# 远程先行提交 (GitHub 网页操作) 导致冲突时:
git pull origin main --no-edit && git push origin main
# 本地完全覆盖远程 (谨慎):
git push -u origin main --force
```

## Project Overview

Fork of Ultralytics YOLOv8 (v8.4.56) with a custom Flask web application for inference. Chinese-language UI targeting PCB defect detection and autonomous vehicle competition use cases. License: AGPL-3.0.

Web app is at v5 iteration — polling-based video detection, camera session grouping, mixed image/video batch upload, standalone history detail pages, custom model support.

## Environment

- **操作系统**: Windows 11 Pro (10.0.26200)
- **Shell**: Bash (Git for Windows / MinGW) — 使用 Unix 风格路径 `/` 分隔
- **Conda 虚拟环境**: **`yolov11`** (主要运行环境)
  - 路径: `C:/Users/WJK-PC/anaconda3/envs/yolov11`
  - Python: 3.12.12
  - 激活: `conda activate yolov11`
  - 所有开发和运行操作都应在此环境下进行
- **其他 Conda 环境**:
  - `base` — `C:/Users/WJK-PC/anaconda3`
  - `cnn_pruning_GPU` — `C:/Users/WJK-PC/anaconda3/envs/cnn_pruning_GPU`
- **VS Code**: 配置了 Conda 作为默认 Python 环境管理器 (`.vscode/settings.json`)
- **Python 版本要求**: ≥ 3.8 (支持 3.8–3.12)

### 关键依赖 (yolov11 环境)

| 包名 | 版本 | 用途 |
| --- | --- | --- |
| `torch` | 2.7.0+cu128 | GPU 推理 (CUDA 12.8) |
| `torchvision` | 0.22.0+cu128 | 图像处理 |
| `ultralytics` | 8.4.56 | YOLO 核心 (可编辑安装, 指向项目根目录) |
| `opencv-python` | 4.12.0.88 | 视频/图像 I/O |
| `flask` | 3.1.3 | Web 框架 |
| `flask-cors` | 6.0.2 | 跨域支持 |
| `waitress` | 3.0.2 | WSGI 生产服务器 |
| `imageio-ffmpeg` | 0.6.0 | H.264 视频重编码 |
| `numpy` | 2.2.6 | 数值计算 |
| `pillow` | 12.0.0 | 图像处理 |

- **Windows 特定**: PyTorch 2.4.0 因 CPU 兼容性错误被排除

## Development Commands

```bash
# 启动 Web 服务器 (Flask + Waitress, 端口 5000)
python app/web_app.py

# 安装 ultralytics 包 (可编辑模式)
pip install -e .

# 安装 Web 应用依赖
pip install flask flask-cors waitress imageio-ffmpeg

# CLI 推理
yolo detect predict model=yolov8n.pt source=image.jpg
yolo segment predict model=yolov8n-seg.pt source=image.jpg

# 训练 / 导出 / 验证
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100
yolo export model=yolov8n.pt format=onnx
yolo detect val model=yolov8n.pt data=coco128.yaml
```

无测试套件。`pyproject.toml` 配置了 pytest 但 `tests/` 目录未包含。

## Runtime Directory Layout

以下目录由应用在运行时自动创建，已被 `.gitignore` 排除：

```text
models/         # 下载的 YOLO 模型权重 (.pt)
temp/           # 临时文件 (检测结果中间文件)
uploads/        # 用户上传文件
runs/detect/    # 检测历史记录 (metadata.json + 结果图片/视频)
data/samples/   # 示例图片缓存
.claude/        # Claude Code 本地配置
```

`static/img/logo.png` 也被 gitignore (`.png` 规则) — 用户自定义 logo 不进入版本控制。

## Architecture

### Layered Structure

1. **CLI/API** (`ultralytics/cfg/__init__.py`) — `entrypoint()` parses CLI args, dispatches by task+mode. Tasks: `detect`, `segment`, `classify`, `pose`, `obb`, `semantic`. Modes: `train`, `val`, `predict`, `export`, `benchmark`, `track`.

2. **Model abstraction** (`ultralytics/engine/model.py`) — `Model` base class wraps all operations: `train()`, `val()`, `predict()`, `export()`, `track()`. Instantiated via `YOLO("model.pt")`.

3. **Task implementations** (`ultralytics/models/yolo/{detect,segment,classify,pose,obb,semantic}/`) — Each directory contains `predict.py`, `train.py`, `val.py` with task-specific subclasses of the engine base classes.

4. **Neural network** (`ultralytics/nn/`) — YAML-driven model construction in `tasks.py` (`parse_model()` builds layers from YAML). Building blocks in `modules/` (conv.py, block.py, head.py). 16 inference backends in `backends/` via `autobackend.py`.

5. **Data pipeline** (`ultralytics/data/`) — `BaseDataset` → `YOLODataset`. Augmentation in `augment.py` (137KB). Loaders in `loaders.py`.

6. **Utilities** (`ultralytics/utils/`) — Metrics (`metrics.py`), loss functions (`loss.py`), NMS (`ops.py`, `nms.py`), plotting, distributed training, 10+ logging callback integrations.

7. **Solutions** (`ultralytics/solutions/`) — Ready-made CV apps: object counting, heatmaps, speed estimation, parking management, etc.

### Custom Application Layer (Web App v5)

**Backend** ([app/web_app.py](app/web_app.py)):

- Flask + Waitress on port 5000, CORS enabled
- 19 API endpoints covering: image/batch/URL/video/camera/sample detection, model selection, config management, history CRUD, file serving, sample pre-check
- **Config** class: `confidence` (default 0.25), `iou_threshold` (default 0.7, not exposed in UI). `image_size` removed — YOLO uses original image dimensions.
- **Model dictionary**: 25 YOLOv8 variants (detect/seg/pose/cls/obb × n/s/m/l/x) defined in [models_db.py](app/models_db.py), auto-download to `models/`. Custom `.pt` files placed in `models/` are auto-detected by `scan_local()` and listed under "自定义模型" category.
- **`_ensure_dirs()`** helper: re-creates `temp/`, `runs/detect/`, `uploads/` before every file-writing endpoint. Critical — the `temp/` directory can disappear between runs, causing FileNotFoundError on all file saves.
- **Video detection**: Polling pattern (not SSE — SSE is broken on Waitress/WSGI due to buffering and hop-by-hop header restrictions). `POST /api/detect/video` uploads + starts background `threading.Thread` processing, returns `job_id`. Frontend polls `GET /api/detect/video/status/<job_id>` at 300ms intervals. After frame processing completes, uses `imageio_ffmpeg` to re-encode from `mp4v` to H.264 (required for browser playback).
- **Async model download**: `YOLO("model.pt")` constructor downloads models synchronously over HTTP, which freezes the UI for ~90s on network errors. `load_model_safe()` spawns a background `_download_worker(name)` thread and returns immediately. Frontend polls `GET /api/models/download-status/<name>` (120s timeout) with animated progress dots, auto-completes on download finish. Job state tracked in `_download_jobs` dict.
- **Model delete protection**: `DELETE /api/models/<name>` refuses (409) if model is currently active. User must switch to another model first.
- **Custom model support**: `config.scan_local()` scans `models/*.pt`. `api_models()` adds non-MODELS files as custom entries. `config.model_info()` returns `{'name': n, 'desc': '自定义模型'}` for custom models. `api_select_model()` and `api_delete_model()` both accept custom model names if the file exists locally.
- **Sample images**: Downloaded to `data/samples/` via `requests`. `GET /api/samples/check/<name>` pre-checks local existence (~5ms); frontend shows spinner (local) vs progress bar (downloading). Samples only available in image detection mode.
- **`_create_video_writer()` helper**: Wraps H264 codec probing with `os.dup2` stderr → `/dev/null` redirection to suppress OpenH264 C-level warnings. Falls back to `mp4v` if H264 unavailable. All video writer call sites (batch + single video) must use this helper.
- **Batch detection**: Supports mixed image/video uploads. Async polling pattern: `POST /api/detect/batch` returns `job_id`, frontend polls `GET /api/detect/batch/status/<job_id>`. Per-file progress: indeterminate sweep bar for images, determinate percentage for videos. Shows "processing file X of Y: filename".
- **Camera sessions**: Multiple snapshots grouped into single history record (`_camera_sessions` dict). Finalized on stop, auto-expires after 30min timeout.
- **History**: File-system-backed via `runs/detect/<id>/metadata.json`. Lightweight polling endpoint `/api/history/check` returns `latest` (mtime) + `total` count. Frontend tracks `_latestCount` for change detection — triggers reload on count change (handles manual folder deletion). History detail served via standalone page at `/history/<id>` → `static/pages/history.html`
- **Structured logging**: `logging.basicConfig()` with format `'%(asctime)s  %(message)s'`. Log variable used across all endpoints for API calls, detections, model operations, config changes, history actions.

**Frontend** ([static/](static/)):

- [index.html](static/index.html) — SPA shell: navbar (brand icon with custom logo fallback, status badge), sidebar (model info, 5 detection modes, confidence slider only, history list), content area
- [js/app.js](static/js/app.js) — State management via `state` object, polling-based video/batch progress (300ms interval, ETA display), camera toggle with `toggleCamera()`, file size warning (>10MB), batch mixed file upload (`accept='image/*,video/*'`), async model download polling with `pollDownloadProgress()` (120s timeout, network retry feedback), media viewer with arrow-key navigation for batch results (`showBatchItemPreview()`, `navigateBatchPreview()`), smart sample loading with pre-check (`detectSample()`)
- [js/common.js](static/js/common.js) — Shared utilities: `escapeHtml()`, `toast()`, `showLoading()`/`showLoadingProgress()`/`updateLoadingProgress()`/`hideLoading()` (unified blur overlay with spinner/progress bar modes)
- [pages/history.html](static/pages/history.html) — Unified media viewer for history detail: `buildMediaItems()` constructs flat `mediaItems` array from snapshots, batch results, or single result; `selectMediaItem(index)` switches hero view with wrapping; `renderMediaViewerInner()` renders arrows + video/image; arrow-key navigation; old-record compatibility via regex extension correction (`.png/.bmp/.webp/.tiff` → `.jpg`)
- [css/style.css](static/css/style.css) — Apple-inspired design with CSS custom properties, responsive sidebar, smooth progress bar animation (indeterminate sweep), media viewer nav button styles (`.media-nav-btn`, `.media-prev/.media-next`), batch row hover/selected states, model active indicator (`.model-item.active` with left border accent + `.model-check` blue circle ✓), model delete button (`.model-del-btn` red tint with hover full-red)
- [img/logo.png](static/img/logo.png) — Custom logo (optional, 32×32). `index.html` loads `<img src="img/logo.png">` with SVG fallback via `onerror`. Place any PNG here to customize the brand icon.

**CLI Tool** ([app/yolo_app.py](app/yolo_app.py)):

- Independent terminal application with colored UI, 10 menu options, in-memory history (not disk-backed)

### Configuration System

- `ultralytics/cfg/default.yaml` — Global defaults for all training/validation/prediction parameters.
- `ultralytics/cfg/models/{v3,v5,v6,v8,v9,v10,11,12,26,rt-detr}/*.yaml` — Model architecture definitions. YAML specifies layer-by-layer construction parsed by `nn/tasks.py:parse_model()`.
- `ultralytics/cfg/datasets/*.yaml` — 42 dataset configs (paths, class names, augmentation).
- `ultralytics/cfg/trackers/*.yaml` — BoT-SORT and ByteTrack configs.

### Supported Model Generations

YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, YOLO26, RT-DETR, SAM/SAM3, FastSAM, NAS, YOLOWorld, YOLOE.

## Key Patterns

- **Lazy imports**: `ultralytics/__init__.py` uses `__getattr__` for lazy model class loading.
- **Settings persistence**: `ultralytics/utils/__init__.py` manages `SETTINGS` dict persisted to `~/.config/Ultralytics/settings.json`.
- **Monkey-patching**: `ultralytics/utils/patches.py` patches cv2, torch, and other libraries at import time.
- **Callback system**: `ultralytics/utils/callbacks/` — hook-based system for training lifecycle events (on_train_start, on_epoch_end, etc.). Integrations for TensorBoard, W&B, MLflow, ClearML, Comet, Neptune, DVC, Ray Tune.
- **Device auto-detection**: `ultralytics/utils/autodevice.py` and `autobatch.py` handle GPU/CPU selection and batch size optimization.
- **Media viewer navigation**: Unified `mediaItems` array pattern across batch preview (app.js), history detail (history.html), and camera snapshots. Clickable thumbnails/rows call `selectMediaItem(i)` / `showBatchItemPreview(i)`. Arrow-key navigation with wrapping: `(currentIndex + direction + total) % total`. Active state highlighting via CSS `.snapshot-card.active` / `.batch-result-row.selected`.

## Known Gotchas & Pitfalls

- **`temp/` directory fragility**: The `temp/` directory can disappear between server runs. Always use `_ensure_dirs()` before file writes, or call `mkdir(parents=True, exist_ok=True)`.
- **SSE does NOT work with Waitress**: Waitress buffers WSGI responses and rejects hop-by-hop headers (`Connection`). Do not use EventSource/SSE — use polling instead.
- **`mp4v` codec video won't play in browsers**: OpenCV's default MPEG-4 Part 2 codec is browser-incompatible. Requires `imageio-ffmpeg` for H.264 re-encoding. The video detection code auto-attempts this — check the `playable` flag in the response.
- **OpenH264 console noise**: `os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'` MUST be set BEFORE `import cv2` — ultralytics imports cv2 internally, and the FFMPEG backend initializes on first import. Additionally, `cv2.VideoWriter_fourcc(*'H264')` calls produce C-level stderr output. The `_create_video_writer()` helper wraps H264 probing with `os.dup2` stderr→`/dev/null` redirection. Always use this helper instead of directly creating video writers. Also call `cv2.setLogLevel(0)` after import for extra silencing.
- **YOLO `save=True` always outputs `.jpg`**: `model.predict(save=True)` saves ALL annotated images as `.jpg` regardless of source format (`.png`, `.bmp`, `.webp`, `.tiff`). When building result URLs from `Path` objects, use `tp.stem + '.jpg'` — never `tp.name`. Old history records may still reference original extensions; add regex fallback (`.png/.bmp/.webp/.tiff` → `.jpg`) for backward compatibility.
- **`YOLO()` constructor downloads synchronously**: When a model file is missing locally, `YOLO("model.pt")` triggers a synchronous HTTP download that retries 3 times (~90s). With GitHub unreachable in China, this freezes the entire request thread. Use `load_model_safe(name)` which spawns `_download_worker()` in a daemon thread + returns immediately. Frontend should poll `GET /api/models/download-status/<name>` (same polling pattern as video detection).
- **Inline onclick with innerHTML is fragile**: Use `addEventListener` or `DOMContentLoaded`-bound handlers instead. Quote escaping in innerHTML strings often breaks.
- **Video detection is blocking per frame**: Each YOLO `predict()` call on a single frame blocks. Keep the polling-based architecture — don't switch back to synchronous HTTP responses for video.
- **Windows Chinese filenames in paths**: Files with Chinese characters in names (e.g., "屏幕截图") work fine with `Path` objects, but avoid shell commands that might misinterpret encoding.

## Environment

- Python ≥ 3.8 (supports 3.8–3.12)
- Conda is the configured environment manager (see `.vscode/settings.json`)
- Core deps: PyTorch, OpenCV, NumPy, Matplotlib, PyYAML, Requests, SciPy, Polars
- Web app additionally requires: flask, flask-cors, waitress, imageio-ffmpeg (not in pyproject.toml)
- Windows-specific: PyTorch 2.4.0 excluded due to CPU errors
