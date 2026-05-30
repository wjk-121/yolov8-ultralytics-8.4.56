# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Fork of Ultralytics YOLOv8 (v8.4.56) with a custom Flask web application for inference. Chinese-language UI targeting PCB defect detection and autonomous vehicle competition use cases. License: AGPL-3.0.

## Development Commands

```bash
# Install in editable mode (required for development)
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"          # pytest, coverage, ruff
pip install -e ".[export]"       # ONNX, TensorRT, CoreML, etc.

# Run the web application (Flask + Waitress on port 5000)
python app/web_app.py

# CLI inference via ultralytics entrypoint
yolo detect predict model=yolov8n.pt source=image.jpg
yolo segment predict model=yolov8n-seg.pt source=image.jpg
yolo pose predict model=yolov8n-pose.pt source=image.jpg

# Training
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100

# Export model
yolo export model=yolov8n.pt format=onnx

# Validation
yolo detect val model=yolov8n.pt data=coco128.yaml
```

No test suite is present in this fork. The `pyproject.toml` configures pytest but the `tests/` directory was not included.

## Architecture

### Layered Structure

1. **CLI/API** (`ultralytics/cfg/__init__.py`) — `entrypoint()` parses CLI args, dispatches by task+mode. Tasks: `detect`, `segment`, `classify`, `pose`, `obb`, `semantic`. Modes: `train`, `val`, `predict`, `export`, `benchmark`, `track`.

2. **Model abstraction** (`ultralytics/engine/model.py`) — `Model` base class wraps all operations: `train()`, `val()`, `predict()`, `export()`, `track()`. Instantiated via `YOLO("model.pt")`.

3. **Task implementations** (`ultralytics/models/yolo/{detect,segment,classify,pose,obb,semantic}/`) — Each directory contains `predict.py`, `train.py`, `val.py` with task-specific subclasses of the engine base classes.

4. **Neural network** (`ultralytics/nn/`) — YAML-driven model construction in `tasks.py` (`parse_model()` builds layers from YAML). Building blocks in `modules/` (conv.py, block.py, head.py). 16 inference backends in `backends/` via `autobackend.py`.

5. **Data pipeline** (`ultralytics/data/`) — `BaseDataset` → `YOLODataset`. Augmentation in `augment.py` (137KB). Loaders in `loaders.py`.

6. **Utilities** (`ultralytics/utils/`) — Metrics (`metrics.py`), loss functions (`loss.py`), NMS (`ops.py`, `nms.py`), plotting, distributed training, 10+ logging callback integrations.

7. **Solutions** (`ultralytics/solutions/`) — Ready-made CV apps: object counting, heatmaps, speed estimation, parking management, etc.

### Custom Application Layer

- `app/web_app.py` — Flask server serving `static/` frontend. Endpoints for model selection, image upload, inference, and result history. Uses YOLO's native `save=True` for output.
- `app/yolo_app.py` — CLI management tool with colored terminal UI.
- `static/` — Frontend (index.html, app.js, style.css) — Chinese language UI.

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

## Environment

- Python ≥ 3.8 (supports 3.8–3.12)
- Conda is the configured environment manager (see `.vscode/settings.json`)
- Core deps: PyTorch, OpenCV, NumPy, Matplotlib, PyYAML, Requests, SciPy, Polars
- Web app additionally requires: flask, flask-cors, waitress (not in pyproject.toml)
- Windows-specific: PyTorch 2.4.0 excluded due to CPU errors
