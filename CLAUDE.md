# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

Ultralytics YOLOv8 (v8.4.56) — 一个计算机视觉框架，支持目标检测、实例分割、语义分割、图像分类、姿态估计、有向边界框检测和目标跟踪。采用 AGPL-3.0 许可证。

## 开发命令

### 安装（可编辑模式，包含开发依赖）

```bash
pip install -e ".[dev,export,solutions]"
```

### 运行测试

```bash
# 标准并行运行（带覆盖率报告）
pytest -n auto --dist=loadfile --cov=ultralytics/ --cov-report=xml tests/

# 包含慢速测试（通过 --slow 标志启用）
pytest -n auto --dist=loadfile --slow --cov=ultralytics/ tests/

# 运行单个测试文件
pytest tests/test_python.py -v -s

# 运行单个测试函数
pytest tests/test_python.py::test_function_name -v -s

# 仅运行 CUDA 测试
pytest tests/test_cuda.py -sv
```

### 代码检查与格式化

```bash
# Ruff（主要的 linter/formatter，行宽 120，Google 风格文档字符串）
ruff check ultralytics/
ruff format ultralytics/

# 拼写检查
codespell ultralytics/
```

## 架构设计

所有源代码位于 `ultralytics/` 目录下，采用分层架构：

### 配置层 (`ultralytics/cfg/`)

- `__init__.py` — CLI 入口点（`yolo`/`ultralytics` 命令），参数解析，配置验证
- `default.yaml` — 所有默认超参数（训练、验证、预测、导出、数据增强）
- `datasets/` — 40+ 个 YAML 数据集配置（COCO、VOC、ImageNet 等）
- `models/` — 按版本组织的 YAML 模型架构定义（v3–v26、rt-detr）

### 引擎层 (`ultralytics/engine/`)

核心框架类，其他所有模块都基于此构建：

- `model.py` — `Model` 基类（继承 `torch.nn.Module`）；统一的训练/验证/预测/导出/基准测试/调优 API
- `trainer.py` — `BaseTrainer`；训练循环、分布式数据并行（DDP）、指数移动平均（EMA）、检查点保存、学习率调度
- `validator.py` — `BaseValidator`；验证循环
- `predictor.py` — `BasePredictor`；推理管线
- `exporter.py` — `Exporter`；导出到 17+ 种格式（ONNX、TensorRT、CoreML 等）
- `results.py` — `Results`；检测/分割/姿态结果容器，含绘图功能

### 神经网络层 (`ultralytics/nn/`)

- `tasks.py` — 通过 `parse_model()` 从 YAML 配置构建模型；包含 `DetectionModel`、`SegmentationModel`、`ClassificationModel`、`PoseModel`、`OBBModel`、`SemanticSegmentationModel`、`WorldModel`、`YOLOEModel`
- `autobackend.py` — `AutoBackend`，统一所有导出格式的推理接口
- `modules/` — 构建模块：`block.py`（C2f、SPPF、CSP）、`head.py`（检测/分割/姿态头）、`conv.py`（Conv 变体）、`transformer.py`（注意力模块）
- `backends/` — 17 种后端特定的推理封装（PyTorch、ONNX、TensorRT、OpenVINO、CoreML、TFLite、NCNN 等）

### 模型实现 (`ultralytics/models/`)

每个任务都有独立子目录，包含任务特定的 Trainer/Validator/Predictor：

- `yolo/detect/`、`yolo/segment/`、`yolo/classify/`、`yolo/pose/`、`yolo/obb/`、`yolo/semantic/`
- `yolo/world/` — YOLO-World（开放词汇检测）
- `yolo/yoloe/` — YOLOE（视觉/文本提示检测/分割）
- `rtdetr/`、`sam/`、`fastsam/`、`nas/` — 替代模型架构

每个模型类中的 `task_map` 字典将任务名称映射到对应的 Model/Trainer/Validator/Predictor 类。

### 数据层 (`ultralytics/data/`)

- `dataset.py` — `BaseDataset`、`YOLODataset`
- `augment.py` — 所有数据增强变换（马赛克、混 MixUp、复制粘贴、HSV、几何变换）
- `loaders.py` — 图像/视频/流加载器
- `converter.py` — 格式转换器（COCO→YOLO、DOTA 等）

### 工具层 (`ultralytics/utils/`)

- `metrics.py` — AP、mAP、混淆矩阵、F1、P/R 曲线
- `loss.py` — 检测损失、分割损失、姿态损失、DFL
- `ops.py` — NMS、边界框变换、掩码操作
- `torch_utils.py` — PyTorch 工具、FLOPs 计算、性能分析、DDP
- `callbacks/` — 10 种回调集成（TensorBoard、W&B、MLflow、ClearML 等）

## CLI 使用模式

```bash
yolo TASK MODE ARGS
```

- 任务：`detect`、`segment`、`classify`、`pose`、`obb`、`semantic`
- 模式：`train`、`val`、`predict`、`export`、`track`、`benchmark`

```python
from ultralytics import YOLO

model = YOLO("yol26n.pt")
model.train(data="coco8.yaml", epochs=100)
model.predict("image.jpg")
```

## 代码风格

- **行宽：** 120 个字符
- **文档字符串：** Google 风格（`[tool.ruff.lint.pydocstyle] convention = "google"`）
- **格式化：** Ruff（无 pre-commit 钩子；CI 通过 `ultralytics/actions` 强制执行）
- **惰性导入：** 模型类（`YOLO`、`SAM`、`RTDETR` 等）通过 `ultralytics/__init__.py` 中的 `__getattr__` 惰性加载

## 关键约定

- 模型架构定义在 `ultralytics/cfg/models/` 下的 YAML 配置中，由 `nn/tasks.py:parse_model()` 解析
- `engine/model.py` 中的 `Model` 基类通过 `task_map` 分发到任务特定实现
- 训练默认值在 `ultralytics/cfg/default.yaml` 中；CLI 参数可覆盖这些默认值
- `utils/__init__.py` 中的 `RANK` 变量控制分布式训练行为（-1 表示单 GPU）
- `utils/__init__.py` 中的 `SETTINGS` 字典将用户偏好持久化到磁盘
- 测试使用 `isolated_model` 固件（将模型复制到临时目录）以避免 `pytest-xdist` 下的文件竞争
