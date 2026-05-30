# -*- coding: utf-8 -*-
"""YOLOv8 共享模型注册表 — web_app.py 与 yolo_app.py 共用"""

# ── 模型变体 ────────────────────────────────────────────────────────────────
_VARIANTS = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'l': 'Large', 'x': 'XLarge'}

# ── 任务定义 ────────────────────────────────────────────────────────────────
_TASKS = {
    'detect': {'category': '检测模型', 'desc': '目标检测',     'cls': 80,   'params': {'n': '3.2M', 's': '11.2M', 'm': '25.9M', 'l': '43.7M', 'x': '68.2M'}},
    'seg':    {'category': '分割模型', 'desc': '实例分割',     'cls': 80,   'params': {'n': '3.4M', 's': '11.8M', 'm': '27.3M', 'l': '46.0M', 'x': '71.8M'}},
    'pose':   {'category': '姿态估计', 'desc': '姿态估计',     'cls': 17,   'params': {'n': '3.3M', 's': '11.4M', 'm': '26.4M', 'l': '44.4M', 'x': '69.4M'}},
    'cls':    {'category': '分类模型', 'desc': '图像分类',     'cls': 1000, 'params': {'n': '2.7M', 's': '6.4M',  'm': '17.0M', 'l': '37.5M', 'x': '57.4M'}},
    'obb':    {'category': '旋转检测', 'desc': '旋转目标检测', 'cls': 15,   'params': {'n': '3.3M', 's': '11.5M', 'm': '26.4M', 'l': '44.5M', 'x': '69.5M'}},
}

# ── 构建模型字典 ────────────────────────────────────────────────────────────
MODELS = {}
for _sfx, _tk in _TASKS.items():
    for _vr, _vn in _VARIANTS.items():
        _key = f'yolov8{_vr}' if _sfx == 'detect' else f'yolov8{_vr}-{_sfx}'
        _suffix = ' ' + _tk['desc'] if _sfx != 'detect' else ''
        MODELS[_key] = {
            'name': f'YOLOv8 {_vn}{_suffix}',
            'desc': f'{_vn}级{_tk["desc"]}模型',
            'params': _tk['params'][_vr],
            'num_classes': str(_tk['cls']),
            'category': _tk['category'],
        }

# ── 按类别分组 ──────────────────────────────────────────────────────────────
MODEL_CATEGORIES = {_tk['category']: [] for _tk in _TASKS.values()}
for _sfx, _tk in _TASKS.items():
    for _vr in _VARIANTS:
        _key = f'yolov8{_vr}' if _sfx == 'detect' else f'yolov8{_vr}-{_sfx}'
        MODEL_CATEGORIES[_tk['category']].append(_key)

# 别名 — web_app.py 使用 CATS 这个名字
CATS = MODEL_CATEGORIES
