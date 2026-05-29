import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── 依赖导入 ──────────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ============================================================================
#  终端颜色
# ============================================================================

class C:
    """终端 ANSI 颜色和样式"""

    BLACK, RED, GREEN, YELLOW = '\033[30m', '\033[31m', '\033[32m', '\033[33m'
    BLUE, MAGENTA, CYAN, WHITE = '\033[34m', '\033[35m', '\033[36m', '\033[37m'
    BRIGHT_RED, BRIGHT_GREEN, BRIGHT_YELLOW = '\033[91m', '\033[92m', '\033[93m'
    BRIGHT_BLUE, BRIGHT_MAGENTA, BRIGHT_CYAN = '\033[94m', '\033[95m', '\033[96m'
    BOLD, DIM, ITALIC, UNDERLINE = '\033[1m', '\033[2m', '\033[3m', '\033[4m'
    RESET = '\033[0m'

    @classmethod
    def init(cls):
        """Windows 终端 ANSI 支持初始化"""
        if sys.platform == 'win32':
            os.system('')


C.init()


# ============================================================================
#  UI 工具函数
# ============================================================================

def clear_screen():
    os.system('cls' if sys.platform == 'win32' else 'clear')


def _sep(char='=', length=70, color=C.DIM):
    print(f'{color}{char * length}{C.RESET}')


def _header(title, color=C.CYAN):
    print(f'\n{color}{"=" * 70}{C.RESET}')
    print(f'{color}{C.BOLD}  {title}{C.RESET}')
    print(f'{color}{"=" * 70}{C.RESET}\n')


def ok(msg):
    print(f'{C.GREEN}{C.BOLD}✓ {msg}{C.RESET}')


def err(msg):
    print(f'{C.RED}{C.BOLD}✗ {msg}{C.RESET}')


def warn(msg):
    print(f'{C.YELLOW}{C.BOLD}⚠ {msg}{C.RESET}')


def info(msg):
    print(f'{C.BLUE}{C.BOLD}ℹ {msg}{C.RESET}')


def loading(msg):
    print(f'{C.MAGENTA}{C.BOLD}⟳ {msg}{C.RESET}')


BANNER = f"""{C.CYAN}{C.BOLD}
╔════════════════════════════════════════════════════════════════════════════╗
║              ███████╗██╗   ██╗██╗      ██████╗  ██████╗                    ║
║              ╚══██╔╝╚██╗ ██╔╝██║     ██╔═══██╗██╔═══██╗                   ║
║                 ██║   ╚████╔╝ ██║     ██║   ██║██║   ██║                   ║
║                 ██║    ╚██╔╝  ██║     ██║   ██║██║   ██║                   ║
║                 ██║     ██║   ███████╗╚██████╔╝╚██████╔╝                   ║
║                 ╚═╝     ╚═╝   ╚══════╝ ╚═════╝  ╚═════╝                   ║
║                                                                            ║
║                    YOLOv8 交互式目标检测应用 v2.0                           ║
║                    支持自动检测本地模型文件                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
{C.RESET}"""


# ── 用户输入函数 ────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = '', valid: Optional[List[str]] = None) -> str:
    """获取用户输入，支持默认值和选项校验"""
    while True:
        hint = f'{C.CYAN}{prompt} [{default}]: {C.RESET}' if default else f'{C.CYAN}{prompt}: {C.RESET}'
        raw = input(hint).strip()
        if not raw:
            if default:
                return default
            warn('输入不能为空，请重新输入')
            continue
        if valid and raw.lower() not in {v.lower() for v in valid}:
            err(f'无效选项，请选择: {", ".join(valid)}')
            continue
        return raw


def _ask_int(prompt: str, default: int, lo: int = 0, hi: int = 10000) -> int:
    while True:
        raw = input(f'{C.CYAN}{prompt} [{default}]: {C.RESET}').strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            err(f'请输入 {lo} 到 {hi} 之间的数字')
        except ValueError:
            err('请输入有效的整数')


def _ask_float(prompt: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    while True:
        raw = input(f'{C.CYAN}{prompt} [{default}]: {C.RESET}').strip()
        if not raw:
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            err(f'请输入 {lo} 到 {hi} 之间的数字')
        except ValueError:
            err('请输入有效的数字')


def confirm(prompt: str, default: bool = True) -> bool:
    suffix = '[Y/n]' if default else '[y/N]'
    raw = input(f'{C.YELLOW}{prompt} {suffix}: {C.RESET}').strip().lower()
    if not raw:
        return default
    return raw in {'y', 'yes', '是', '1', 'true'}


def print_table(headers: List[str], rows: List[List[str]], title: str = ''):
    if title:
        print(f'\n{C.BOLD}{C.CYAN}  {title}{C.RESET}\n')
    widths = [max(len(h), *(len(str(c)) for c in col)) for h, *col in zip(headers, *[(r + [''] * (len(headers) - len(r)))[:len(headers)] for r in rows])]
    header_line = '  '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f'{C.BOLD}{C.WHITE}{header_line}{C.RESET}')
    print(f'{C.DIM}{"─" * (sum(widths) + 2 * (len(headers) - 1))}{C.RESET}')
    for row in rows:
        print('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    print()


# ============================================================================
#  模型信息数据库
# ============================================================================

_MODEL_VARIANTS = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'l': 'Large', 'x': 'XLarge'}
_MODEL_TASKS = {
    'detect':  {'category': '检测模型',    'desc': '目标检测',       'cls': 80,  'params': {'n': '3.2M', 's': '11.2M', 'm': '25.9M', 'l': '43.7M', 'x': '68.2M'}},
    'seg':     {'category': '分割模型',    'desc': '实例分割',       'cls': 80,  'params': {'n': '3.4M', 's': '11.8M', 'm': '27.3M', 'l': '46.0M', 'x': '71.8M'}},
    'pose':    {'category': '姿态估计',    'desc': '姿态估计',       'cls': 17,  'params': {'n': '3.3M', 's': '11.4M', 'm': '26.4M', 'l': '44.4M', 'x': '69.4M'}},
    'cls':     {'category': '分类模型',    'desc': '图像分类',       'cls': 1000,'params': {'n': '2.7M', 's': '6.4M',  'm': '17.0M', 'l': '37.5M', 'x': '57.4M'}},
    'obb':     {'category': '旋转检测',    'desc': '旋转目标检测',   'cls': 15,  'params': {'n': '3.3M', 's': '11.5M', 'm': '26.4M', 'l': '44.5M', 'x': '69.5M'}},
}

# 构建模型信息表（通过后缀和变体组合生成，避免重复书写）
MODELS: Dict[str, Dict[str, str]] = {}
for suffix, task in _MODEL_TASKS.items():
    for variant, variant_name in _MODEL_VARIANTS.items():
        model_key = f'yolov8{variant}' if suffix == 'detect' else f'yolov8{variant}-{suffix}'
        MODELS[model_key] = {
            'name': f'YOLOv8 {variant_name}{" " + task["desc"] if suffix != "detect" else ""}',
            'desc': f'{variant_name}级{task["desc"]}模型',
            'params': task['params'][variant],
            'num_classes': str(task['cls']),
        }

MODEL_CATEGORIES = {task['category']: [] for task in _MODEL_TASKS.values()}
for suffix, task in _MODEL_TASKS.items():
    for variant in _MODEL_VARIANTS:
        model_key = f'yolov8{variant}' if suffix == 'detect' else f'yolov8{variant}-{suffix}'
        MODEL_CATEGORIES[task['category']].append(model_key)


# ============================================================================
#  配置管理
# ============================================================================

class Config:
    """应用配置管理器"""

    DEFAULTS: Dict[str, Any] = {
        'model': 'yolov8n', 'confidence': 0.25, 'iou_threshold': 0.7,
        'image_size': 640, 'max_detections': 300, 'save_results': True,
        'show_labels': True, 'show_conf': True, 'line_width': 2,
        'models_dir': './models',
    }

    def __init__(self):
        self._cfg = self.DEFAULTS.copy()
        self.model: Optional[YOLO] = None
        self._local_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def models_dir(self) -> Path:
        return Path(self._cfg['models_dir'])

    def __getattr__(self, name):
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(name)

    def get(self, key): return self._cfg.get(key)
    def set(self, key, value): self._cfg[key] = value
    def model_name(self) -> str: return self._cfg['model']
    def model_info(self, name: str = '') -> Dict[str, Any]:
        return MODELS.get(name or self._cfg['model'], {})

    # ── 本地模型扫描 ──────────────────────────────────────────────────────

    def scan_local(self) -> Dict[str, Dict[str, Any]]:
        """扫描 models_dir 下的 .pt 文件，更新缓存"""
        self._local_cache.clear()
        d = self.models_dir
        d.mkdir(parents=True, exist_ok=True)
        for pt in d.glob('*.pt'):
            sz = pt.stat().st_size / (1024 * 1024)
            self._local_cache[pt.stem] = {
                'path': str(pt), 'size_mb': round(sz, 1),
                'modified': datetime.fromtimestamp(pt.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            }
        return self._local_cache

    def is_local(self, name: str) -> bool:
        return name in self._local_cache or (self.models_dir / f'{name}.pt').exists()

    def model_path(self, name: str) -> str:
        if name in self._local_cache:
            return self._local_cache[name]['path']
        p = self.models_dir / f'{name}.pt'
        if p.exists():
            return str(p)
        return f'{name}.pt'  # 让 YOLO 自动下载


# ============================================================================
#  核心应用
# ============================================================================

class App:
    """YOLOv8 交互式应用"""

    # ── 菜单定义 ──────────────────────────────────────────────────────────

    MAIN_MENU = [
        ('1', '🖼️  单张图片检测', '对单张图片进行目标检测'),
        ('2', '📁 批量图片检测', '对文件夹中的多张图片进行检测'),
        ('3', '🎥 视频文件检测', '对视频文件进行目标检测'),
        ('4', '📷 实时摄像头检测', '使用摄像头进行实时检测'),
        ('5', '🔗 网络图片/视频检测', '对网络URL进行检测'),
        ('6', '🔄 切换模型', '选择并切换检测模型'),
        ('7', '⚙️  参数配置', '调整检测参数'),
        ('8', '📊 查看检测历史', '查看历史检测记录'),
        ('9', '❓ 帮助说明', '查看使用帮助'),
        ('0', '🚪 退出程序', '退出应用'),
    ]

    CONFIG_MENU = [
        ('1', '调整置信度阈值'),
        ('2', '调整 IoU 阈值'),
        ('3', '调整图像尺寸'),
        ('4', '设置模型目录'),
        ('5', '恢复默认配置'),
        ('0', '返回主菜单'),
    ]

    VALID_MAIN = [k for k, _, _ in MAIN_MENU]
    VALID_CFG  = [k for k, _ in CONFIG_MENU]

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def __init__(self):
        self.cfg = Config()
        self.running = True
        self.history: List[Dict[str, Any]] = []

    def run(self):
        try:
            clear_screen()
            print(BANNER)
            self._init_models()
            self._main_loop()
        except KeyboardInterrupt:
            print(); warn('程序被用户中断')
        except Exception as e:
            err(f'程序发生错误: {e}')
            import traceback; traceback.print_exc()
        finally:
            print(); info('程序已退出')

    def _init_models(self):
        """启动初始化：扫描本地模型 → 加载默认模型"""
        _header('初始化')
        loading('正在扫描本地模型文件...')
        local = self.cfg.scan_local()
        if local:
            ok(f'在本地找到 {len(local)} 个模型文件')
            for name, data in local.items():
                print(f'  {C.GREEN}✓{C.RESET} {name} ({data["size_mb"]}MB)')
        else:
            info('本地未找到模型文件，将使用默认模型')
        print()
        if not self._load_model():
            err('模型加载失败，程序退出')
            self.running = False

    # ── 主循环 ────────────────────────────────────────────────────────────

    def _main_loop(self):
        while self.running:
            try:
                self._show_main_menu()
                choice = _ask('请选择功能', valid=self.VALID_MAIN)
                print()
                actions: Dict[str, Callable] = {
                    '1': self.detect_image, '2': self.detect_batch,
                    '3': self.detect_video, '4': self.detect_camera,
                    '5': self.detect_url,   '6': self._switch_model,
                    '7': self._config_loop, '8': self.show_history,
                    '9': self.show_help,    '0': self._exit,
                }
                action = actions.get(choice)
                if action:
                    action()
                if self.running and choice != '0':
                    print(); input(f'{C.DIM}按回车键返回主菜单...{C.RESET}')
            except KeyboardInterrupt:
                print(); warn('操作已中断')
                if not confirm('是否返回主菜单？'):
                    self._exit()

    def _config_loop(self):
        while True:
            self._show_config()
            c = _ask('请选择配置项', '0', self.VALID_CFG)
            if c == '1': self._adj('confidence', '置信度阈值', _ask_float)
            elif c == '2': self._adj('iou_threshold', 'IoU 阈值', _ask_float)
            elif c == '3': self._adj('image_size', '图像尺寸', lambda p, d: _ask_int(p, d, 320, 1920))
            elif c == '4':
                d = _ask('新的模型目录', self.cfg.models_dir)
                self.cfg.set('models_dir', d); self.cfg.scan_local()
                ok(f'模型目录已设置为: {d}')
            elif c == '5': self._restore_defaults()
            elif c == '0': return

    def _adj(self, key, label, fn):
        v = fn(f'新的{label}', self.cfg.get(key))
        self.cfg.set(key, v); ok(f'{label}已设置为: {v}')

    # ── 模型管理 ──────────────────────────────────────────────────────────

    def _load_model(self, name: Optional[str] = None) -> bool:
        name = name or self.cfg.model_name()
        models_dir = self.cfg.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        is_local = self.cfg.is_local(name)

        try:
            if is_local:
                path = self.cfg.model_path(name)
                loading(f'正在加载本地模型 {name}...'); info(f'模型路径: {path}')
                self.cfg.model = YOLO(path)
            else:
                warn(f'模型 {name} 未在本地找到，将自动下载...')
                info('下载过程可能需要几分钟，请耐心等待')
                self.cfg.model = YOLO(f'{name}.pt')

                # 移动到 models 目录
                downloaded = Path(f'{name}.pt')
                target = models_dir / f'{name}.pt'
                if downloaded.exists():
                    shutil.move(str(downloaded), str(target))
                    ok(f'模型已保存到: {target}')
                else:
                    warn('未找到下载的模型文件，可能已被缓存到其他位置')
                self.cfg.scan_local()

            mi = MODELS.get(name, {})
            ok(f'模型加载成功: {mi.get("name", name)}')
            if hasattr(self.cfg.model, 'names'):
                info(f'模型支持 {len(self.cfg.model.names)} 个目标类别')
            self.cfg.set('model', name)
            return True
        except Exception as e:
            err(f'模型加载失败: {e}')
            if not is_local:
                info('请检查网络连接，或手动下载模型到 ./models 目录')
            return False

    def _switch_model(self, name: Optional[str] = None):
        if name is None:
            models = self._display_model_list()
            print(f'{C.BOLD}提示:{C.RESET}')
            print(f'  {C.GREEN}[✓ 已存在]{C.RESET} - 已下载到本地，可直接使用')
            print(f'  {C.YELLOW}[需下载]{C.RESET} - 需从网络下载，首次使用需等待\n')
            name = self._select_model(models)
            if name is None:
                return
        self._show_model_detail(name)
        if not confirm('确认使用此模型？'):
            return
        if self._load_model(name):
            ok('模型切换成功！')
        else:
            err('模型切换失败')

    def _display_model_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        _header('可用模型列表')
        local = self.cfg.scan_local()
        all_models = []
        idx = 1
        for cat, names in MODEL_CATEGORIES.items():
            print(f'\n  {C.BOLD}{C.MAGENTA}▸ {cat}{C.RESET}')
            print(f'  {C.DIM}{"─" * 60}{C.RESET}')
            for name in names:
                mi = MODELS.get(name, {})
                is_local = name in local
                status = f'{C.GREEN}{C.BOLD}[✓ 已存在]{C.RESET}' if is_local else f'{C.YELLOW}[需下载]{C.RESET}'
                sz = f'({local[name]["size_mb"]}MB)' if is_local else ''
                print(f'  {C.CYAN}{idx:2d}{C.RESET}. {status} {C.BOLD}{name}{C.RESET} {sz}')
                print(f'      {C.DIM}{mi.get("name", "?")} - {mi.get("desc", "")}{C.RESET}')
                print(f'      {C.DIM}参数: {mi.get("params", "N/A")} | 类别数: {mi.get("num_classes", "N/A")}{C.RESET}')
                all_models.append((name, {'info': mi, 'is_local': is_local}))
                idx += 1
        _sep('-')
        local_cnt = sum(1 for _, d in all_models if d['is_local'])
        print(f'\n  {C.BOLD}模型统计:{C.RESET}')
        print(f'  {C.GREEN}✓ 已下载: {local_cnt} 个{C.RESET}')
        print(f'  {C.YELLOW}⬇ 需下载: {len(all_models) - local_cnt} 个{C.RESET}')
        print(f'  {C.CYAN}  共 {len(all_models)} 个{C.RESET}')
        return all_models

    def _select_model(self, models: List[Tuple[str, Dict[str, Any]]]) -> Optional[str]:
        while True:
            try:
                raw = input(f'{C.CYAN}请选择模型 (1-{len(models)}) 或输入模型名称: {C.RESET}').strip()
                if not raw:
                    warn('请输入有效的选择'); continue
                try:
                    n = int(raw)
                    if 1 <= n <= len(models):
                        return models[n - 1][0]
                    err(f'请输入 1 到 {len(models)} 之间的数字')
                except ValueError:
                    name = raw.lower()
                    if name in MODELS:
                        return name
                    err(f'未知模型: {name}')
                    info('请从列表中选择或输入有效的模型名称')
            except KeyboardInterrupt:
                print(); return None

    def _show_model_detail(self, name: str):
        mi = MODELS.get(name, {})
        is_local = self.cfg.is_local(name)
        print(f'\n  {C.BOLD}选择的模型:{C.RESET}')
        for k, v in [('名称', name), ('全名', mi.get('name', '?')), ('描述', mi.get('desc', '')), ('参数', mi.get('params', ''))]:
            print(f'  {C.CYAN}•{C.RESET} {k}: {v}')
        if is_local:
            print(f'  {C.CYAN}•{C.RESET} 状态: {C.GREEN}✓ 已存在于本地{C.RESET}')
            print(f'  {C.CYAN}•{C.RESET} 路径: {self.cfg.model_path(name)}')
        else:
            print(f'  {C.CYAN}•{C.RESET} 状态: {C.YELLOW}⬇ 需要下载{C.RESET}')
        print()

    # ── 菜单显示 ──────────────────────────────────────────────────────────

    def _show_main_menu(self):
        _header('主菜单')
        model = self.cfg.model_name()
        mi = self.cfg.model_info()
        is_local = self.cfg.is_local(model)
        status = f'{C.GREEN}✓ 已加载{C.RESET}' if is_local else f'{C.YELLOW}⬇ 需下载{C.RESET}'
        print(f'  {C.BOLD}当前模型:{C.RESET}')
        print(f'  {C.CYAN}•{C.RESET} 模型: {C.BOLD}{model}{C.RESET} ({mi.get("name", "?")})')
        print(f'  {C.CYAN}•{C.RESET} 状态: {status}\n')
        for k, t, d in self.MAIN_MENU:
            print(f'  {C.BOLD}{C.CYAN}{k}{C.RESET}  {t}')
            print(f'      {C.DIM}{d}{C.RESET}')
        print(f'\n{C.DIM}{"-" * 70}{C.RESET}')

    def _show_config(self):
        _header('参数配置')
        print(f'  {C.BOLD}当前配置:{C.RESET}\n')
        model = self.cfg.model_name()
        mi = self.cfg.model_info()
        status = f'{C.GREEN}✓ 已加载{C.RESET}' if self.cfg.is_local(model) else f'{C.YELLOW}⬇ 需下载{C.RESET}'
        for name, value, desc in [
            ('模型', f'{model} ({mi.get("name", "?")})', status),
            ('置信度阈值', str(self.cfg.confidence), '检测置信度阈值 (0-1)'),
            ('IoU 阈值', str(self.cfg.iou_threshold), 'NMS IoU 阈值 (0-1)'),
            ('图像尺寸', str(self.cfg.image_size), '推理图像大小 (像素)'),
            ('最大检测数', str(self.cfg.max_detections), '每张图片最大检测数量'),
            ('保存结果', '是' if self.cfg.save_results else '否', '保存检测结果'),
            ('显示标签', '是' if self.cfg.show_labels else '否', '显示类别标签'),
            ('显示置信度', '是' if self.cfg.show_conf else '否', '显示置信度'),
            ('线宽', str(self.cfg.line_width), '检测框线宽'),
            ('模型目录', str(self.cfg.models_dir), '本地模型存储目录'),
        ]:
            print(f'  {C.CYAN}•{C.RESET} {name}: {C.BOLD}{value}{C.RESET}')
            print(f'    {C.DIM}{desc}{C.RESET}')
        print(f'\n{C.DIM}{"-" * 70}{C.RESET}')
        print(f'\n  {C.BOLD}配置选项:{C.RESET}\n')
        for k, t in self.CONFIG_MENU:
            print(f'  {C.CYAN}{k}{C.RESET}  {t}')
        print()

    def _restore_defaults(self):
        if confirm('确认恢复默认配置？'):
            self.cfg = Config(); ok('已恢复默认配置')

    # ── 共享检测逻辑 ──────────────────────────────────────────────────────

    def _run_detect(self, mode: str, source, title: str, conf: Optional[float] = None,
                    stream: bool = False, show: bool = False):
        if conf is None:
            conf = _ask_float('置信度阈值', self.cfg.confidence)
            self.cfg.set('confidence', conf)
            print()

        loading(title)
        t0 = time.time()
        results = self.cfg.model.predict(
            source=source, save=self.cfg.save_results, conf=conf,
            iou=self.cfg.iou_threshold, imgsz=self.cfg.image_size,
            max_det=self.cfg.max_detections, show_labels=self.cfg.show_labels,
            show_conf=self.cfg.show_conf, line_width=self.cfg.line_width,
            stream=stream, show=show,
        )

        if stream:
            frames, dets = 0, 0
            for r in results:
                frames += 1
                n = len(r.boxes); dets += n
                if frames % 30 == 0:
                    info(f'帧 {frames}: 检测 {n} 个目标, 累计 {dets} 个')
            elapsed = time.time() - t0
            print()
            ok(f'{mode}完成！')
            info(f'处理帧数: {frames} | 总检测数: {dets} | 耗时: {elapsed:.2f}s | '
                 f'FPS: {frames / elapsed:.1f}')
            self._add_history(mode, str(source), elapsed, dets)
        else:
            elapsed = time.time() - t0
            print()
            self._display_results(results, elapsed)
            dt = sum(len(r.boxes) for r in results) if isinstance(results, list) else 0
            self._add_history(mode, str(source), elapsed, dt)

    # ── 检测功能 ──────────────────────────────────────────────────────────

    def detect_image(self):
        _header('单张图片检测')
        print(f'  {C.BOLD}输入源:{C.RESET}')
        print(f'  {C.CYAN}1{C.RESET}. 本地图片  {C.DIM}2{C.RESET}. 网络URL  {C.DIM}3{C.RESET}. 示例图片\n')
        c = _ask('请选择', '3', ['1', '2', '3'])
        if c == '1':
            src = _ask('请输入图片路径')
            if not os.path.exists(src): err(f'文件不存在: {src}'); return
        elif c == '2':
            src = _ask('请输入图片 URL')
        else:
            src = 'https://ultralytics.com/images/bus.jpg'; info(f'使用示例图片: {src}')
        self._run_detect('单张图片检测', src, '正在进行目标检测...')

    def detect_batch(self):
        _header('批量图片检测')
        folder = _ask('请输入图片文件夹路径')
        if not os.path.isdir(folder): err(f'文件夹不存在: {folder}'); return
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if Path(f).suffix.lower() in exts]
        if not files: err('未找到图片文件'); return
        info(f'找到 {len(files)} 张图片')
        self._run_detect('批量图片检测', files, '正在进行批量检测...')
        # 补充显示每张图片结果
        results = self.cfg.model.predict(
            source=files, save=self.cfg.save_results, conf=self.cfg.confidence,
            iou=self.cfg.iou_threshold, imgsz=self.cfg.image_size,
            max_det=self.cfg.max_detections, show_labels=self.cfg.show_labels,
            show_conf=self.cfg.show_conf, line_width=self.cfg.line_width, show=False,
        )
        t0 = time.time()
        elapsed = time.time() - t0
        self._display_batch_results(results, elapsed, len(files))
        self._add_history('批量图片检测', folder, elapsed,
                          sum(len(r.boxes) for r in results))

    def detect_video(self):
        _header('视频文件检测')
        src = _ask('请输入视频文件路径')
        if not os.path.exists(src): err(f'文件不存在: {src}'); return
        show = confirm('是否显示实时预览？')
        self._run_detect('视频文件检测', src, '正在处理视频...', stream=True, show=show)

    def detect_camera(self):
        _header('实时摄像头检测')
        idx = _ask_int('摄像头索引 (0=默认)', 0, 0, 10)
        print(); info("按 'q' 键退出实时检测"); info('正在打开摄像头...')
        self._run_detect('摄像头检测', idx, '', stream=True, show=True)

    def detect_url(self):
        _header('网络图片/视频检测')
        print(f'  {C.BOLD}支持:{C.RESET} 图片URL、视频URL、YouTube链接\n')
        src = _ask('请输入 URL')
        self._run_detect('网络检测', src, '正在下载并处理...')

    # ── 结果展示 ──────────────────────────────────────────────────────────

    def _display_results(self, results, elapsed: float):
        if not results: warn('没有检测结果'); return
        r = results[0]; n = len(r.boxes)
        print(f'  {C.BOLD}检测结果:{C.RESET}\n')
        if n == 0:
            info('未检测到任何目标')
        else:
            rows = []
            for i, box in enumerate(r.boxes, 1):
                cid = int(box.cls[0]); cf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append([str(i), r.names[cid], f'{cf:.2%}', f'({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})'])
            print_table(['序号', '类别', '置信度', '位置 (x1,y1,x2,y2)'], rows, f'检测到 {n} 个目标')
        print(f'  {C.BOLD}性能:{C.RESET}')
        print(f'  {C.CYAN}•{C.RESET} 耗时: {elapsed:.3f}s | 速度: {1/elapsed:.1f}FPS | 检测: {n}个')
        if self.cfg.save_results: print(); info("结果已保存到 'runs/detect/predict'")

    def _display_batch_results(self, results, elapsed: float, total: int):
        if not results: return
        td = sum(len(r.boxes) for r in results)
        has = sum(1 for r in results if len(r.boxes) > 0)
        rows = [['总图片数', str(total)], ['有检测结果的', str(has)],
                ['总检测目标数', str(td)], ['平均/张', f'{td/total:.1f}'],
                ['耗时', f'{elapsed:.2f}s'], ['速度', f'{total/elapsed:.1f}张/s']]
        print_table(['统计项', '数值'], rows, '批量检测统计')
        for i, r in enumerate(results, 1):
            n = len(r.boxes)
            if n > 0: print(f'  {C.CYAN}图片{i}{C.RESET}: 检测到 {n} 个目标')
            else:     print(f'  {C.DIM}图片{i}: 未检测到目标{C.RESET}')
        if self.cfg.save_results: print(); info("结果已保存到 'runs/detect/predict'")

    # ── 历史 / 帮助 ──────────────────────────────────────────────────────

    def _add_history(self, t: str, src: str, elapsed: float, dets: int = 0):
        self.history.append({
            'type': t, 'source': src, 'detections': dets,
            'elapsed_time': elapsed, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    def show_history(self):
        _header('检测历史')
        if not self.history: info('暂无检测历史记录'); return
        rows = []
        for i, r in enumerate(self.history, 1):
            src = r['source']; src = src[:40] + '...' if len(src) > 40 else src
            rows.append([str(i), r['type'], src, str(r['detections']),
                         f"{r['elapsed_time']:.2f}s", r['timestamp']])
        print_table(['序号', '类型', '来源', '检测数', '耗时', '时间'], rows, f'共 {len(self.history)} 条')

    def show_help(self):
        _header('帮助说明')
        print(f"""
  {C.BOLD}YOLOv8 交互式目标检测应用 v2.0{C.RESET}

  {C.CYAN}模型管理:{C.RESET}
  • 应用自动扫描 ./models 目录下的 .pt 模型文件
  • {C.GREEN}[✓ 已存在]{C.RESET} — 已在本地，可直接加载使用
  • {C.YELLOW}[需下载]{C.RESET}   — 需从网络下载，首次使用需等待
  • 选择未下载的模型会自动下载并保存到 ./models 目录

  {C.CYAN}参数说明:{C.RESET}
  • 置信度阈值: 只显示高于此值的检测结果 (推荐 0.25-0.5)
  • IoU 阈值: NMS 去重阈值 (推荐 0.5-0.7)
  • 图像尺寸: 越大精度越高但速度越慢 (推荐 640)

  {C.CYAN}快捷操作:{C.RESET}
  • 直接回车使用默认值  • 输入 '0' 返回上级  • Ctrl+C 中断操作
""")

    # ── 退出 ──────────────────────────────────────────────────────────────

    def _exit(self):
        print(f'\n{C.CYAN}{"=" * 70}{C.RESET}')
        print(f'\n  {C.CYAN}{C.BOLD}感谢使用 YOLOv8 交互式目标检测应用！{C.RESET}\n')
        print(f'  {C.DIM}如有问题或建议，欢迎反馈{C.RESET}\n')
        print(f'{C.CYAN}{"=" * 70}{C.RESET}\n')
        self.running = False


# ============================================================================

if __name__ == '__main__':
    App().run()