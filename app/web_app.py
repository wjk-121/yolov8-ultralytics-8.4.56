# -*- coding: utf-8 -*-
'''YOLOv8 Web Application v5'''
import os
# ⚠ 必须在导入 cv2 / ultralytics 之前设置, 否则 FFMPEG 后端初始化后无法抑制 OpenH264 警告
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

import shutil, time, uuid, base64, json, traceback, threading, logging, sys
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from waitress import serve
from ultralytics import YOLO
import cv2
from models_db import MODELS, CATS

cv2.setLogLevel(0)  # 0=SILENT, 仅显示致命错误

# ── 日志 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
log = logging.getLogger('web_app')

def _create_video_writer(path, fps, size, try_h264=True):
    """创建 VideoWriter, 静默处理 H264 不可用的情况 (抑制 OpenH264 噪音)"""
    import sys
    # 保存原始 stderr fd
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, stderr_fd)
    os.close(null)
    try:
        if try_h264:
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'H264'), fps, size)
            if w.isOpened():
                os.dup2(saved, stderr_fd)
                os.close(saved)
                return w, True
            w.release()
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        os.dup2(saved, stderr_fd)
        os.close(saved)
        return w, False
    except Exception:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        raise

# 项目根
BASE_DIR = Path(__file__).parent.parent.resolve()

app = Flask(__name__, static_folder=str(BASE_DIR / 'static'), static_url_path='')
CORS(app)

# ── 目录 ────────────────────────────────────────────────────────────────
RUNS_DIR = BASE_DIR / 'runs' / 'detect'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR  = BASE_DIR / 'temp'
TMP_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLES_DIR = BASE_DIR / 'data' / 'samples'
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES = {
    'bus':    {'url': 'https://ultralytics.com/images/bus.jpg',    'file': 'bus.jpg'},
    'zidane': {'url': 'https://ultralytics.com/images/zidane.jpg', 'file': 'zidane.jpg'},
}

# ── 模型注册表 — 从 models_db 导入 (见文件顶部) ───────────────────────

# ── 配置 ──────────────────────────────────────────────────────────────────
class Config:
    DEFAULTS = {'model':'yolov8n','confidence':0.25,'iou_threshold':0.7}
    def __init__(self):
        self._cfg = self.DEFAULTS.copy()
        self.model = None
        self._lc = {}
    @property
    def models_dir(self): return BASE_DIR / 'models'
    def get(self,k): return self._cfg.get(k)
    def set(self,k,v): self._cfg[k]=v
    def model_name(self): return self._cfg['model']
    def model_info(self,name=''):
        n = name or self._cfg['model']
        if n in MODELS: return MODELS[n]
        if self.is_local(n): return {'name': n, 'desc': '自定义模型'}
        return {}
    def scan_local(self):
        self._lc.clear()
        self.models_dir.mkdir(parents=True,exist_ok=True)
        for pt in self.models_dir.glob('*.pt'):
            sz = pt.stat().st_size/(1024*1024)
            self._lc[pt.stem] = {'path':str(pt),'size_mb':round(sz,1),'modified':datetime.fromtimestamp(pt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
        return self._lc
    def is_local(self,name): return name in self._lc or (self.models_dir/f'{name}.pt').exists()
    def model_path(self,name):
        if name in self._lc: return self._lc[name]['path']
        p = self.models_dir / f'{name}.pt'; return str(p) if p.exists() else f'{name}.pt'
    def to_dict(self): return {'model':self._cfg['model'],'confidence':self._cfg['confidence'],'iou_threshold':self._cfg['iou_threshold']}

config = Config()

def load_model(name=None):
    name = name or config.model_name(); config.models_dir.mkdir(parents=True,exist_ok=True)
    try:
        if config.is_local(name): config.model = YOLO(config.model_path(name))
        else:
            config.model = YOLO(f'{name}.pt')
            dl = Path(f'{name}.pt')
            tgt = config.models_dir / f'{name}.pt'
            if dl.exists(): shutil.move(str(dl),str(tgt)); config.scan_local()
        config.set('model',name); return True
    except: traceback.print_exc(); return False

def load_model_safe(name):
    """安全加载模型: 本地模型同步加载, 远程模型抛出 RuntimeError 提示异步下载"""
    config.models_dir.mkdir(parents=True,exist_ok=True)
    if config.is_local(name):
        return load_model(name)
    else:
        raise RuntimeError(f'模型 {name} 未下载, 请通过 /api/models/select 接口触发异步下载')

# ── 模型下载任务追踪 ──────────────────────────────────────────────────
_download_jobs = {}  # name -> {status, progress, error, started, name}

def _download_worker(name):
    """后台下载模型线程"""
    try:
        config.models_dir.mkdir(parents=True,exist_ok=True)
        _download_jobs[name] = {'status':'downloading','progress':0,'error':None,'started':time.time(),'name':name}
        # YOLO() 构造函数会自动触发 ultralytics 下载逻辑
        m = YOLO(f'{name}.pt')
        dl = Path(f'{name}.pt')
        tgt = config.models_dir / f'{name}.pt'
        if dl.exists():
            shutil.move(str(dl),str(tgt))
            config.scan_local()
        config.model = m
        config.set('model',name)
        _download_jobs[name] = {'status':'complete','progress':100,'error':None,'name':name}
        log.info(f'✓ 模型下载完成: {name} → models/{name}.pt')
    except Exception as e:
        _download_jobs[name] = {'status':'error','progress':0,'error':str(e)[:500],'name':name}
        log.error(f'✗ 模型下载失败: {name} - {e}')
        traceback.print_exc()

def _ensure_dirs():
    """确保所有必需目录存在"""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── 运行目录管理 ──────────────────────────────────────────────────────────
def _scan_runs():
    runs = []
    if not RUNS_DIR.exists(): return runs
    for d in sorted(RUNS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir(): continue
        mf = d / 'metadata.json'
        if mf.exists():
            try:
                with open(mf,'r',encoding='utf-8') as f: runs.append(json.load(f))
            except: pass
    return runs

def _save_run_meta(run_dir: Path, meta: dict):
    with open(run_dir / 'metadata.json','w',encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _delete_run_dir(run_name: str):
    d = RUNS_DIR / run_name
    if d.exists(): shutil.rmtree(str(d), ignore_errors=True)

def _extract_detections(result):
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cid = int(box.cls[0]); cf = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            dets.append({'class_id':cid,'class_name':result.names[cid],'confidence':round(cf,4),'bbox':[round(x1,1),round(y1,1),round(x2,1),round(y2,1)]})
    return dets

# ── 摄像头会话管理 ────────────────────────────────────────────────────────
_camera_sessions = {}  # session_id -> {run_dir, snapshots: [...], created, last_active}
CAMERA_SESSION_TIMEOUT = 1800  # 30分钟超时

def _get_or_create_camera_session(session_id=None):
    now = time.time()
    # 清理过期会话
    expired = [sid for sid, s in _camera_sessions.items() if now - s['last_active'] > CAMERA_SESSION_TIMEOUT]
    for sid in expired:
        _finalize_camera_session(sid)

    if session_id and session_id in _camera_sessions:
        session = _camera_sessions[session_id]
        session['last_active'] = now
        return session_id, session

    # 创建新会话
    sid = f'cam_{uuid.uuid4().hex[:8]}'
    run_dir = RUNS_DIR / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    session = {
        'run_dir': run_dir,
        'session_id': sid,
        'snapshots': [],
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_active': now,
        'total_detections': 0
    }
    _camera_sessions[sid] = session
    return sid, session

def _finalize_camera_session(session_id):
    if session_id not in _camera_sessions: return
    session = _camera_sessions.pop(session_id)
    run_dir = session['run_dir']
    if not session['snapshots']: return

    # 保存会话元数据
    result_files = sorted(run_dir.glob('*'))
    _save_run_meta(run_dir, {
        'id': session['session_id'],
        'type': '摄像头检测',
        'source': f'{len(session["snapshots"])} 张抓拍',
        'detections': session['total_detections'],
        'elapsed_time': 0,
        'timestamp': session['created'],
        'client_id': session.get('client_id', ''),
        'result_image': session['snapshots'][-1]['result_image'] if session['snapshots'] else '',
        'result_data': {
            'snapshot_count': len(session['snapshots']),
            'snapshots': session['snapshots'],
            'session_start': session['created'],
            'session_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'files': [f.name for f in result_files]
    })

# ── 路由 ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(BASE_DIR / 'static'), 'index.html')

@app.route('/api/status')
def api_status():
    local = config.scan_local()
    return jsonify({'success':True,'data':{'current_model':config.model_name(),'model_info':config.model_info(),'is_model_loaded':config.model is not None,'is_local':config.is_local(config.model_name()),'local_models':local,'config':config.to_dict()}})

@app.route('/api/models')
def api_models():
    local = config.scan_local(); ml = []
    for cat,names in CATS.items():
        for name in names:
            mi = MODELS.get(name,{}); il = name in local
            ml.append({'name':name,'full_name':mi.get('name','?'),'desc':mi.get('desc',''),'params':mi.get('params','N/A'),'num_classes':mi.get('num_classes','N/A'),'category':mi.get('category',''),'is_local':il,'size_mb':local.get(name,{}).get('size_mb',0),'is_current':name==config.model_name()})
    # 自定义模型 (models/ 文件夹中不在官方注册表里的 .pt 文件)
    for name, info in local.items():
        if name not in MODELS:
            ml.append({'name':name,'full_name':name,'desc':'自定义模型','params':'N/A','num_classes':'N/A','category':'自定义模型','is_local':True,'size_mb':info.get('size_mb',0),'is_current':name==config.model_name()})
    lc = sum(1 for m in ml if m['is_local'])
    return jsonify({'success':True,'data':{'models':ml,'stats':{'total':len(ml),'local':lc,'remote':len(ml)-lc}}})

@app.route('/api/models/select',methods=['POST'])
def api_select_model():
    data = request.get_json(force=True,silent=True) or {}
    name = data.get('name')
    if not name: return jsonify({'success':False,'error':'无效的模型名称'}),400
    if name not in MODELS and not config.is_local(name): return jsonify({'success':False,'error':'无效的模型名称'}),400

    if config.is_local(name):
        # 本地模型 — 同步加载 (毫秒级)
        if load_model(name):
            mi = config.model_info(name)
            log.info(f'⟳ 模型切换: {mi.get("name", name)} (本地)')
            return jsonify({'success':True,'message':f'模型切换成功: {mi.get("name", name)}','data':{'model':name,'info':mi,'is_local':True,'status':'ready'}})
        return jsonify({'success':False,'error':'模型加载失败，请检查文件完整性'}),500

    # 远程模型 — 异步下载, 不阻塞请求
    if name in _download_jobs and _download_jobs[name]['status'] == 'downloading':
        return jsonify({'success':True,'message':f'{MODELS[name]["name"]} 正在下载中...','data':{'status':'downloading','name':name}})

    # 启动后台下载
    model_label = MODELS[name]['name']
    log.info(f'↓ 开始下载模型: {model_label} ({MODELS[name]["params"]})')
    threading.Thread(target=_download_worker, args=(name,), daemon=True).start()
    return jsonify({'success':True,'message':f'开始下载 {model_label} ({MODELS[name]["params"]}), 请稍候...','data':{'status':'downloading','name':name,'model_name':model_label,'params':MODELS[name]['params']}})

@app.route('/api/models/download-status/<name>')
def api_model_download_status(name):
    """轮询模型下载进度"""
    if name in _download_jobs:
        job = _download_jobs[name]
        return jsonify({'success':True,'data':job})
    # 检查是否已下载完成 (可能在之前的会话中)
    if config.is_local(name):
        return jsonify({'success':True,'data':{'status':'complete','progress':100,'error':None,'name':name}})
    return jsonify({'success':False,'error':'无下载记录'}),404

@app.route('/api/models/<name>', methods=['DELETE'])
def api_delete_model(name):
    """删除本地模型文件"""
    if name not in MODELS and not (config.models_dir / f'{name}.pt').exists():
        return jsonify({'success': False, 'error': '无效的模型名称'}), 400
    if name == config.model_name():
        return jsonify({'success': False, 'error': '无法删除正在使用的模型，请先切换到其他模型'}), 409
    model_path = config.models_dir / f'{name}.pt'
    if not model_path.exists():
        return jsonify({'success': False, 'error': '模型文件不存在'}), 404
    try:
        sz_mb = round(model_path.stat().st_size / (1024 * 1024), 1)
        model_path.unlink()
        config.scan_local()
        log.info(f'🗑 删除模型: {name} ({sz_mb}MB)')
        return jsonify({'success': True, 'message': f'模型 {name} 已删除 (释放 {sz_mb}MB)'})
    except Exception as e:
        log.error(f'✗ 删除模型失败: {name} - {e}')
        return jsonify({'success': False, 'error': f'删除失败: {str(e)}'}), 500

@app.route('/api/config',methods=['GET'])
def api_get_config():
    return jsonify({'success':True,'data':config.to_dict()})

@app.route('/api/config',methods=['POST'])
def api_set_config():
    data = request.get_json(force=True,silent=True) or {}
    for key in ['confidence','iou_threshold']:
        if key in data: config.set(key,data[key])
    log.info(f'⚙ 配置更新: conf={config.get("confidence")}')
    return jsonify({'success':True,'message':'配置已更新','data':config.to_dict()})

# ── 图片检测 ─────────────────────────────────────────────────────────────
@app.route('/api/detect/image',methods=['POST'])
def api_detect_image():
    _ensure_dirs()
    try:
        if config.model is None: return jsonify({'success':False,'error':'模型未加载'}),500
        file = request.files.get('file')
        conf_val = request.form.get('confidence', config.get('confidence'))
        client_id = (request.form.get('client_id') or '').strip()
        if not file or not file.filename: return jsonify({'success':False,'error':'请上传图片'}),400

        # ── 文件类型验证 ──
        ok, err = _validate_file(file, _IMG_EXTS, _IMG_MIMES)
        if not ok:
            log.warning(f'✗ 图片上传被拒绝 | {file.filename} → {err}')
            return jsonify({'success': False, 'error': err}), 400
        log.info(f'→ 图片检测 | {file.filename} | conf={conf_val} | client={client_id[:8] if client_id else "-"}')
        tmp_path = TMP_DIR / f'img_{uuid.uuid4().hex}{Path(file.filename).suffix}'
        file.save(str(tmp_path))

        run_name = f'img_{uuid.uuid4().hex[:8]}'
        run_dir = RUNS_DIR / run_name
        t0 = time.time()
        results = config.model.predict(
            source=str(tmp_path),
            save=True, project=str(RUNS_DIR), name=run_name,
            exist_ok=True, verbose=False,
            conf=float(conf_val), iou=config.get('iou_threshold')
        )
        elapsed = time.time() - t0
        tmp_path.unlink(missing_ok=True)

        r = results[0]; n = len(r.boxes) if r.boxes is not None else 0
        dets = _extract_detections(r)

        result_files = sorted(run_dir.glob('*'))
        result_image = ''
        for rf in result_files:
            if rf.suffix.lower() in ('.jpg','.jpeg','.png'):
                result_image = f'/api/runs/{run_name}/{rf.name}'
                break

        _save_run_meta(run_dir, {
            'id':run_name, 'type':'图片检测', 'source':file.filename,
            'detections':n, 'elapsed_time':round(elapsed,3),
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'client_id':client_id,
            'result_image':result_image,
            'result_data':{'detections':dets,'image_size':{'height':r.orig_shape[0],'width':r.orig_shape[1]} if hasattr(r,'orig_shape') else None},
            'files':[rf.name for rf in result_files]
        })

        log.info(f'✓ 图片检测完成 | {n} 目标 | {elapsed:.2f}s | {run_name}')
        return jsonify({'success':True,'message':f'检测完成，发现 {n} 个目标',
            'data':{'result_image':result_image,'total_detections':n,
                    'elapsed_time':round(elapsed,3),'fps':round(1/elapsed,1) if elapsed>0 else 0,
                    'detections':dets,'run_name':run_name}})
    except Exception as e:
        log.error(f'✗ 图片检测失败: {e}')
        traceback.print_exc()
        return jsonify({'success':False,'error':f'检测失败: {str(e)}'}),500

# ── 批量检测 (异步轮询, 支持图片+视频混搭, 实时进度) ──────────────────
_IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'}
_VID_EXTS = {'.mp4','.avi','.mov','.mkv','.webm','.wmv','.flv'}

# MIME 类型映射 (二次校验, 浏览器可能发送不准确的 Content-Type, 仅作辅助)
_IMG_MIMES = {'image/jpeg','image/png','image/bmp','image/webp','image/tiff'}
_VID_MIMES = {'video/mp4','video/x-msvideo','video/quicktime','video/x-matroska','video/webm','video/x-ms-wmv','video/x-flv'}

# 大文件参考阈值 (不拒绝上传, 仅用于前端/日志提示)
LARGE_IMAGE_MB = 20   # 图片超过此值给出提示
LARGE_VIDEO_MB = 100  # 视频超过此值给出提示

def _validate_file(file, allowed_exts, allowed_mimes):
    """验证上传文件: 扩展名 + MIME + 空文件检测 + 双扩展名防护
    注意: 不限制文件大小 — 超大文件仅记录日志, 不拒绝上传
    Returns: (is_valid, error_message)
    """
    # 1. 检查文件名
    if not file or not file.filename:
        return False, '未选择文件'

    fname = file.filename

    # 2. 检查文件是否为空
    try:
        file.seek(0, 2)
        fsize = file.tell()
        file.seek(0)
        if fsize == 0:
            return False, f'文件 "{fname}" 为空, 请重新选择'
    except Exception:
        fsize = None  # 流式上传无法检测大小
        pass

    # 3. 扩展名校验
    ext = Path(fname).suffix.lower()
    if not ext:
        return False, f'文件 "{fname}" 没有扩展名, 无法识别文件类型'
    if ext not in allowed_exts:
        allowed_str = ', '.join(sorted(allowed_exts))
        return False, f'不支持的文件类型 "{ext}" (允许: {allowed_str})'

    # 4. 大文件提示 (不拒绝, 仅记录日志)
    if fsize:
        size_mb = fsize / (1024 * 1024)
        is_img = ext in _IMG_EXTS
        threshold = LARGE_IMAGE_MB if is_img else LARGE_VIDEO_MB
        if size_mb > threshold:
            log.info(f'⚠ 大文件上传: {fname} ({size_mb:.1f}MB), 检测可能需要较长时间')

    # 5. MIME 类型辅助校验 — 仅当浏览器提供了 Content-Type 且明显不匹配时警告
    if allowed_mimes and file.content_type:
        ct = file.content_type.split(';')[0].strip().lower()
        if ct and ct not in allowed_mimes and ct != 'application/octet-stream':
            # 不直接拒绝 (浏览器 MIME 可能不准), 但可在此记录日志
            log.warning(f'⚠ MIME 类型可疑: {fname} → {ct} (预期: {", ".join(sorted(allowed_mimes))})')

    # 6. 文件名安全校验 (防止双扩展名伪装攻击)
    parts = fname.rsplit('.', 2)
    if len(parts) >= 3:
        second_ext = '.' + parts[-2].lower()
        dangerous = {'.exe','.dll','.bat','.cmd','.ps1','.sh','.vbs','.js','.py','.php','.com','.msi','.scr'}
        if second_ext in dangerous:
            return False, f'文件名 "{fname}" 包含可疑的双扩展名, 已被拒绝'

    return True, None

# 扩展名友好名称映射
_EXT_FRIENDLY = {
    **{ext: '图片' for ext in _IMG_EXTS},
    **{ext: '视频' for ext in _VID_EXTS},
}
for ext in ('.jpg','.jpeg','.png','.bmp'): _EXT_FRIENDLY[ext] = '图片'
for ext in ('.mp4','.avi','.mov','.mkv'): _EXT_FRIENDLY[ext] = '视频'

# ── 任务队列 (串行处理 + 客户端隔离) ──────────────────────────────────────
_processing_lock = threading.Lock()
_job_queue = []  # 有序等待队列: [(job_id, type), ...]

def _update_all_queue_positions():
    """更新所有排队任务的 queue_position"""
    for i, (qid, qtype) in enumerate(_job_queue):
        d = _video_jobs if qtype == 'video' else _batch_jobs
        if qid in d:
            d[qid]['queue_position'] = i + 1

def _calculate_eta_for(job_id):
    """计算排队任务的预计等待时间 (秒)"""
    total = 0
    # 当前正在处理的任务剩余时间
    for jobs in (_video_jobs, _batch_jobs):
        for jid, job in jobs.items():
            if job.get('status') == 'processing':
                eta = job.get('eta', 0)
                if eta:
                    total += eta
                break
    # 累加队列中前方任务的粗略预估
    for qid, qtype in _job_queue:
        if qid == job_id:
            break
        if qtype == 'video':
            total += 120
        else:
            j = _batch_jobs.get(qid, {})
            n_vids = j.get('total_videos', 0)
            n_imgs = j.get('total_images', 0)
            total += n_vids * 120 + n_imgs * 5
    return max(round(total), 1)

def _run_queued_job(job_id, job_type, jobs_dict, process_fn):
    """任务执行包装器: 获取锁/排队 → 执行 → 释放锁"""
    got_lock = _processing_lock.acquire(blocking=False)
    if got_lock:
        jobs_dict[job_id]['status'] = 'processing'
        jobs_dict[job_id]['queue_position'] = 0
    else:
        jobs_dict[job_id]['status'] = 'queued'
        _job_queue.append((job_id, job_type))
        _update_all_queue_positions()
        _processing_lock.acquire()  # 阻塞等待
        # 从队列中移除自己
        for i, (qid, qtype) in enumerate(_job_queue):
            if qid == job_id:
                _job_queue.pop(i)
                break
        jobs_dict[job_id]['status'] = 'processing'
        jobs_dict[job_id]['queue_position'] = 0
        _update_all_queue_positions()
    try:
        process_fn()
    finally:
        _processing_lock.release()


_batch_jobs = {}  # job_id -> {status, progress, current_file_index, total_files, ...}

@app.route('/api/detect/batch',methods=['POST'])
def api_detect_batch():
    _ensure_dirs()
    try:
        if config.model is None: return jsonify({'success':False,'error':'模型未加载'}),500
        files = request.files.getlist('files')
        if not files: return jsonify({'success':False,'error':'请上传文件'}),400
        conf_val = float(request.form.get('confidence', config.get('confidence')))
        client_id = (request.form.get('client_id') or '').strip()

        # 分类并验证文件
        saved = []      # [(temp_path, original_filename, type)]
        rejected = []   # [{filename, reason}]
        for f in files:
            if not f.filename:
                rejected.append({'filename': '(空文件名)', 'reason': '文件名为空'})
                continue

            ext = Path(f.filename).suffix.lower()
            if ext in _IMG_EXTS:
                ok, err = _validate_file(f, _IMG_EXTS, _IMG_MIMES)
                if not ok:
                    rejected.append({'filename': f.filename, 'reason': err})
                    continue
                tp = TMP_DIR / f'batchimg_{uuid.uuid4().hex}_{f.filename}'
                f.save(str(tp))
                saved.append((tp, f.filename, 'image'))
            elif ext in _VID_EXTS:
                ok, err = _validate_file(f, _VID_EXTS, _VID_MIMES)
                if not ok:
                    rejected.append({'filename': f.filename, 'reason': err})
                    continue
                tp = TMP_DIR / f'batchvid_{uuid.uuid4().hex}_{f.filename}'
                f.save(str(tp))
                saved.append((tp, f.filename, 'video'))
            else:
                # 扩展名不在任何允许列表中
                rejected.append({
                    'filename': f.filename,
                    'reason': f'不支持的文件类型 "{ext}" (允许图片: {", ".join(sorted(_IMG_EXTS))}; 视频: {", ".join(sorted(_VID_EXTS))})'
                })

        if rejected:
            names = ', '.join(r['filename'] for r in rejected[:5])
            extra = f' ...等{len(rejected)}个' if len(rejected) > 5 else ''
            log.warning(f'✗ 批量上传部分被拒绝 | {len(rejected)}/{len(files)} 文件: {names}{extra}')

        if not saved:
            return jsonify({
                'success': False,
                'error': f'所有 {len(rejected)} 个文件均不符合要求',
                'data': {'rejected': rejected, 'total_rejected': len(rejected)}
            }), 400

        n_imgs = sum(1 for s in saved if s[2] == 'image')
        n_vids = sum(1 for s in saved if s[2] == 'video')
        log.info(f'→ 批量检测 | {len(saved)} 文件 ({n_imgs} 图片 + {n_vids} 视频) | conf={conf_val} | client={client_id[:8] if client_id else "-"}')

        job_id = f'batch_{uuid.uuid4().hex[:8]}'
        has_videos = any(s[2] == 'video' for s in saved)
        now = time.time()
        _batch_jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'current_file_index': 0,
            'total_files': len(saved),
            'current_file_name': saved[0][1],
            'current_file_type': saved[0][2],
            'current_file_progress': 0,
            'current_file_frame': 0,
            'current_file_total_frames': 0,
            'total_images': sum(1 for s in saved if s[2] == 'image'),
            'total_videos': sum(1 for s in saved if s[2] == 'video'),
            'total_detections': 0,
            'elapsed': 0,
            'eta': None,
            'results': [],
            'run_name': job_id,
            'has_videos': has_videos,
            'error': None,
            'start_time': now,
            'client_id': client_id,
            'queue_position': 0
        }

        def process_batch():
            try:
                run_dir = RUNS_DIR / job_id
                run_dir.mkdir(parents=True)
                t0 = time.time()
                total_dets = 0
                all_results = []

                for i, (tp, fname, ftype) in enumerate(saved):
                    # 更新当前文件信息
                    file_share = 100.0 / len(saved)
                    base_pct = i * file_share
                    _batch_jobs[job_id].update({
                        'current_file_index': i,
                        'current_file_name': fname,
                        'current_file_type': ftype,
                        'current_file_progress': 0,
                        'current_file_frame': 0,
                        'current_file_total_frames': 0,
                        'progress': round(base_pct, 1)
                    })

                    if ftype == 'image':
                        # ── 图片: 单张预测 ──────────────────────────
                        results = config.model.predict(
                            source=str(tp),
                            save=True, project=str(RUNS_DIR), name=job_id,
                            exist_ok=True, verbose=False,
                            conf=conf_val, iou=config.get('iou_threshold')
                        )
                        r = results[0]
                        n = len(r.boxes) if r.boxes is not None else 0
                        total_dets += n
                        result_image = f'/api/runs/{job_id}/{tp.stem}.jpg'
                        all_results.append({
                            'index': len(all_results) + 1,
                            'filename': fname, 'type': 'image',
                            'detections_count': n,
                            'detections': _extract_detections(r),
                            'result_image': result_image
                        })
                        tp.unlink(missing_ok=True)
                        # 图片完成, 进度跳到该文件份额末尾
                        file_end_pct = round((i + 1) * file_share, 1)
                        _batch_jobs[job_id].update({
                            'progress': file_end_pct,
                            'current_file_progress': 100,
                            'total_detections': total_dets,
                            'elapsed': round(time.time() - t0, 3)
                        })
                        log.info(f'  └─ [{i+1}/{len(saved)}] 图片 {fname} → {n} 目标 ({_batch_jobs[job_id]["elapsed"]:.1f}s)')

                    else:
                        # ── 视频: 逐帧处理 + 实时进度 ──────────────
                        cap = cv2.VideoCapture(str(tp))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        out_name = f'batchvid_{uuid.uuid4().hex[:8]}.mp4'
                        out_path = run_dir / out_name
                        out_writer, h264_opened = _create_video_writer(out_path, fps_vid, (w, h))

                        _batch_jobs[job_id].update({
                            'current_file_total_frames': total_frames,
                            'current_file_frame': 0,
                            'current_file_progress': 0
                        })

                        vid_dets = 0
                        frame_idx = 0
                        update_interval = max(1, total_frames // 100) if total_frames > 0 else 10

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_idx += 1
                            try:
                                r = config.model(frame, conf=conf_val, iou=config.get('iou_threshold'), verbose=False)[0]
                                n = len(r.boxes) if r.boxes is not None else 0
                                vid_dets += n
                                out_writer.write(r.plot())
                            except Exception:
                                pass

                            if frame_idx % update_interval == 0 or frame_idx == total_frames:
                                frame_pct = round(frame_idx / total_frames * 100, 1) if total_frames > 0 else 100
                                overall_pct = round(base_pct + (frame_idx / total_frames) * file_share, 1) if total_frames > 0 else round(base_pct + file_share, 1)
                                e = time.time() - t0
                                fps_overall = frame_idx / max(e, 0.001)
                                eta = (total_frames - frame_idx) / max(fps_overall, 0.001) if total_frames > 0 else 0
                                _batch_jobs[job_id].update({
                                    'progress': min(overall_pct, 100),
                                    'current_file_frame': frame_idx,
                                    'current_file_progress': frame_pct,
                                    'total_detections': total_dets + vid_dets,
                                    'elapsed': round(e, 3),
                                    'eta': round(eta, 1)
                                })

                        out_writer.release()
                        cap.release()
                        total_dets += vid_dets
                        playable = h264_opened
                        all_results.append({
                            'index': len(all_results) + 1,
                            'filename': fname, 'type': 'video',
                            'detections_count': vid_dets,
                            'total_frames': total_frames,
                            'result_video': f'/api/runs/{job_id}/{out_name}',
                            'playable': playable
                        })
                        tp.unlink(missing_ok=True)
                        # 视频完成
                        file_end_pct = round((i + 1) * file_share, 1)
                        _batch_jobs[job_id].update({
                            'progress': file_end_pct,
                            'current_file_progress': 100,
                            'total_detections': total_dets
                        })
                        log.info(f'  └─ [{i+1}/{len(saved)}] 视频 {fname} → {vid_dets} 目标, {total_frames} 帧')

                # ── 全部完成 ───────────────────────────────────────
                elapsed = time.time() - t0

                # 找代表图片
                result_image = ''
                for rf in sorted(run_dir.glob('*')):
                    if rf.suffix.lower() in _IMG_EXTS:
                        result_image = f'/api/runs/{job_id}/{rf.name}'
                        break
                if not result_image:
                    for rf in sorted(run_dir.glob('*')):
                        if rf.suffix.lower() == '.mp4':
                            result_image = f'/api/runs/{job_id}/{rf.name}'
                            break

                total_images = sum(1 for s in saved if s[2] == 'image')
                total_videos = sum(1 for s in saved if s[2] == 'video')
                source_desc = f'{total_images} 图片' if total_videos == 0 else f'{total_images} 图片 + {total_videos} 视频'
                _save_run_meta(run_dir, {
                    'id': job_id, 'type': '批量检测', 'source': source_desc,
                    'detections': total_dets, 'elapsed_time': round(elapsed, 3),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'client_id': client_id,
                    'result_image': result_image,
                    'result_data': {
                        'total_images': total_images, 'total_videos': total_videos,
                        'total_detections': total_dets, 'results': all_results
                    },
                    'files': [rf.name for rf in sorted(run_dir.glob('*'))]
                })

                _batch_jobs[job_id].update({
                    'status': 'complete',
                    'progress': 100,
                    'elapsed': round(elapsed, 3),
                    'eta': 0,
                    'total_detections': total_dets,
                    'results': all_results,
                    'result_image': result_image
                })

                log.info(f'✓ 批量检测完成 | {len(saved)} 文件 ({total_images} 图片 + {total_videos} 视频) | {total_dets} 目标 | {elapsed:.2f}s')
                log.info(f'  Results saved to {run_dir}')

            except Exception as e:
                _batch_jobs[job_id]['status'] = 'error'
                _batch_jobs[job_id]['error'] = str(e)
                log.error(f'✗ 批量检测失败: {e}')
                traceback.print_exc()

        threading.Thread(target=lambda: _run_queued_job(job_id, 'batch', _batch_jobs, process_batch), daemon=True).start()
        is_queued = _batch_jobs[job_id].get('status') == 'queued'
        resp_data = {
            'job_id': job_id,
            'total_files': len(saved),
            'total_images': sum(1 for s in saved if s[2] == 'image'),
            'total_videos': sum(1 for s in saved if s[2] == 'video'),
            'has_videos': has_videos,
            'status': _batch_jobs[job_id]['status']
        }
        if is_queued:
            resp_data['queue_position'] = _batch_jobs[job_id].get('queue_position', 0)
            resp_data['estimated_wait'] = _calculate_eta_for(job_id)
        if rejected:
            resp_data['rejected'] = rejected
            resp_data['total_rejected'] = len(rejected)
        return jsonify({
            'success': True,
            'message': f'批量检测已启动 ({len(saved)} 个文件)' + (' (排队中)' if is_queued else '') + (f', {len(rejected)} 个被拒绝' if rejected else ''),
            'data': resp_data
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'批量检测失败: {str(e)}'}), 500


@app.route('/api/detect/batch/status/<job_id>')
def api_detect_batch_status(job_id):
    job = _batch_jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': '任务不存在'}), 404
    data = dict(job)
    if data.get('status') == 'queued':
        data['estimated_wait'] = _calculate_eta_for(job_id)
    return jsonify({'success': True, 'data': data})

# ── URL 检测 ─────────────────────────────────────────────────────────────
@app.route('/api/detect/url',methods=['POST'])
def api_detect_url():
    _ensure_dirs()
    try:
        if config.model is None: return jsonify({'success':False,'error':'模型未加载'}),500
        data = request.get_json(force=True,silent=True) or {}
        url = data.get('url')
        if not url: return jsonify({'success':False,'error':'请提供URL'}),400
        conf_val = data.get('confidence', config.get('confidence'))
        client_id = (data.get('client_id') or '').strip()
        log.info(f'→ URL检测 | {url[:120]} | conf={conf_val}')

        run_name = f'url_{uuid.uuid4().hex[:8]}'
        run_dir = RUNS_DIR / run_name
        t0 = time.time()
        results = config.model.predict(
            source=url,
            save=True, project=str(RUNS_DIR), name=run_name,
            exist_ok=True, verbose=False,
            conf=float(conf_val), iou=config.get('iou_threshold')
        )
        elapsed = time.time() - t0

        r = results[0]; n = len(r.boxes) if r.boxes is not None else 0
        dets = _extract_detections(r)

        result_files = sorted(run_dir.glob('*'))
        result_image = ''
        for rf in result_files:
            if rf.suffix.lower() in ('.jpg','.jpeg','.png'):
                result_image = f'/api/runs/{run_name}/{rf.name}'
                break

        _save_run_meta(run_dir, {
            'id':run_name, 'type':'URL检测', 'source':url[:200],
            'detections':n, 'elapsed_time':round(elapsed,3),
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'client_id':client_id,
            'result_image':result_image, 'result_data':{'detections':dets},
            'files':[rf.name for rf in result_files]
        })

        log.info(f'✓ URL检测完成 | {n} 目标 | {elapsed:.2f}s | {run_name}')
        return jsonify({'success':True,'message':f'检测完成，发现 {n} 个目标',
            'data':{'result_image':result_image,'total_detections':n,
                    'elapsed_time':round(elapsed,3),'detections':dets,'run_name':run_name}})
    except Exception as e:
        log.error(f'✗ URL检测失败: {e}')
        traceback.print_exc()
        return jsonify({'success':False,'error':f'URL检测失败: {str(e)}'}),500

# ── 示例图片检测 (本地优先, 按需下载到 data/samples/) ──────────────────
@app.route('/api/samples/check/<name>')
def api_samples_check(name):
    """快速检查示例图片是否已在本地"""
    if name not in SAMPLES:
        return jsonify({'success': False, 'error': '未知示例'}), 400
    sample_path = SAMPLES_DIR / SAMPLES[name]['file']
    return jsonify({'success': True, 'data': {'exists': sample_path.exists(), 'name': name}})

@app.route('/api/detect/sample', methods=['POST'])
def api_detect_sample():
    _ensure_dirs()
    try:
        if config.model is None:
            return jsonify({'success': False, 'error': '模型未加载'}), 500
        data = request.get_json(force=True, silent=True) or {}
        name = data.get('name', 'bus')
        if name not in SAMPLES:
            return jsonify({'success': False, 'error': f'未知示例: {name}'}), 400
        conf_val = float(data.get('confidence', config.get('confidence')))
        client_id = (data.get('client_id') or '').strip()

        sample = SAMPLES[name]
        sample_path = SAMPLES_DIR / sample['file']

        # 如果本地没有, 下载到 data/samples/
        if not sample_path.exists():
            log.info(f'↓ 下载示例图片: {sample["url"]} → {sample_path}')
            import requests as _req
            try:
                resp = _req.get(sample['url'], timeout=30)
                resp.raise_for_status()
                with open(sample_path, 'wb') as f:
                    f.write(resp.content)
                log.info(f'✓ 示例图片已保存: {sample_path}')
            except Exception as dl_err:
                log.error(f'✗ 示例图片下载失败: {dl_err}')
                return jsonify({'success': False, 'error': f'示例图片下载失败: {dl_err}'}), 502

        log.info(f'→ 示例检测 | {name} | conf={conf_val}')
        run_name = f'sample_{name}_{uuid.uuid4().hex[:6]}'
        run_dir = RUNS_DIR / run_name
        t0 = time.time()
        results = config.model.predict(
            source=str(sample_path),
            save=True, project=str(RUNS_DIR), name=run_name,
            exist_ok=True, verbose=False,
            conf=conf_val, iou=config.get('iou_threshold')
        )
        elapsed = time.time() - t0

        r = results[0]
        n = len(r.boxes) if r.boxes is not None else 0
        dets = _extract_detections(r)

        result_files = sorted(run_dir.glob('*'))
        result_image = ''
        for rf in result_files:
            if rf.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                result_image = f'/api/runs/{run_name}/{rf.name}'
                break

        _save_run_meta(run_dir, {
            'id': run_name, 'type': '示例检测', 'source': f'{name} ({sample["url"]})',
            'detections': n, 'elapsed_time': round(elapsed, 3),
            'client_id': client_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result_image': result_image,
            'result_data': {'detections': dets},
            'files': [rf.name for rf in result_files]
        })

        log.info(f'✓ 示例检测完成 | {name} | {n} 目标 | {elapsed:.2f}s')
        return jsonify({
            'success': True, 'message': f'示例检测完成，发现 {n} 个目标',
            'data': {
                'result_image': result_image, 'total_detections': n,
                'elapsed_time': round(elapsed, 3),
                'fps': round(1 / elapsed, 1) if elapsed > 0 else 0,
                'detections': dets, 'run_name': run_name, 'sample_name': name
            }
        })
    except Exception as e:
        log.error(f'✗ 示例检测失败: {e}')
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'示例检测失败: {str(e)}'}), 500

# ── 视频检测 (轮询模式) ─────────────────────────────────────────────────
_video_jobs = {}  # job_id -> {status, progress, frame, total_frames, ...}

@app.route('/api/detect/video',methods=['POST'])
def api_detect_video():
    _ensure_dirs()
    try:
        if 'file' not in request.files: return jsonify({'success':False,'error':'请上传视频文件'}),400
        file = request.files['file']
        if not file.filename: return jsonify({'success':False,'error':'未选择文件'}),400

        # ── 文件类型验证 ──
        ok, err = _validate_file(file, _VID_EXTS, _VID_MIMES)
        if not ok:
            log.warning(f'✗ 视频上传被拒绝 | {file.filename} → {err}')
            return jsonify({'success': False, 'error': err}), 400
        conf_val = float(request.form.get('confidence', config.get('confidence')))
        client_id = (request.form.get('client_id') or '').strip()
        log.info(f'→ 视频检测 | {file.filename} | conf={conf_val} | client={client_id[:8] if client_id else "-"}')
        tmp_path = TMP_DIR / f'vid_{uuid.uuid4().hex}{Path(file.filename).suffix}'
        file.save(str(tmp_path))

        job_id = f'vid_{uuid.uuid4().hex[:8]}'
        now = time.time()
        _video_jobs[job_id] = {
            'status':'processing', 'progress':0, 'frame':0, 'total_frames':0,
            'total_detections':0, 'elapsed':0, 'eta':None, 'error':None,
            'result_video':None, 'run_name':job_id, 'filename':file.filename,
            'start_time':now, 'client_id':client_id, 'queue_position':0
        }

        def process_video():
            try:
                cap = cv2.VideoCapture(str(tmp_path))
                if not cap.isOpened():
                    _video_jobs[job_id]['status'] = 'error'
                    _video_jobs[job_id]['error'] = '无法打开视频文件'
                    return
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video = cap.get(cv2.CAP_PROP_FPS) or 25; fps_video = max(1,fps_video)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _video_jobs[job_id]['total_frames'] = total_frames

                run_dir = RUNS_DIR / job_id; run_dir.mkdir(parents=True)
                out_name = f'annotated_{job_id}.mp4'
                out_path = run_dir / out_name
                out_writer, h264_opened = _create_video_writer(out_path, fps_video, (w, h))

                frame_idx = 0; t0 = time.time(); total_dets = 0
                # 每隔 N 帧更新一次状态 (减少锁竞争)
                update_interval = max(1, total_frames // 100) if total_frames > 0 else 10
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_idx += 1
                    try:
                        results = config.model(frame, conf=conf_val, iou=config.get('iou_threshold'), verbose=False)
                        r = results[0]
                        n = len(r.boxes) if r.boxes is not None else 0
                        total_dets += n
                        out_writer.write(r.plot())
                    except Exception: pass
                    if frame_idx % update_interval == 0 or frame_idx == total_frames:
                        e = time.time()-t0
                        eta = (e/frame_idx)*(total_frames-frame_idx) if frame_idx>0 else 0
                        _video_jobs[job_id].update({
                            'frame': frame_idx, 'progress': round(frame_idx/total_frames*100,1),
                            'total_detections': total_dets, 'elapsed': round(e, 3),
                            'eta': round(eta, 1)
                        })

                out_writer.release(); cap.release()
                elapsed = time.time()-t0
                playable = h264_opened

                _save_run_meta(run_dir, {
                    'id':job_id, 'type':'视频检测', 'source':file.filename,
                    'detections':total_dets, 'elapsed_time':round(elapsed,3),
                    'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'client_id':client_id,
                    'result_image':f'/api/runs/{job_id}/{out_name}',
                    'result_data':{'total_frames':total_frames,'total_detections':total_dets,'fps':round(total_frames/elapsed,1) if elapsed>0 else 0,'playable':playable},
                    'files':[out_name]
                })

                _video_jobs[job_id].update({
                    'status':'complete', 'progress':100, 'eta':0,
                    'result_video':f'/api/runs/{job_id}/{out_name}',
                    'fps':round(total_frames/elapsed,1) if elapsed>0 else 0,
                    'playable':playable
                })
                log.info(f'✓ 视频检测完成 | {total_dets} 目标, {total_frames} 帧 | {elapsed:.2f}s | {job_id}')
                log.info(f'  Results saved to {run_dir}')
                try: tmp_path.unlink(missing_ok=True)
                except: pass
            except Exception as e:
                _video_jobs[job_id]['status'] = 'error'
                _video_jobs[job_id]['error'] = str(e)
                log.error(f'✗ 视频检测失败: {e}')
                traceback.print_exc()

        thread = threading.Thread(target=lambda: _run_queued_job(job_id, 'video', _video_jobs, process_video), daemon=True)
        thread.start()
        is_queued = _video_jobs[job_id].get('status') == 'queued'
        resp = {'job_id':job_id, 'filename':file.filename, 'status':_video_jobs[job_id]['status']}
        if is_queued:
            resp['queue_position'] = _video_jobs[job_id].get('queue_position', 0)
            resp['estimated_wait'] = _calculate_eta_for(job_id)
        return jsonify({'success':True,'message':'视频检测已启动'+(' (排队中)' if is_queued else ''),'data':resp})
    except Exception as e:
        log.error(f'✗ 视频上传失败: {e}')
        traceback.print_exc()
        return jsonify({'success':False,'error':f'视频上传失败: {str(e)}'}),500

@app.route('/api/detect/video/status/<job_id>')
def api_detect_video_status(job_id):
    job = _video_jobs.get(job_id)
    if not job: return jsonify({'success':False,'error':'任务不存在'}),404
    data = dict(job)
    if data.get('status') == 'queued':
        data['estimated_wait'] = _calculate_eta_for(job_id)
    return jsonify({'success':True,'data':data})

# ── 活跃任务查询 (前端断连恢复) ────────────────────────────────────────────
@app.route('/api/jobs/active')
def api_jobs_active():
    """返回当前客户端的所有活跃任务 (queued + processing)"""
    client_id = request.args.get('client_id', '').strip()
    if not client_id:
        return jsonify({'success': False, 'error': '缺少 client_id'}), 400
    jobs = []
    for jid, job in _video_jobs.items():
        if job.get('client_id') == client_id and job.get('status') in ('queued', 'processing'):
            j = {
                'job_id': jid, 'type': 'video',
                'status': job.get('status'), 'filename': job.get('filename', ''),
                'start_time': job.get('start_time', 0), 'progress': job.get('progress', 0),
                'queue_position': job.get('queue_position', 0)
            }
            if job.get('status') == 'queued':
                j['estimated_wait'] = _calculate_eta_for(jid)
            jobs.append(j)
    for jid, job in _batch_jobs.items():
        if job.get('client_id') == client_id and job.get('status') in ('queued', 'processing'):
            j = {
                'job_id': jid, 'type': 'batch',
                'status': job.get('status'),
                'filename': f"{job.get('total_files', 0)} 个文件",
                'start_time': job.get('start_time', 0), 'progress': job.get('progress', 0),
                'queue_position': job.get('queue_position', 0)
            }
            if job.get('status') == 'queued':
                j['estimated_wait'] = _calculate_eta_for(jid)
            jobs.append(j)
    jobs.sort(key=lambda j: j['start_time'], reverse=True)
    return jsonify({'success': True, 'data': {'jobs': jobs, 'total': len(jobs)}})

# ── 摄像头检测 ───────────────────────────────────────────────────────────
@app.route('/api/detect/camera',methods=['POST'])
def api_detect_camera():
    _ensure_dirs()
    try:
        if config.model is None: return jsonify({'success':False,'error':'模型未加载'}),500
        data = request.get_json(force=True,silent=True) or {}
        image_b64 = data.get('image_base64')
        if not image_b64: return jsonify({'success':False,'error':'未提供图片数据'}),400
        if ',' in image_b64: image_b64 = image_b64.split(',')[1]
        tmp_path = TMP_DIR / f'cam_{uuid.uuid4().hex}.jpg'
        with open(tmp_path,'wb') as f: f.write(base64.b64decode(image_b64))
        conf_val = data.get('confidence', config.get('confidence'))

        results = config.model.predict(
            source=str(tmp_path),
            conf=float(conf_val), iou=config.get('iou_threshold'), verbose=False
        )
        tmp_path.unlink(missing_ok=True)

        r = results[0]; n = len(r.boxes) if r.boxes is not None else 0
        dets = _extract_detections(r)
        return jsonify({'success':True,'data':{'total_detections':n,'detections':dets}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success':False,'error':f'检测失败: {str(e)}'}),500

@app.route('/api/camera/snapshot',methods=['POST'])
def api_camera_snapshot():
    _ensure_dirs()
    try:
        data = request.get_json(force=True,silent=True) or {}
        image_b64 = data.get('image_base64')
        if not image_b64: return jsonify({'success':False,'error':'未提供图片数据'}),400
        if ',' in image_b64: image_b64 = image_b64.split(',')[1]

        session_id = data.get('session_id')
        client_id = (data.get('client_id') or '').strip()
        sid, session = _get_or_create_camera_session(session_id)
        session['client_id'] = client_id
        run_dir = session['run_dir']
        snapshot_idx = len(session['snapshots']) + 1

        # 保存原始图片到临时文件, 调用 YOLO predict + r.plot() 原生标注
        tmp_img = TMP_DIR / f'camsnap_{uuid.uuid4().hex}.jpg'
        img_bytes = base64.b64decode(image_b64)
        with open(tmp_img, 'wb') as f:
            f.write(img_bytes)

        conf_val = data.get('confidence', config.get('confidence'))
        results = config.model.predict(
            source=str(tmp_img),
            conf=float(conf_val), iou=config.get('iou_threshold'),
            verbose=False
        )
        r = results[0]
        n = len(r.boxes) if r.boxes is not None else 0

        annotated_name = f'snapshot_{snapshot_idx:03d}.jpg'
        annotated_path = run_dir / annotated_name
        annotated = r.plot()  # 原生 YOLO 标注, 返回 BGR numpy array
        cv2.imwrite(str(annotated_path), annotated)
        tmp_img.unlink(missing_ok=True)

        # 提取检测结果 (使用服务端 predict 结果, 非前端传入)
        dets = _extract_detections(r)

        result_image = f'/api/runs/{sid}/{annotated_name}'

        snapshot_info = {
            'index': snapshot_idx,
            'result_image': result_image,
            'detections_count': n,
            'detections': dets,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        session['snapshots'].append(snapshot_info)
        session['total_detections'] += n

        log.info(f'📷 抓拍 #{snapshot_idx} | 会话 {sid} | {n} 目标')
        return jsonify({
            'success':True, 'message':f'抓拍 #{snapshot_idx} 已保存 · {n} 目标',
            'data':{
                'result_image':result_image,
                'run_name':sid,
                'session_id':sid,
                'snapshot_index':snapshot_idx,
                'total_snapshots':len(session['snapshots']),
                'detections_count':n
            }
        })
    except Exception as e:
        log.error(f'✗ 抓拍失败: {e}')
        traceback.print_exc()
        return jsonify({'success':False,'error':f'抓拍失败: {str(e)}'}),500

@app.route('/api/camera/finalize',methods=['POST'])
def api_camera_finalize():
    data = request.get_json(force=True,silent=True) or {}
    session_id = data.get('session_id')
    if not session_id or session_id not in _camera_sessions:
        return jsonify({'success':False,'error':'会话不存在'}),404
    _finalize_camera_session(session_id)
    log.info(f'📷 摄像头会话结束: {session_id}')
    return jsonify({'success':True,'message':'摄像头会话已保存'})

# ── 文件大小检查 ─────────────────────────────────────────────────────────
@app.route('/api/check-file-size',methods=['POST'])
def api_check_file_size():
    data = request.get_json(force=True,silent=True) or {}
    size_bytes = data.get('size_bytes', 0)
    size_mb = size_bytes / (1024*1024)
    warning = None
    if size_mb > 20:
        warning = f'文件大小 {size_mb:.1f}MB，文件过大可能导致检测时间较长'
    elif size_mb > 10:
        warning = f'文件大小 {size_mb:.1f}MB，检测可能需要较长时间'
    return jsonify({'success':True,'data':{'size_mb':round(size_mb,1),'warning':warning}})

# ── 运行结果文件服务 ─────────────────────────────────────────────────────
@app.route('/api/runs/<path:filepath>')
def api_serve_run(filepath):
    path = RUNS_DIR / filepath
    if not path.exists():
        return jsonify({'success': False, 'error': '文件不存在'}), 404
    # 视频文件显式设置 MIME type, 确保浏览器识别为可播放视频
    mimetype = None
    if path.suffix.lower() == '.mp4':
        mimetype = 'video/mp4'
    return send_from_directory(str(RUNS_DIR), filepath, mimetype=mimetype)

@app.route('/api/uploads/<path:filename>')
def api_serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

# ── 历史记录 ──────────────────────────────────────────────────────────────
@app.route('/api/history')
def api_history():
    runs = _scan_runs()
    return jsonify({'success':True,'data':{'history':runs,'total':len(runs)}})

@app.route('/api/history/check')
def api_history_check():
    """轻量检查: 返回最新记录的 ID、时间戳和总数, 前端对比后决定是否拉取完整列表"""
    latest = None
    total = 0
    if RUNS_DIR.exists():
        dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        total = len(dirs)
        if dirs:
            d = dirs[0]
            latest = {'id': d.name, 'mtime': d.stat().st_mtime}
    return jsonify({'success':True,'data':{'latest':latest,'total':total}})

@app.route('/api/history/<history_id>')
def api_history_detail(history_id):
    meta_file = RUNS_DIR / history_id / 'metadata.json'
    if not meta_file.exists(): return jsonify({'success':False,'error':'记录不存在'}),404
    try:
        with open(meta_file,'r',encoding='utf-8') as f:
            return jsonify({'success':True,'data':json.load(f)})
    except:
        return jsonify({'success':False,'error':'读取失败'}),500

@app.route('/history/<history_id>')
def history_detail_page(history_id):
    meta_file = RUNS_DIR / history_id / 'metadata.json'
    if not meta_file.exists():
        return '记录不存在', 404
    try:
        with open(meta_file,'r',encoding='utf-8') as f:
            data = json.load(f)
    except:
        return '读取失败', 500
    return send_from_directory(str(BASE_DIR / 'static'), 'pages/history.html')

@app.route('/api/history/clear',methods=['POST'])
def api_clear_history():
    try:
        # 先清理所有活跃的摄像头会话
        _camera_sessions.clear()
        for d in RUNS_DIR.iterdir():
            if d.is_dir(): shutil.rmtree(str(d), ignore_errors=True)
        log.info('🗑 历史记录全部清空')
        return jsonify({'success':True,'message':'所有记录已清空'})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/history/<history_id>',methods=['DELETE'])
def api_delete_history(history_id):
    d = RUNS_DIR / history_id
    if d.exists():
        shutil.rmtree(str(d), ignore_errors=True)
        log.info(f'🗑 删除记录: {history_id}')
        return jsonify({'success':True,'message':'记录已删除'})
    return jsonify({'success':False,'error':'记录不存在'}),404

# ── 启动 ──────────────────────────────────────────────────────────────────
def init_app():
    _ensure_dirs()
    log.info('=' * 50)
    log.info('  YOLOv8 Web 应用 v5 启动中...')
    log.info('=' * 50)
    local = config.scan_local()
    log.info(f'本地模型: {len(local)} 个')
    for name, data in local.items():
        log.info(f'  • {name} ({data["size_mb"]}MB)')
    log.info(f'加载模型: {config.model_name()}...')
    if load_model():
        mi = config.model_info()
        log.info(f'模型加载成功: {mi.get("name", config.model_name())}')
    else:
        log.warning('模型加载失败!')
    log.info(f'访问地址: http://localhost:5000')
    log.info(f'结果目录: {RUNS_DIR}')
    log.info('=' * 50)

if __name__ == '__main__':
    init_app()
    serve(app, host='0.0.0.0', port=5000)
    