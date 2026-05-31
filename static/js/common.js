/** YOLOv8 Web App — 共享工具函数 + 统一模糊遮罩加载 */

/**
 * HTML 转义
 */
function escapeHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/**
 * Toast 通知
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {string} msg
 */
function toast(type, msg) {
    const wrap = document.getElementById('toastWrap');
    if (!wrap) return;

    const icons = {
        success: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };

    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.innerHTML = (icons[type] || icons.info) + '<span class="toast-msg">' + msg + '</span>';
    wrap.appendChild(el);
    setTimeout(function () { el.remove(); }, type === 'info' ? 5000 : 3000);
}

// ── 统一模糊遮罩加载 ─────────────────────────────────────────────────────

/**
 * 显示模糊遮罩 (spinner 模式)
 * @param {string} msg - 主提示文字
 * @param {string} [warning] - 可选的警告提示 (如大文件提醒), 显示在遮罩内部
 */
function showLoading(msg, warning) {
    var spinner = document.getElementById('loadingSpinner');
    var progress = document.getElementById('loadingProgressWrap');
    var warnEl = document.getElementById('loadingWarning');
    if (spinner) spinner.style.display = '';
    if (progress) progress.style.display = 'none';
    document.getElementById('loadingText').textContent = msg || '处理中...';
    if (warnEl) {
        if (warning) { warnEl.textContent = '⚠ ' + warning; warnEl.style.display = ''; }
        else { warnEl.textContent = ''; warnEl.style.display = 'none'; }
    }
    document.getElementById('loading').classList.add('active');
}

/**
 * 显示模糊遮罩 (进度条模式)
 * @param {string} msg - 主提示文字
 * @param {boolean} [indeterminate] - true=不确定进度(扫动动画), false/省略=确定进度(0%起)
 * @param {string} [warning] - 可选的警告提示, 显示在遮罩内部
 */
function showLoadingProgress(msg, indeterminate, warning) {
    var spinner = document.getElementById('loadingSpinner');
    var progress = document.getElementById('loadingProgressWrap');
    var bar = document.getElementById('loadingProgressBar');
    var warnEl = document.getElementById('loadingWarning');
    if (spinner) spinner.style.display = 'none';
    if (progress) {
        progress.style.display = '';
        bar.classList.toggle('indeterminate', !!indeterminate);
        if (indeterminate) {
            bar.style.width = '';
            document.getElementById('loadingProgressPct').textContent = '';
        } else {
            bar.style.width = '0%';
            document.getElementById('loadingProgressPct').textContent = '0%';
        }
        document.getElementById('loadingProgressStatus').textContent = '';
        document.getElementById('loadingProgressLive').innerHTML = '';
    }
    document.getElementById('loadingText').textContent = msg || '处理中...';
    if (warnEl) {
        if (warning) { warnEl.textContent = '⚠ ' + warning; warnEl.style.display = ''; }
        else { warnEl.textContent = ''; warnEl.style.display = 'none'; }
    }
    document.getElementById('loading').classList.add('active');
}

/**
 * 更新进度条
 * @param {number} pct - 0-100
 * @param {string} [status] - 状态文字（如 "检测中... 120/300 帧"）
 * @param {string} [live] - 实时数据 (HTML)
 */
function updateLoadingProgress(pct, status, live) {
    var bar = document.getElementById('loadingProgressBar');
    bar.classList.remove('indeterminate');
    if (bar) bar.style.width = pct + '%';
    document.getElementById('loadingProgressPct').textContent = Math.round(pct) + '%';
    if (status !== undefined) document.getElementById('loadingProgressStatus').textContent = status;
    if (live !== undefined) document.getElementById('loadingProgressLive').innerHTML = live;
}

/**
 * 隐藏模糊遮罩
 */
function hideLoading() {
    document.getElementById('loading').classList.remove('active');
    var warnEl = document.getElementById('loadingWarning');
    if (warnEl) { warnEl.textContent = ''; warnEl.style.display = 'none'; }
}
