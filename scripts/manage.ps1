# YOLOv8 Web App - 统一项目管理脚本
# 用法: .\manage.ps1 [start|stop|restart|status]
# 示例: .\scripts\manage.ps1 start   (从项目根目录运行)

param(
    [ValidateSet("start","stop","restart","status")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
# 项目根目录 (脚本位于 scripts/ 下)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$port = 5000

function Write-Step { param($msg,$color="Cyan") Write-Host "  " -NoNewline; Write-Host $msg -ForegroundColor $color }
function Write-OK { Write-Host "  [✓] " -NoNewline -ForegroundColor Green; Write-Host $args[0] }
function Write-Err { Write-Host "  [✗] " -NoNewline -ForegroundColor Red; Write-Host $args[0] }
function Write-Warn { Write-Host "  [!] " -NoNewline -ForegroundColor Yellow; Write-Host $args[0] }

function Get-PidOnPort {
    $conn = netstat -ano 2>$null | Select-String ":$port.*LISTENING"
    if ($conn) { return ($conn -split '\s+')[-1] }
    return $null
}

function Stop-Service {
    Write-Step "正在停止服务..." "Yellow"
    $found = $false
    $procId = Get-PidOnPort
    if ($procId) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue; $found = $true }
    Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
            if ($cmd -match "web_app") { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue; $found = $true }
        } catch {}
    }
    Start-Sleep -Seconds 2
    if (-not (Get-PidOnPort)) { Write-OK "服务已停止" }
    elseif ($found) { Write-Warn "部分进程已终止，端口可能仍被占用" }
    else { Write-Warn "未发现运行中的服务" }
}

function Start-Service {
    Write-Step "正在启动服务..." "Yellow"
    Set-Location $ProjectRoot

    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { Write-Err "未找到 Python，请先安装 Python 3.10+"; return }

    # 检查 conda 环境
    $condaEnv = $env:CONDA_DEFAULT_ENV
    if ($condaEnv) { Write-Step "使用 Conda 环境: $condaEnv" "Green" }

    # 检查依赖
    $deps = python -c "import flask,ultralytics,cv2,PIL; print('ok')" 2>$null
    if ($deps -ne "ok") {
        Write-Warn "正在安装依赖..."
        pip install flask flask-cors ultralytics opencv-python pillow waitress -q
    }

    $existingPid = Get-PidOnPort
    if ($existingPid) { Write-Warn "端口 $port 已被占用(PID:$existingPid)，正在释放..."; Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue; Start-Sleep 2 }

    Write-Step "启动 app/web_app.py ..."
    Start-Process python -ArgumentList "app/web_app.py" -WindowStyle Minimized
    Start-Sleep -Seconds 5

    $newPid = Get-PidOnPort
    if ($newPid) {
        Write-OK "服务已启动 (PID: $newPid)"
        Write-OK "访问地址: http://localhost:$port"
        Start-Process "http://localhost:$port"
    } else {
        Write-Err "服务可能仍在加载中，请稍后访问 http://localhost:$port"
        Start-Process "http://localhost:$port"
    }
}

function Show-Status {
    $procId = Get-PidOnPort
    if ($procId) {
        Write-OK "服务运行中 (PID: $procId)"
        Write-Step "  访问: http://localhost:$port" "White"
        try {
            $r = Invoke-RestMethod "http://localhost:$port/api/status" -TimeoutSec 5
            if ($r.success) {
                Write-OK "模型: $($r.data.current_model)"
            }
        } catch { Write-Warn "服务响应异常" }
    } else {
        Write-Warn "服务未运行"
    }
}

# ── 入口 ──
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║     YOLOv8 智能目标检测 Web 应用 v2          ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

switch ($Action) {
    "start"   { Start-Service }
    "stop"    { Stop-Service }
    "restart" { Stop-Service; Start-Service }
    "status"  { Show-Status }
}
