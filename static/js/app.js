/** YOLOv8 Web App v5 — 优化进度条/历史刷新/批量混搭 */
const state={mode:'image',config:null,model:null,history:[],_latestMtime:null,_latestCount:0,detecting:false,cameraStream:null,cameraActive:false,cameraTimer:null,cameraFpsTimer:null,cameraFrameCount:0,_lastCameraDets:[],_lastCameraB64:'',cameraSessionId:null,cameraSnapshotCount:0,_batchMediaItems:[],_batchMediaIndex:0,_previewMode:false};
document.addEventListener('DOMContentLoaded',()=>{init();setupUpload();setupKeyboard()});

async function init(){
    try{const r=await fetch('/api/status');const d=await r.json();if(d.success){state.model=d.data;state.config=d.data.config;updateModelDisplay();updateStatus();updateSliders()}}catch(e){toast('error','初始化失败: '+e.message)}
    loadHistory();
    // 轻量轮询历史更新 (每10秒检查一次)
    setInterval(checkHistoryUpdate,10000);
}

async function checkHistoryUpdate(){
    try{const r=await fetch('/api/history/check');const d=await r.json();if(d.success){
        var count=d.data.total||0;
        if(state._latestCount!==count){await loadHistory()}
    }}catch(e){/* 静默失败 */}
}

function setupUpload(){
    const b=document.getElementById('uploadBox'),i=document.getElementById('fileInput');
    b.addEventListener('click',e=>{if(e.target.closest('.btn-secondary')||e.target.closest('.btn-primary'))return;i.click()});
    document.getElementById('selectBtn').addEventListener('click',e=>{e.stopPropagation();i.click()});
    i.addEventListener('change',e=>{
        const files=e.target.files;
        if(!files||!files.length)return;
        if(state.mode==='batch'){checkFileSize(files[0]);detectBatch(Array.from(files))}
        else{checkFileSize(files[0]);handleFile(files[0])}
    });
    b.addEventListener('dragover',e=>{e.preventDefault();b.classList.add('dragover')});
    b.addEventListener('dragleave',()=>{b.classList.remove('dragover')});
    b.addEventListener('drop',e=>{
        e.preventDefault();b.classList.remove('dragover');
        const files=e.dataTransfer.files;
        if(!files||!files.length)return;
        if(state.mode==='batch'){checkFileSize(files[0]);detectBatch(Array.from(files))}
        else{checkFileSize(files[0]);handleFile(files[0])}
    });
    document.getElementById('urlInput').addEventListener('keypress',e=>{if(e.key==='Enter')detectUrl()})
}

function setupKeyboard(){document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeAllModals();if(state._previewMode){state._previewMode=false;resetAll();switchMode(state.mode)}}if(state._previewMode&&(e.key==='ArrowLeft'||e.key==='ArrowRight')){e.preventDefault();navigateBatchPreview(e.key==='ArrowLeft'?-1:1)}})}

function switchMode(m){
    state.mode=m;
    document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
    document.getElementById('uploadArea').style.display=(m==='image'||m==='batch'||m==='video')?'block':'none';
    document.getElementById('urlArea').style.display=m==='url'?'block':'none';
    document.getElementById('cameraArea').style.display=m==='camera'?'block':'none';
    var sb=document.getElementById('sampleBtn');if(sb)sb.style.display=(m==='image'||m==='url')?'':'none';
    const h=document.getElementById('uploadHint'),i=document.getElementById('fileInput');
    if(m==='image'){h.textContent='或点击选择图片文件';i.accept='image/*';i.multiple=false}
    else if(m==='batch'){h.textContent='可多选图片和视频，支持混搭上传';i.accept='image/*,video/*';i.multiple=true}
    else if(m==='video'){h.textContent='或点击选择视频文件';i.accept='video/*';i.multiple=false}
    if(m!=='camera'&&state.cameraActive)stopCamera();
    hideResults();
    document.getElementById('fileSizeWarning').style.display='none';
}

function handleFile(f){if(state.mode==='image')detectImage(f);else if(state.mode==='batch'){const i=document.getElementById('fileInput');detectBatch(Array.from(i.files))}else if(state.mode==='video')detectVideo(f)}
function loadSample(){detectSample('bus')}
function detectSample(name){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}state.detecting=true;fetch('/api/samples/check/'+name).then(function(r){return r.json()}).then(function(check){if(check.data&&check.data.exists){showLoading('正在检测示例图片...')}else{showLoadingProgress('图片正在下载，请稍后',true)}return fetch('/api/detect/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,confidence:state.config.confidence})})}).then(function(r){return r.json()}).then(function(d){if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}).catch(function(e){toast('error','示例检测失败: '+e.message)}).finally(function(){hideLoading();state.detecting=false})}
function setUrl(u){document.getElementById('urlInput').value=u}

// ── 文件大小检查 ─────────────────────────────────────────────────────────
async function checkFileSize(f){
    const warnEl=document.getElementById('fileSizeWarning'),warnText=document.getElementById('fileSizeWarningText');
    warnEl.style.display='none';
    try{const r=await fetch('/api/check-file-size',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({size_bytes:f.size})});const d=await r.json();if(d.success&&d.data.warning){warnText.textContent=d.data.warning;warnEl.style.display='flex'}}catch(e){}
}

// ── 图片检测 ──────────────────────────────────────────────────────────
async function detectImage(f){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}state.detecting=true;showLoadingProgress('正在检测图片...', true);try{const fd=new FormData();fd.append('file',f);fd.append('confidence',state.config.confidence);const r=await fetch('/api/detect/image',{method:'POST',body:fd});const d=await r.json();if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}catch(e){toast('error','检测失败: '+e.message)}finally{hideLoading();state.detecting=false}}

function detectBatch(fs){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}state.detecting=true;let pollInterval=null;var bar=document.getElementById('loadingProgressBar');var pctEl=document.getElementById('loadingProgressPct');var statusEl=document.getElementById('loadingProgressStatus');var liveEl=document.getElementById('loadingProgressLive');var hasVideos=fs.some(function(f){var n=f.name||'';return /\.(mp4|avi|mov|mkv|webm|wmv|flv)$/i.test(n)});showLoadingProgress('正在批量检测 '+fs.length+' 个文件...', !hasVideos);try{var fd=new FormData();fs.forEach(function(f){fd.append('files',f)});fd.append('confidence',state.config.confidence);fetch('/api/detect/batch',{method:'POST',body:fd}).then(function(r){return r.json()}).then(function(d){if(!d.success){toast('error',d.error);hideLoading();state.detecting=false;return}var jobId=d.data.job_id;hideAllInputs();document.getElementById('statsRow').style.display='none';document.getElementById('detailTable').style.display='none';document.getElementById('resultArea').style.display='none';pollInterval=setInterval(function(){fetch('/api/detect/batch/status/'+jobId).then(function(r){return r.json()}).then(function(pd){if(!pd.success)return;var j=pd.data;if(j.status==='processing'){var fileLabel='第 '+(j.current_file_index+1)+'/'+j.total_files+' 个文件';var status=fileLabel+': '+j.current_file_name;if(j.current_file_type==='image'){bar.classList.add('indeterminate');bar.style.width='';pctEl.textContent='';statusEl.textContent=status;liveEl.innerHTML='目标累计: <b>'+j.total_detections+'</b> | 耗时: '+j.elapsed+'s'}else{bar.classList.remove('indeterminate');var fps=j.current_file_frame/Math.max(j.elapsed||0.001,0.001);var live='目标累计: <b>'+j.total_detections+'</b> | 帧: '+j.current_file_frame+'/'+j.current_file_total_frames+' | FPS: '+Math.round(fps)+' | 耗时: '+j.elapsed+'s';if(j.eta)status+=' | 预计剩余 '+formatETA(j.eta);updateLoadingProgress(j.progress,status,live)}}else if(j.status==='complete'){clearInterval(pollInterval);pollInterval=null;updateLoadingProgress(100,'✓ 批量检测完成 — '+j.total_files+' 个文件, '+j.total_detections+' 目标, '+j.elapsed+'s','');hideLoading();toast('success','批量检测完成: '+j.total_detections+' 个目标');showBatchResult({total_images:j.total_images,total_videos:j.total_videos,total_detections:j.total_detections,elapsed_time:j.elapsed,results:j.results,run_name:j.run_name});state.detecting=false;loadHistory()}else if(j.status==='error'){clearInterval(pollInterval);pollInterval=null;hideLoading();toast('error','批量检测失败: '+(j.error||'未知错误'));state.detecting=false}}).catch(function(){/* 轮询网络错误-继续 */})},300)}).catch(function(e){toast('error','批量检测失败: '+e.message);hideLoading();state.detecting=false})}catch(e){toast('error','批量检测失败: '+e.message);hideLoading();if(pollInterval)clearInterval(pollInterval);state.detecting=false}}

async function detectUrl(u){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}u=u||document.getElementById('urlInput').value.trim();if(!u){toast('warning','请输入URL');return}state.detecting=true;showLoadingProgress('正在下载并检测...', true);try{const r=await fetch('/api/detect/url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u,confidence:state.config.confidence})});const d=await r.json();if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}catch(e){toast('error','URL检测失败: '+e.message)}finally{hideLoading();state.detecting=false}}

// ── 显示图片结果 ─────────────────────────────────────────────────────
function showImageResult(d){hideAllInputs();const ra=document.getElementById('resultArea');ra.style.display='block';document.getElementById('imageBox').innerHTML='<img id="resultImage" src="" alt="">';const img=document.getElementById('resultImage');img.style.display='block';img.classList.add('loading-img');img.src=d.result_image;img.onload=()=>{img.classList.remove('loading-img')};img.onerror=()=>{img.classList.remove('loading-img');toast('error','标注图片加载失败')};showStats(d);if(d.detections&&d.detections.length>0)showDetailTable(d.detections);else{document.getElementById('detailTable').style.display='none'}}

// ── 视频检测 (轮询 + 优化进度) ───────────────────────────────────────
function formatETA(secs){if(!secs||secs<=0)return'计算中...';if(secs<60)return Math.round(secs)+'秒';if(secs<3600)return Math.floor(secs/60)+'分'+Math.round(secs%60)+'秒';return Math.floor(secs/3600)+'时'+Math.floor((secs%3600)/60)+'分'}

async function detectVideo(f){
    if(state.detecting)return;
    if(!state.config){toast('warning','正在初始化，请稍后再试');return}
    state.detecting=true;
    let pollInterval=null;
    try{
        showLoading('正在上传视频...');
        const fd=new FormData();fd.append('file',f);fd.append('confidence',state.config.confidence);
        const uR=await fetch('/api/detect/video',{method:'POST',body:fd});
        const uD=await uR.json();
        if(!uD.success){toast('error',uD.error);hideLoading();state.detecting=false;return}

        // 切换到模糊遮罩进度条模式
        showLoadingProgress('正在分析视频...');
        hideAllInputs();
        document.getElementById('statsRow').style.display='none';
        document.getElementById('detailTable').style.display='none';
        document.getElementById('resultArea').style.display='none';

        const jobId=uD.data.job_id;

        pollInterval=setInterval(async()=>{
            try{
                const r=await fetch('/api/detect/video/status/'+jobId);
                const d=await r.json();
                if(!d.success)return;
                const j=d.data;

                const fps=Math.round(j.frame/Math.max(j.elapsed,0.001));
                let status='检测中... '+j.frame+'/'+j.total_frames+' 帧';
                if(j.eta)status+=' | 预计剩余 '+formatETA(j.eta);
                let live='目标累计: <b>'+j.total_detections+'</b> | FPS: '+fps+' | 耗时: '+j.elapsed+'s';

                updateLoadingProgress(j.progress, status, live);

                if(j.status==='complete'){
                    clearInterval(pollInterval);pollInterval=null;
                    updateLoadingProgress(100, '✓ 检测完成 — '+j.total_frames+' 帧, '+j.total_detections+' 目标, '+j.elapsed+'s', '目标累计: <b style="color:var(--success)">'+j.total_detections+'</b> | FPS: '+j.fps+' | 耗时: '+j.elapsed+'s');
                    hideLoading();
                    showStats({total_detections:j.total_detections, elapsed_time:j.elapsed, fps:j.fps});
                    if(j.result_video){showVideoResult(j.result_video,j.playable)}
                    toast('success','视频检测完成');
                    state.detecting=false;loadHistory();
                }else if(j.status==='error'){
                    clearInterval(pollInterval);pollInterval=null;
                    hideLoading();
                    toast('error','视频检测失败: '+(j.error||'未知错误'));
                    state.detecting=false;
                }
            }catch(e){/* 轮询网络错误-继续 */}
        },300);

    }catch(e){toast('error','视频检测失败: '+e.message);hideLoading();if(pollInterval)clearInterval(pollInterval);state.detecting=false}
}

function showVideoResult(videoUrl,playable){
    const ra=document.getElementById('resultArea');ra.style.display='block';
    const ib=document.getElementById('imageBox');
    if(playable!==false){
        ib.innerHTML='<video id="resultVideo" controls autoplay loop playsinline style="max-width:100%;max-height:60vh;border-radius:8px;background:#000"><source src="'+videoUrl+'" type="video/mp4"></video>';
        const vid=document.getElementById('resultVideo');
        if(vid){vid.addEventListener('error',function(){ib.innerHTML='<div class="video-fallback"><p>视频无法播放</p><a href="'+videoUrl+'" download class="btn-primary">下载视频</a><p class="hint">安装 ffmpeg 后可在浏览器播放: conda install -c conda-forge ffmpeg</p></div>'})}
    }else{
        ib.innerHTML='<div class="video-fallback"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text3)" stroke-width="1.5" style="margin-bottom:16px"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><p>视频已保存，浏览器无法直接播放</p><p class="hint">安装 ffmpeg 后自动转为浏览器兼容格式</p><a href="'+videoUrl+'" download class="btn-primary">下载视频文件</a></div>';
    }
}

function hideAllInputs(){document.getElementById('uploadArea').style.display='none';document.getElementById('urlArea').style.display='none';document.getElementById('cameraArea').style.display='none';document.getElementById('progressWrap').style.display='none'}

// ── 摄像头 ─────────────────────────────────────────────────────────────
function toggleCamera(){if(state.cameraActive)stopCamera();else startCamera()}

async function startCamera(){
    try{
        const ds=await navigator.mediaDevices.enumerateDevices();
        const vds=ds.filter(d=>d.kind==='videoinput');
        const s=document.getElementById('cameraSelect');
        s.innerHTML='';
        vds.forEach((d,i)=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||'摄像头 '+(i+1);s.appendChild(o)});
        s.onchange=()=>switchCamera(s.value);
        const st=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}}});
        const v=document.getElementById('cameraVideo');v.srcObject=st;v.muted=true;
        state.cameraStream=st;state.cameraActive=true;
        state.cameraSessionId=null;state.cameraSnapshotCount=0;
        document.getElementById('cameraToggle').innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> 关闭摄像头';
        document.getElementById('captureBtn').disabled=false;document.getElementById('snapshotBtn').disabled=false;
        document.getElementById('cameraSnaps').style.display='inline';document.getElementById('cameraSnaps').textContent='0 抓拍';
        toast('success','摄像头已开启');startRealtimeDetection();
    }catch(e){
        console.error('摄像头启动失败:',e);
        if(e.name==='NotAllowedError')toast('error','摄像头权限被拒绝，请在浏览器设置中允许访问摄像头');
        else if(e.name==='NotFoundError')toast('error','未检测到摄像头设备');
        else if(e.name==='NotReadableError')toast('error','摄像头被其他应用占用');
        else toast('error','无法打开摄像头: '+e.message);
    }
}

function stopCamera(){
    if(state.cameraSessionId&&state.cameraSnapshotCount>0){fetch('/api/camera/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:state.cameraSessionId})}).then(()=>{loadHistory()}).catch(e=>console.error(e))}
    if(state.cameraStream){state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null}
    if(state.cameraTimer){clearInterval(state.cameraTimer);state.cameraTimer=null}
    if(state.cameraFpsTimer){clearInterval(state.cameraFpsTimer);state.cameraFpsTimer=null}
    const v=document.getElementById('cameraVideo');v.srcObject=null;
    state.cameraActive=false;state.cameraFrameCount=0;state.cameraSessionId=null;state.cameraSnapshotCount=0;
    document.getElementById('cameraToggle').innerHTML='<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg> 开启摄像头';
    document.getElementById('captureBtn').disabled=true;document.getElementById('snapshotBtn').disabled=true;
    document.getElementById('cameraFps').textContent='-- FPS';document.getElementById('cameraDets').textContent='0 目标';
    document.getElementById('cameraSnaps').style.display='none';
}

async function switchCamera(did){if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());try{const st=await navigator.mediaDevices.getUserMedia({video:{deviceId:{exact:did},width:{ideal:1280},height:{ideal:720}}});const v=document.getElementById('cameraVideo');v.srcObject=st;v.muted=true;state.cameraStream=st}catch(e){toast('error','切换摄像头失败')}}

function startRealtimeDetection(){
    if(!state.cameraActive)return;state.cameraFrameCount=0;
    state.cameraFpsTimer=setInterval(()=>{document.getElementById('cameraFps').textContent=state.cameraFrameCount+' FPS';if(state.cameraFrameCount>15)document.getElementById('cameraFps').style.color='#34C759';else document.getElementById('cameraFps').style.color='var(--text2)';state.cameraFrameCount=0},1000);
    async function loop(){if(!state.cameraActive)return;await captureAndDetect();if(state.cameraActive)state.cameraTimer=setTimeout(loop,100)}loop()
}

async function captureAndDetect(){
    if(!state.cameraActive||state.detecting)return;const v=document.getElementById('cameraVideo');if(!v.videoWidth)return;
    const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);
    const b64=c.toDataURL('image/jpeg',0.8);state.detecting=true;state.cameraFrameCount++;
    try{const r=await fetch('/api/detect/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:b64,confidence:state.config.confidence})});const d=await r.json();if(d.success){drawCameraResult(d.data);const td=d.data.total_detections;document.getElementById('cameraDets').textContent=td+' 目标';document.getElementById('cameraDets').style.color=td>0?'#34C759':'var(--text2)';state._lastCameraDets=d.data.detections;state._lastCameraB64=b64}}catch(e){console.error(e)}finally{state.detecting=false}
}

function drawCameraResult(d){
    const v=document.getElementById('cameraVideo'),c=document.getElementById('cameraCanvas'),ctx=c.getContext('2d');
    c.width=v.videoWidth;c.height=v.videoHeight;
    const wr=v.parentElement,wb=wr.getBoundingClientRect(),vb=v.getBoundingClientRect();
    c.style.position='absolute';c.style.left=(vb.left-wb.left)+'px';c.style.top=(vb.top-wb.top)+'px';c.style.width=vb.width+'px';c.style.height=vb.height+'px';
    ctx.clearRect(0,0,c.width,c.height);
    const cl=['#FF3B30','#FF9500','#FFCC00','#34C759','#007AFF','#5856D6','#AF52DE','#FF2D55'];
    d.detections.forEach((dt,i)=>{const[x1,y1,x2,y2]=dt.bbox,co=cl[i%cl.length];ctx.strokeStyle=co;ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);const lb=dt.class_name+' '+(dt.confidence*100).toFixed(0)+'%';ctx.font='bold 14px sans-serif';const tw=ctx.measureText(lb).width;ctx.fillStyle=co;ctx.fillRect(x1,y1-22,tw+10,22);ctx.fillStyle='#fff';ctx.fillText(lb,x1+5,y1-6)})
}

async function snapshotCamera(){
    if(!state.cameraActive)return;if(!state._lastCameraB64){toast('warning','请等待检测完成');return}
    try{const r=await fetch('/api/camera/snapshot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:state._lastCameraB64,detections:state._lastCameraDets||[],session_id:state.cameraSessionId})});const d=await r.json();if(d.success){state.cameraSessionId=d.data.session_id;state.cameraSnapshotCount=d.data.total_snapshots;document.getElementById('cameraSnaps').textContent=d.data.total_snapshots+' 抓拍';toast('success',d.message)}else toast('error',d.error)}catch(e){toast('error','抓拍失败')}
}

// ── UI 辅助 ─────────────────────────────────────────────────────────────
function showStats(d){const sr=document.getElementById('statsRow');sr.style.display='grid';document.getElementById('statDets').textContent=d.total_detections;document.getElementById('statTime').textContent=d.elapsed_time+'s';document.getElementById('statFps').textContent=d.fps+' FPS'}
function showDetailTable(ds){const dt=document.getElementById('detailTable');dt.style.display='block';const tb=document.getElementById('detailBody');tb.innerHTML='';ds.forEach((det,i)=>{const cf=det.confidence,cc=cf>=0.7?'conf-high':cf>=0.5?'conf-mid':'conf-low';const row=document.createElement('tr');row.innerHTML='<td>'+(i+1)+'</td><td><span class="class-tag">'+det.class_name+'</span></td><td><span class="conf-badge '+cc+'">'+(cf*100).toFixed(1)+'%</span></td><td style="font-family:monospace;font-size:12px">'+det.bbox.map(v=>Math.round(v)).join(', ')+'</td>';tb.appendChild(row)})}

function showBatchResult(d){
    hideAllInputs();const ra=document.getElementById('resultArea');ra.style.display='block';
    // 存储媒体列表供预览使用
    state._batchMediaItems=d.results.map(br=>({url:br.result_image||br.result_video||'',type:br.type,label:br.filename,dets:br.detections_count}));
    state._batchMediaIndex=0;state._previewMode=false;
    document.getElementById('imageBox').innerHTML='<div class="batch-summary"><div class="batch-summary-icon">✓</div><div><h4>批量检测完成</h4><p>'+d.total_images+' 图片'+(d.total_videos>0?' + '+d.total_videos+' 视频':'')+' · '+d.total_detections+' 个目标 · '+d.elapsed_time+'s</p><p style="margin-top:8px;font-size:12px;color:var(--text3)">💡 点击下方列表项预览结果</p></div></div>';
    document.getElementById('statsRow').style.display='grid';
    document.getElementById('statDets').textContent=d.total_detections;
    document.getElementById('statTime').textContent=d.elapsed_time+'s';
    document.getElementById('statFps').textContent='-';
    document.getElementById('detailTable').style.display='block';
    const tb=document.getElementById('detailBody');tb.innerHTML='';
    d.results.forEach((br,i)=>{
        const icon=br.type==='video'?'🎬':'🖼';
        const row=document.createElement('tr');
        row.className='batch-result-row';
        row.style.cursor='pointer';
        row.innerHTML='<td>'+br.index+'</td><td>'+icon+' '+br.filename+'</td><td>'+br.detections_count+'</td><td>'+(br.detections&&br.detections.length>0?br.detections.map(dt=>dt.class_name+' '+(dt.confidence*100).toFixed(0)+'%').join(', '):'无')+'</td>';
        row.addEventListener('click',()=>showBatchItemPreview(i));
        row.addEventListener('mouseenter',function(){this.style.background='var(--primary-bg)';this.classList.add('hover')});
        row.addEventListener('mouseleave',function(){this.style.background='';this.classList.remove('hover')});
        tb.appendChild(row);
    })
    // 默认预览第一个有效项目
    const firstValid=state._batchMediaItems.findIndex(m=>m.url);
    if(firstValid>=0)showBatchItemPreview(firstValid);
}

// ── 批量结果预览（带左右箭头导航）──────────────────────────────────
function showBatchItemPreview(idx){
    if(idx<0||idx>=state._batchMediaItems.length)return;
    state._batchMediaIndex=idx;state._previewMode=true;
    const item=state._batchMediaItems[idx];
    if(!item.url){document.getElementById('imageBox').innerHTML='<div class="batch-summary" style="padding:40px;text-align:center;color:var(--text3)">⚠ 此项目无可用预览</div>';return}
    let mediaEl='';
    if(item.type==='video')mediaEl='<video id="batchPreviewVideo" controls autoplay loop playsinline style="max-width:100%;max-height:55vh;border-radius:8px;background:#000"><source src="'+item.url+'" type="video/mp4"></video>';
    else mediaEl='<img id="batchPreviewImage" src="'+item.url+'" alt="'+item.label+'" style="max-width:100%;max-height:55vh;object-fit:contain;border-radius:8px" onerror="this.parentElement.innerHTML=\'<div style=padding:40px;text-align:center;color:var(--text3)>图片加载失败</div>\'">';
    const total=state._batchMediaItems.length;
    document.getElementById('imageBox').innerHTML=
        '<div class="media-viewer">'+
        (total>1?'<button class="media-nav-btn media-prev" onclick="navigateBatchPreview(-1)" title="上一个 (←)"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg></button>':'')+
        '<div class="media-viewer-content">'+mediaEl+'<div class="media-viewer-info"><span class="media-viewer-label">'+escapeHtml(item.label)+'</span><span class="media-viewer-counter">'+(idx+1)+' / '+total+'</span><span class="media-viewer-dets">'+item.dets+' 目标</span></div></div>'+
        (total>1?'<button class="media-nav-btn media-next" onclick="navigateBatchPreview(1)" title="下一个 (→)"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg></button>':'')+
        '</div>';
    // 高亮表格中当前行
    document.querySelectorAll('.batch-result-row').forEach((r,i)=>{r.classList.toggle('selected',i===idx)});
}

function navigateBatchPreview(direction){
    if(!state._previewMode)return;
    const total=state._batchMediaItems.length;
    if(total===0)return;
    const newIdx=(state._batchMediaIndex+direction+total)%total;
    showBatchItemPreview(newIdx);
}

function resetAll(){hideResults();document.getElementById('uploadArea').style.display=(state.mode==='image'||state.mode==='batch'||state.mode==='video')?'block':'none';document.getElementById('urlArea').style.display=state.mode==='url'?'block':'none';document.getElementById('cameraArea').style.display=state.mode==='camera'?'block':'none';document.getElementById('fileInput').value='';document.getElementById('fileSizeWarning').style.display='none'}
function hideResults(){document.getElementById('resultArea').style.display='none';document.getElementById('statsRow').style.display='none';document.getElementById('detailTable').style.display='none';document.getElementById('progressWrap').style.display='none';document.getElementById('imageBox').innerHTML='<img id="resultImage" src="" alt="">'}

// ── 模型/配置弹窗 ───────────────────────────────────────────────────────
function updateModelDisplay(){const m=state.model;document.getElementById('modelName').textContent=m.model_info.name||m.current_model;document.getElementById('modelDesc').textContent=m.model_info.desc||''}
function updateStatus(){const d=document.getElementById('statusDot'),t=document.getElementById('statusText');if(state.model&&state.model.is_model_loaded){d.className='status-dot active';t.textContent='已就绪'}else{d.className='status-dot error';t.textContent='未加载'}}
function updateSliders(){if(!state.config)return;document.getElementById('confVal').textContent=state.config.confidence.toFixed(2);document.getElementById('confSlider').value=state.config.confidence}
async function showModelModal(){showLoading('加载模型列表...');try{const r=await fetch('/api/models');const d=await r.json();if(d.success){renderModelList(d.data);document.getElementById('modelModal').classList.add('active')}}catch(e){toast('error','加载失败')}finally{hideLoading()}}
function closeModelModal(){document.getElementById('modelModal').classList.remove('active')}
function renderModelList(d){document.getElementById('localCnt').textContent=d.stats.local;document.getElementById('remoteCnt').textContent=d.stats.remote;var l=document.getElementById('modelList');l.innerHTML='';d.models.forEach(function(m){var item=document.createElement('div');item.className='model-item'+(m.is_current?' active':'');var delBtn=m.is_local?'<button class="model-del-btn" title="删除本地模型" onclick="event.stopPropagation();deleteModel(\''+m.name+'\')">×</button>':'';var checkMark=m.is_current?'<div class="model-check">✓</div>':'';item.innerHTML='<div class="model-item-info"><div class="model-item-name">'+m.full_name+'</div><div class="model-item-desc">'+m.desc+' · '+m.params+' · '+m.num_classes+'类</div></div><div class="model-item-meta"><span>'+(m.is_local?'本地('+m.size_mb+'MB)':'需下载')+'</span></div>'+delBtn+checkMark;item.addEventListener('click',function(){selectModel(m.name)});l.appendChild(item)})}
function filterModels(q){const items=document.querySelectorAll('.model-item');items.forEach(i=>{const n=i.textContent.toLowerCase();i.style.display=n.includes(q.toLowerCase())?'flex':'none'})}
async function selectModel(n){closeModelModal();showLoading('正在切换模型: '+n+'...');var downloading=false;try{const r=await fetch('/api/models/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});const d=await r.json();if(d.success){if(d.data&&d.data.status==='downloading'){downloading=true;toast('info','开始下载 '+d.data.model_name+' ('+d.data.params+')，请稍候...');pollDownloadProgress(n,d.data.model_name);return}toast('success',d.message);const sr=await fetch('/api/status');const sd=await sr.json();if(sd.success){state.model=sd.data;state.config=sd.data.config;updateModelDisplay();updateStatus()}}else toast('error',d.error)}catch(e){toast('error','切换失败: '+e.message)}finally{if(!downloading)hideLoading()}}
function pollDownloadProgress(name,modelName){showLoadingProgress('正在下载 '+modelName+', 请耐心等待', true);var downloadStart=Date.now();var TIMEOUT=120;var errorShown=false;var check=setInterval(function(){var elapsed=Math.floor((Date.now()-downloadStart)/1000);if(elapsed>TIMEOUT){clearInterval(check);errorShown=true;updateLoadingProgress(0,'✗ 下载超时 — 已等待 '+TIMEOUT+'s，请检查网络','');setTimeout(function(){hideLoading();toast('error','下载超时: '+modelName+'。GitHub 可能无法访问，请手动下载模型文件放入 models/ 目录')},3000);return}fetch('/api/models/download-status/'+name).then(function(r){return r.json()}).then(function(d){if(errorShown)return;if(!d.success){document.getElementById('loadingProgressStatus').textContent='正在下载 '+modelName+'... (已等待 '+elapsed+'s)';return}var job=d.data;if(job.status==='complete'){clearInterval(check);updateLoadingProgress(100,'✓ 下载完成 — 正在加载模型...','');hideLoading();toast('success',modelName+' 下载完成! 正在加载...');fetch('/api/status').then(function(r){return r.json()}).then(function(sd){if(sd.success){state.model=sd.data;state.config=sd.data.config;updateModelDisplay();updateStatus()}});fetch('/api/models/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})})}else if(job.status==='error'){clearInterval(check);errorShown=true;updateLoadingProgress(0,'✗ 下载失败: '+(job.error||'未知错误'),'');setTimeout(function(){hideLoading();toast('error','下载失败: '+(job.error||'未知错误')+'。请手动下载模型文件放入 models/ 目录')},4000)}else if(job.progress>0){updateLoadingProgress(job.progress,'正在下载 '+modelName+'...','')}else{document.getElementById('loadingProgressStatus').textContent='正在下载 '+modelName+'... (已等待 '+elapsed+'s)'}}).catch(function(e){if(!errorShown)document.getElementById('loadingProgressStatus').textContent='正在下载 '+modelName+'... (已等待 '+elapsed+'s, 网络波动重试中)'})},3000)}

async function deleteModel(name){if(!confirm('确定要删除模型 '+name+' 吗？'))return;try{const r=await fetch('/api/models/'+name,{method:'DELETE'});const d=await r.json();if(d.success){toast('success',d.message);const sr=await fetch('/api/status');const sd=await sr.json();if(sd.success){state.model=sd.data;state.config=sd.data.config;updateModelDisplay();updateStatus()}showModelModal()}else toast('error',d.error)}catch(e){toast('error','删除失败: '+e.message)}}

function showConfigModal(){document.getElementById('cfgConf').value=state.config.confidence;document.getElementById('configModal').classList.add('active')}
function closeConfigModal(){document.getElementById('configModal').classList.remove('active')}
async function saveConfig(){const nc={confidence:parseFloat(document.getElementById('cfgConf').value)};try{const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(nc)});const d=await r.json();if(d.success){state.config=d.data;updateSliders();closeConfigModal();toast('success','配置已保存')}}catch(e){toast('error','保存失败')}}
async function resetConfig(){try{const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence:0.25,iou_threshold:0.7})});const d=await r.json();if(d.success){state.config=d.data;showConfigModal();updateSliders();toast('success','已恢复默认配置')}}catch(e){toast('error','恢复失败')}}
function updateConf(v){document.getElementById('confVal').textContent=parseFloat(v).toFixed(2);state.config.confidence=parseFloat(v);fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence:parseFloat(v)})})}

// ── 历史 ────────────────────────────────────────────────────────────────
async function loadHistory(){
    try{const r=await fetch('/api/history');const d=await r.json();if(d.success){state.history=d.data.history;renderHistory();state._latestCount=d.data.total||d.data.history.length}}catch(e){console.error(e)}
}

function renderHistory(){
    const l=document.getElementById('historyList');l.innerHTML='';
    if(!state.history||state.history.length===0){l.innerHTML='<div class="history-empty">暂无检测记录</div>';return}
    state.history.forEach(it=>{
        const el=document.createElement('div');el.className='history-item';
        let ic='';
        switch(it.type){
            case'图片检测':ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>';break;
            case'批量检测':ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>';break;
            case'视频检测':ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>';break;
            case'URL检测':ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/></svg>';break;
            case'摄像头检测':ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>';break;
            default:ic='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>';
        }
        let s=it.source;if(s.length>25)s=s.substring(0,25)+'...';
        el.innerHTML='<div class="history-icon">'+ic+'</div><div class="history-info"><div class="history-type">'+it.type+'</div><div class="history-src">'+s+'</div></div><div class="history-meta"><div class="history-dets">'+it.detections+'</div><div class="history-time">'+it.elapsed_time+'s</div></div>';
        el.addEventListener('click',()=>{window.location.href='/history/'+it.id});l.appendChild(el);
    })
}

async function clearHistory(){if(!confirm('确定清空所有检测记录和结果文件？此操作不可撤销！'))return;try{await fetch('/api/history/clear',{method:'POST'});state.history=[];renderHistory();toast('success','所有记录已清空')}catch(e){toast('error','清空失败')}}
function closeAllModals(){closeModelModal();closeConfigModal()}

// ── Toast / Loading ─────────────────────────────────────────────────────

document.getElementById('modelModal').addEventListener('click',function(e){if(e.target===this)closeModelModal()});
document.getElementById('configModal').addEventListener('click',function(e){if(e.target===this)closeConfigModal()});
