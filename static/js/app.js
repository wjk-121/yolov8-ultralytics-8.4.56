/** YOLOv8 Web App v5 — 优化进度条/历史刷新/批量混搭 */
const state={mode:'image',config:null,model:null,history:[],_latestMtime:null,_latestCount:0,detecting:false,cameraStream:null,cameraActive:false,cameraTimer:null,cameraFpsTimer:null,cameraFrameCount:0,_lastCameraDets:[],_lastCameraB64:'',cameraSessionId:null,cameraSnapshotCount:0,_batchMediaItems:[],_batchMediaIndex:0,_previewMode:false,_queueDismissed:false,_queueJobId:null,_queueJobType:null,_queueSzWarning:null};

// ── 文件类型验证配置 ─────────────────────────────────────────────────────
const FILE_VALIDATION={
    image:{exts:['.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'],label:'图片',warnMB:20},
    video:{exts:['.mp4','.avi','.mov','.mkv','.webm','.wmv','.flv'],label:'视频',warnMB:100},
    batch:{exts:['.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif','.mp4','.avi','.mov','.mkv','.webm','.wmv','.flv'],label:'图片/视频',warnMB:100}
};
function getFileValidation(mode){
    if(mode==='image')return FILE_VALIDATION.image;
    if(mode==='video')return FILE_VALIDATION.video;
    if(mode==='batch')return FILE_VALIDATION.batch;
    return null;
}
function validateFileClient(file,mode){
    /** 客户端文件验证, 返回 {valid:bool, error:string|null, warn:string|null} */
    var v=getFileValidation(mode);
    if(!v)return {valid:true,error:null,warn:null};
    if(!file||!file.name)return {valid:false,error:'文件名为空',warn:null};
    var dot=file.name.lastIndexOf('.'),ext=(dot>=0?file.name.substring(dot):'').toLowerCase();
    if(!ext)return {valid:false,error:'"'+file.name+'" 没有扩展名, 无法识别文件类型',warn:null};
    if(v.exts.indexOf(ext)===-1)return {valid:false,error:'"'+file.name+'" 类型不支持 ('+ext+'), 请上传 '+v.label+' 文件',warn:null};
    if(file.size===0)return {valid:false,error:'"'+file.name+'" 是空文件',warn:null};
    // 大文件提示 (不拒绝, 仅警告 — 具体耗时会由遮罩内 updateSizeWarning 实时展示)
    var sizeMB=file.size/(1024*1024),warn=null;
    if(sizeMB>v.warnMB){
        warn='"'+file.name+'" 文件较大 ('+sizeMB.toFixed(1)+'MB), 检测可能需要较长时间, 请耐心等候';
    }
    // 双扩展名安全检测
    var parts=file.name.split('.'),danger=['exe','dll','bat','cmd','ps1','sh','vbs','js','py','php','com','msi','scr'];
    if(parts.length>=3){var snd=parts[parts.length-2].toLowerCase();if(danger.indexOf(snd)!==-1)return {valid:false,error:'"'+file.name+'" 包含可疑的双扩展名, 已被拒绝',warn:null}}
    return {valid:true,error:null,warn:warn};
}

/**
 * 根据文件大小生成大文件基础信息 (用于实时动态更新遮罩警告)
 * @param {File|File[]} files - 单个文件或文件数组
 * @param {string} mode - 检测模式
 * @returns {{fileName:string, sizeMB:number, label:string}|null} 基础信息, 无需提示时返回 null
 */
function getSizeWarningForLoading(files,mode){
    var arr=Array.isArray(files)?files:[files];
    if(!arr.length)return null;
    var largest=arr[0];
    for(var i=1;i<arr.length;i++){if(arr[i].size>largest.size)largest=arr[i]}
    var mb=largest.size/(1024*1024);
    var v=getFileValidation(mode);
    if(!v||mb<=v.warnMB)return null;
    var label=arr.length>1?'最大文件: ':'';
    return {fileName:largest.name, sizeMB:mb, label:label};
}

/**
 * 根据实时进度更新遮罩内的大文件警告文案
 * @param {{fileName:string, sizeMB:number, label:string}} base - getSizeWarningForLoading 的返回值
 * @param {number} elapsed - 已耗时 (秒)
 * @param {number|null} eta - 预计剩余 (秒), null 表示暂无估算
 */
function updateSizeWarning(base, elapsed, eta){
    var warnEl=document.getElementById('loadingWarning');
    if(!warnEl||!base)return;
    var msg=base.label+'"'+base.fileName+'" 文件较大 ('+base.sizeMB.toFixed(1)+'MB)';
    msg+=' | 已耗时 '+formatETA(elapsed);
    if(eta!==null&&eta>0)msg+=' | 预计剩余 '+formatETA(eta);
    else msg+=' | 正在处理中...';
    warnEl.textContent='⚠ '+msg;
    warnEl.style.display='';
}

// ── 客户端标识 ──────────────────────────────────────────────────────────
function getClientId(){
    try{
        var id=localStorage.getItem('yolo_client_id');
        if(!id){id='c_'+Math.random().toString(36).substring(2,10);localStorage.setItem('yolo_client_id',id)}
        return id;
    }catch(e){return 'c_'+Math.random().toString(36).substring(2,10)}
}

// ── 活跃任务持久化 ─────────────────────────────────────────────────────
function saveActiveJob(jobId,type,filename,sizeMB){
    try{localStorage.setItem('activeJob',JSON.stringify({jobId:jobId,type:type,filename:filename,sizeMB:sizeMB||0,savedAt:Date.now()}))}catch(e){}
}
function getActiveJob(){
    try{var raw=localStorage.getItem('activeJob');if(!raw)return null;var job=JSON.parse(raw);if(Date.now()-job.savedAt>86400000){localStorage.removeItem('activeJob');return null}return job}catch(e){return null}
}
function clearActiveJob(){try{localStorage.removeItem('activeJob')}catch(e){}}

// ── 遮罩退出/恢复 (排队 & 处理中都可退出) ──────────────────────────────
function dismissOverlay(){
    // 隐藏全屏遮罩，显示底部迷你栏
    document.getElementById('loading').classList.remove('active');
    var mini=document.getElementById('queueMini');
    if(mini)mini.style.display='flex';
    state._queueDismissed=true;
}

function restoreOverlay(){
    // 恢复全屏遮罩
    var mini=document.getElementById('queueMini');
    if(mini)mini.style.display='none';
    document.getElementById('loading').classList.add('active');
    state._queueDismissed=false;
}

function updateQueueMini(status,pos,estWait,progress){
    /** 更新底部迷你栏内容 */
    var mini=document.getElementById('queueMini'),txt=document.getElementById('queueMiniText');
    if(!mini||!txt)return;
    mini.style.display='flex';
    if(status==='queued'){
        var posStr=pos?('前方 '+pos):'...';
        var waitStr=estWait?(' · 预计约 '+formatETA(estWait)):'';
        txt.innerHTML='<span class="qmini-icon">⏳</span> 排队中 · '+posStr+waitStr;
        mini.className='queue-mini queue-mini-queued';
    }else if(status==='processing'){
        var pctStr=progress!==undefined?(' · '+Math.round(progress)+'%'):'';
        txt.innerHTML='<span class="qmini-icon">📊</span> 处理中'+pctStr;
        mini.className='queue-mini queue-mini-processing';
    }
    // 点击迷你栏恢复遮罩
    mini.onclick=function(){restoreOverlay()};
}

function hideQueueMini(){
    var mini=document.getElementById('queueMini');
    if(mini)mini.style.display='none';
}

document.addEventListener('DOMContentLoaded',()=>{init();setupUpload();setupKeyboard()});

async function init(){
    try{const r=await fetch('/api/status');const d=await r.json();if(d.success){state.model=d.data;state.config=d.data.config;updateModelDisplay();updateStatus();updateSliders()}}catch(e){toast('error','初始化失败: '+e.message)}
    loadHistory();
    resumeActiveJob();
    // 轻量轮询历史更新 (每10秒检查一次)
    setInterval(checkHistoryUpdate,10000);
}

async function checkHistoryUpdate(){
    try{const r=await fetch('/api/history/check');const d=await r.json();if(d.success){
        var count=d.data.total||0;
        if(state._latestCount!==count){await loadHistory()}
    }}catch(e){/* 静默失败 */}
}

async function resumeActiveJob(){
    /** 页面加载时自动恢复活跃任务 */
    // 1. 先尝试 localStorage
    var saved=getActiveJob();
    var jobId=null,jobType=null;
    if(saved){
        // 验证任务是否还在
        try{
            var r=await fetch('/api/detect/'+saved.type+'/status/'+saved.jobId);
            var d=await r.json();
            if(d.success&&d.data&&(d.data.status==='queued'||d.data.status==='processing')){
                jobId=saved.jobId;jobType=saved.type;
            }else{clearActiveJob()}
        }catch(e){clearActiveJob()}
    }
    // 2. localStorage 无记录时查服务端
    if(!jobId){
        try{
            var r2=await fetch('/api/jobs/active?client_id='+getClientId());
            var d2=await r2.json();
            if(d2.success&&d2.data.jobs.length>0){
                var j=d2.data.jobs[0];  // 取最新
                jobId=j.job_id;jobType=j.type;
            }
        }catch(e){}
    }
    if(!jobId)return;

    // 恢复成功 — 显示遮罩并开始轮询
    state.detecting=true;
    hideAllInputs();
    document.getElementById('statsRow').style.display='none';
    document.getElementById('detailTable').style.display='none';
    document.getElementById('resultArea').style.display='none';
    state._queueDismissed=false;
    state._queueJobId=jobId;
    state._queueJobType=jobType;

    if(jobType==='video'){
        showLoadingProgress('正在恢复视频检测...', false);
        startVideoPolling(jobId,null);
    }else{
        showLoadingProgress('正在恢复批量检测...', false);
        startBatchPolling(jobId,null);
    }
}

function setupUpload(){
    const b=document.getElementById('uploadBox'),i=document.getElementById('fileInput');
    b.addEventListener('click',e=>{if(e.target.closest('.btn-secondary')||e.target.closest('.btn-primary'))return;i.click()});
    document.getElementById('selectBtn').addEventListener('click',e=>{e.stopPropagation();i.click()});
    i.addEventListener('change',e=>{
        const files=e.target.files;
        if(!files||!files.length)return;
        // 客户端文件类型预检
        var valid=[],rejected=[],warnings=[];
        for(var j=0;j<files.length;j++){
            var vc=validateFileClient(files[j],state.mode);
            if(vc.valid){valid.push(files[j]);if(vc.warn)warnings.push(vc.warn)}
            else rejected.push({name:files[j].name,error:vc.error});
        }
        // 大文件提示 (不拒绝, 仅 toast)
        warnings.forEach(function(w){toast('warning','⚠ '+w)});
        if(rejected.length>0){
            rejected.forEach(function(r){toast('error','✗ '+r.error)});
            var warnMsg=rejected.length+' 个文件不符合要求: '+rejected.map(function(r){return r.name}).join(', ');
            if(warnMsg.length>120)warnMsg=warnMsg.substring(0,117)+'...';
            if(rejected.length===1)warnMsg=rejected[0].error;
            var warnEl=document.getElementById('fileSizeWarning'),warnTxt=document.getElementById('fileSizeWarningText');
            warnTxt.textContent=warnMsg;warnEl.style.display='flex';
        }
        if(!valid.length){i.value='';return}
        if(state.mode==='batch'){
            detectBatch(valid);
        }else{
            checkFileSize(valid[0]);handleFile(valid[0]);
        }
    });
    b.addEventListener('dragover',e=>{e.preventDefault();b.classList.add('dragover')});
    b.addEventListener('dragleave',()=>{b.classList.remove('dragover')});
    b.addEventListener('drop',e=>{
        e.preventDefault();b.classList.remove('dragover');
        const files=e.dataTransfer.files;
        if(!files||!files.length)return;
        // 客户端文件类型预检
        var valid=[],rejected=[],warnings=[];
        for(var j=0;j<files.length;j++){
            var vc=validateFileClient(files[j],state.mode);
            if(vc.valid){valid.push(files[j]);if(vc.warn)warnings.push(vc.warn)}
            else rejected.push({name:files[j].name,error:vc.error});
        }
        // 大文件提示 (不拒绝, 仅 toast)
        warnings.forEach(function(w){toast('warning','⚠ '+w)});
        if(rejected.length>0){
            rejected.forEach(function(r){toast('error','✗ '+r.error)});
            var warnMsg=rejected.length+' 个文件不符合要求';
            var warnEl=document.getElementById('fileSizeWarning'),warnTxt=document.getElementById('fileSizeWarningText');
            warnTxt.textContent=warnMsg;warnEl.style.display='flex';
        }
        if(!valid.length)return;
        if(state.mode==='batch'){detectBatch(valid)}
        else{checkFileSize(valid[0]);handleFile(valid[0])}
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
function detectSample(name){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}state.detecting=true;fetch('/api/samples/check/'+name).then(function(r){return r.json()}).then(function(check){if(check.data&&check.data.exists){showLoading('正在检测示例图片...')}else{showLoadingProgress('图片正在下载，请稍后',true)}return fetch('/api/detect/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,confidence:state.config.confidence,client_id:getClientId()})})}).then(function(r){return r.json()}).then(function(d){if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}).catch(function(e){toast('error','示例检测失败: '+e.message)}).finally(function(){hideLoading();state.detecting=false})}
function setUrl(u){document.getElementById('urlInput').value=u}

// ── 文件大小检查 ─────────────────────────────────────────────────────────
async function checkFileSize(f){
    const warnEl=document.getElementById('fileSizeWarning'),warnText=document.getElementById('fileSizeWarningText');
    warnEl.style.display='none';
    try{const r=await fetch('/api/check-file-size',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({size_bytes:f.size})});const d=await r.json();if(d.success&&d.data.warning){warnText.textContent=d.data.warning;warnEl.style.display='flex'}}catch(e){}
}

// ── 图片检测 ──────────────────────────────────────────────────────────
async function detectImage(f){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}state.detecting=true;var szImg=getSizeWarningForLoading(f,'image');showLoadingProgress('正在检测图片...', true, szImg?szImg.label+'"'+szImg.fileName+'" 文件较大 ('+szImg.sizeMB.toFixed(1)+'MB), 请耐心等候...':null);try{const fd=new FormData();fd.append('file',f);fd.append('confidence',state.config.confidence);fd.append('client_id',getClientId());const r=await fetch('/api/detect/image',{method:'POST',body:fd});const d=await r.json();if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}catch(e){toast('error','检测失败: '+e.message)}finally{hideLoading();state.detecting=false}}

// ── 批量轮询 (提取为独立函数, 供初始检测 + 恢复共用) ──────────────
function startBatchPolling(jobId, szBatch){
    var bar=document.getElementById('loadingProgressBar'),pctEl=document.getElementById('loadingProgressPct'),statusEl=document.getElementById('loadingProgressStatus'),liveEl=document.getElementById('loadingProgressLive');
    var pollInterval=setInterval(function(){
        fetch('/api/detect/batch/status/'+jobId).then(function(r){return r.json()}).then(function(pd){
            if(!pd.success)return;
            var j=pd.data;

            // 排队状态
            if(j.status==='queued'){
                if(!state._queueDismissed){
                    var qPos=j.queue_position||0;
                    var estWait=j.estimated_wait||0;
                    document.getElementById('loadingSpinner').style.display='';
                    document.getElementById('loadingProgressWrap').style.display='none';
                    document.getElementById('loadingText').textContent='排队中 · 前方 '+qPos+' 个任务';
                    var wEl2=document.getElementById('loadingWarning');
                    if(wEl2&&estWait>0){wEl2.textContent='⚠ 预计等待约 '+formatETA(estWait);wEl2.style.display=''}
                    document.getElementById('loadingDismissBtn').style.display='';
                }
                updateQueueMini('queued', j.queue_position, j.estimated_wait, null);
                return;
            }

            // 处理中 — 遮罩未收起
            if(j.status==='processing'&&!state._queueDismissed){
                document.getElementById('loadingSpinner').style.display='none';
                document.getElementById('loadingProgressWrap').style.display='';
                document.getElementById('loadingDismissBtn').style.display='';
                var fileLabel='第 '+(j.current_file_index+1)+'/'+j.total_files+' 个文件';
                var status=fileLabel+': '+j.current_file_name;
                if(j.current_file_type==='image'){
                    bar.classList.add('indeterminate');bar.style.width='';pctEl.textContent='';
                    statusEl.textContent=status;
                    liveEl.innerHTML='目标累计: <b>'+j.total_detections+'</b> | 耗时: '+j.elapsed+'s';
                    updateSizeWarning(szBatch,j.elapsed,j.eta);
                }else{
                    bar.classList.remove('indeterminate');
                    var fps=j.current_file_frame/Math.max(j.elapsed||0.001,0.001);
                    var live='目标累计: <b>'+j.total_detections+'</b> | 帧: '+j.current_file_frame+'/'+j.current_file_total_frames+' | FPS: '+Math.round(fps)+' | 耗时: '+j.elapsed+'s';
                    if(j.eta)status+=' | 预计剩余 '+formatETA(j.eta);
                    updateLoadingProgress(j.progress,status,live);
                    updateSizeWarning(szBatch,j.elapsed,j.eta);
                }
            }

            // 处理中 — 遮罩已收起
            if(j.status==='processing'&&state._queueDismissed){
                updateQueueMini('processing', null, null, j.progress);
            }

            if(j.status==='complete'){
                clearInterval(pollInterval);pollInterval=null;
                clearActiveJob();hideQueueMini();
                if(!state._queueDismissed){
                    updateLoadingProgress(100,'✓ 批量检测完成 — '+j.total_files+' 个文件, '+j.total_detections+' 目标, '+j.elapsed+'s','');
                    hideLoading();
                }else{
                    hideLoading();document.getElementById('loading').classList.remove('active');
                }
                toast('success','批量检测完成: '+j.total_detections+' 个目标');
                showBatchResult({total_images:j.total_images,total_videos:j.total_videos,total_detections:j.total_detections,elapsed_time:j.elapsed,results:j.results,run_name:j.run_name});
                state.detecting=false;loadHistory();
            }else if(j.status==='error'){
                clearInterval(pollInterval);pollInterval=null;
                clearActiveJob();hideQueueMini();
                hideLoading();document.getElementById('loading').classList.remove('active');
                toast('error','批量检测失败: '+(j.error||'未知错误'));
                state.detecting=false;
            }
        }).catch(function(){/* 轮询网络错误-继续 */})
    },300);
    return pollInterval;
}

function detectBatch(fs){
    if(state.detecting)return;
    if(!state.config){toast('warning','正在初始化，请稍后再试');return}
    state.detecting=true;
    state._queueDismissed=false;
    state._queueJobId=null;
    state._queueJobType='batch';
    state._queueSzWarning=getSizeWarningForLoading(fs,'batch');
    var szBatch=state._queueSzWarning;
    var hasVideos=fs.some(function(f){var n=f.name||'';return /\.(mp4|avi|mov|mkv|webm|wmv|flv)$/i.test(n)});
    var sizeWarnB=szBatch?szBatch.label+'"'+szBatch.fileName+'" 文件较大 ('+szBatch.sizeMB.toFixed(1)+'MB), 请耐心等候...':null;
    try{
        var fd=new FormData();fs.forEach(function(f){fd.append('files',f)});
        fd.append('confidence',state.config.confidence);fd.append('client_id',getClientId());
        fetch('/api/detect/batch',{method:'POST',body:fd}).then(function(r){return r.json()}).then(function(d){
            if(!d.success){
                var errDetail=d.error||'未知错误';
                if(d.data&&d.data.rejected&&d.data.rejected.length>0){errDetail+='\n被拒绝: '+d.data.rejected.map(function(rr){return rr.filename}).join(', ')}
                toast('error',errDetail);hideLoading();state.detecting=false;return;
            }
            var jobId=d.data.job_id;
            state._queueJobId=jobId;
            saveActiveJob(jobId,'batch',fs.length+' 个文件',szBatch?szBatch.sizeMB:0);
            if(d.data.total_rejected>0){
                var rjNames=d.data.rejected.map(function(rr){return rr.filename}).join(', ');
                setTimeout(function(){toast('warning','⚠ '+d.data.total_rejected+' 个文件被拒绝: '+rjNames)},500);
            }
            hideAllInputs();
            document.getElementById('statsRow').style.display='none';
            document.getElementById('detailTable').style.display='none';
            document.getElementById('resultArea').style.display='none';

            var isQueued=d.data.status==='queued';
            if(isQueued){
                showLoading('排队中...');
                var qPos=d.data.queue_position||0;
                var estWait=d.data.estimated_wait||0;
                document.getElementById('loadingText').textContent='排队中 · 前方 '+qPos+' 个任务';
                if(estWait>0){var wEl3=document.getElementById('loadingWarning');if(wEl3){wEl3.textContent='⚠ 预计等待约 '+formatETA(estWait);wEl3.style.display=''}}
                document.getElementById('loadingSpinner').style.display='';
                document.getElementById('loadingProgressWrap').style.display='none';
            }else{
                showLoadingProgress('正在批量检测 '+fs.length+' 个文件...', !hasVideos, sizeWarnB);
            }
            document.getElementById('loadingDismissBtn').style.display='';

            startBatchPolling(jobId, szBatch);
        }).catch(function(e){toast('error','批量检测失败: '+e.message);hideLoading();state.detecting=false})
    }catch(e){toast('error','批量检测失败: '+e.message);hideLoading();state.detecting=false}

async function detectUrl(u){if(state.detecting)return;if(!state.config){toast('warning','正在初始化，请稍后再试');return}u=u||document.getElementById('urlInput').value.trim();if(!u){toast('warning','请输入URL');return}state.detecting=true;showLoadingProgress('正在下载并检测...', true);try{const r=await fetch('/api/detect/url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u,confidence:state.config.confidence,client_id:getClientId()})});const d=await r.json();if(d.success){showImageResult(d.data);toast('success',d.message);loadHistory()}else toast('error',d.error)}catch(e){toast('error','URL检测失败: '+e.message)}finally{hideLoading();state.detecting=false}}

// ── 显示图片结果 ─────────────────────────────────────────────────────
function showImageResult(d){hideAllInputs();const ra=document.getElementById('resultArea');ra.style.display='block';document.getElementById('imageBox').innerHTML='<img id="resultImage" src="" alt="">';const img=document.getElementById('resultImage');img.style.display='block';img.classList.add('loading-img');img.src=d.result_image;img.onload=()=>{img.classList.remove('loading-img')};img.onerror=()=>{img.classList.remove('loading-img');toast('error','标注图片加载失败')};showStats(d);if(d.detections&&d.detections.length>0)showDetailTable(d.detections);else{document.getElementById('detailTable').style.display='none'}}

// ── 视频检测 (轮询 + 优化进度) ───────────────────────────────────────
function formatETA(secs){if(!secs||secs<=0)return'计算中...';if(secs<60)return Math.round(secs)+'秒';if(secs<3600)return Math.floor(secs/60)+'分'+Math.round(secs%60)+'秒';return Math.floor(secs/3600)+'时'+Math.floor((secs%3600)/60)+'分'}

// ── 视频轮询 (提取为独立函数, 供初始检测 + 恢复共用) ──────────────
function startVideoPolling(jobId, szVid){
    var pollInterval=setInterval(async()=>{
        try{
            var r=await fetch('/api/detect/video/status/'+jobId);
            var d=await r.json();
            if(!d.success)return;
            var j=d.data;

            // 排队状态
            if(j.status==='queued'){
                if(!state._queueDismissed){
                    var qPos=j.queue_position||0;
                    var estWait=j.estimated_wait||0;
                    document.getElementById('loadingSpinner').style.display='';
                    document.getElementById('loadingProgressWrap').style.display='none';
                    document.getElementById('loadingText').textContent='排队中 · 前方 '+qPos+' 个任务';
                    var warnEl=document.getElementById('loadingWarning');
                    if(warnEl&&estWait>0){warnEl.textContent='⚠ 预计等待约 '+formatETA(estWait);warnEl.style.display=''}
                    document.getElementById('loadingDismissBtn').style.display='';
                }
                updateQueueMini('queued', j.queue_position, j.estimated_wait, null);
                return;
            }

            // 首次从排队切换为处理中
            if(j.status==='processing'&&state._queueDismissed){
                updateQueueMini('processing', null, null, j.progress);
            }

            // 处理中 — 遮罩未收起
            if(j.status==='processing'&&!state._queueDismissed){
                document.getElementById('loadingSpinner').style.display='none';
                document.getElementById('loadingProgressWrap').style.display='';
                document.getElementById('loadingDismissBtn').style.display='';
                var fps=Math.round(j.frame/Math.max(j.elapsed,0.001));
                var status='检测中... '+j.frame+'/'+j.total_frames+' 帧';
                if(j.eta)status+=' | 预计剩余 '+formatETA(j.eta);
                var live='目标累计: <b>'+j.total_detections+'</b> | FPS: '+fps+' | 耗时: '+j.elapsed+'s';
                updateLoadingProgress(j.progress, status, live);
                updateSizeWarning(szVid, j.elapsed, j.eta);
            }

            // 处理中 — 遮罩已收起
            if(j.status==='processing'&&state._queueDismissed){
                updateQueueMini('processing', null, null, j.progress);
            }

            if(j.status==='complete'){
                clearInterval(pollInterval);pollInterval=null;
                clearActiveJob();hideQueueMini();
                if(!state._queueDismissed){
                    updateLoadingProgress(100, '✓ 检测完成 — '+j.total_frames+' 帧, '+j.total_detections+' 目标, '+j.elapsed+'s', '目标累计: <b style="color:var(--success)">'+j.total_detections+'</b> | FPS: '+j.fps+' | 耗时: '+j.elapsed+'s');
                    hideLoading();
                }else{
                    hideLoading();document.getElementById('loading').classList.remove('active');
                }
                showStats({total_detections:j.total_detections, elapsed_time:j.elapsed, fps:j.fps});
                if(j.result_video){showVideoResult(j.result_video,j.playable)}
                toast('success','视频检测完成');
                state.detecting=false;loadHistory();
            }else if(j.status==='error'){
                clearInterval(pollInterval);pollInterval=null;
                clearActiveJob();hideQueueMini();
                hideLoading();document.getElementById('loading').classList.remove('active');
                toast('error','视频检测失败: '+(j.error||'未知错误'));
                state.detecting=false;
            }
        }catch(e){/* 轮询网络错误-继续 */}
    },300);
    return pollInterval;
}

async function detectVideo(f){
    if(state.detecting)return;
    if(!state.config){toast('warning','正在初始化，请稍后再试');return}
    state.detecting=true;
    state._queueDismissed=false;
    state._queueJobId=null;
    state._queueJobType='video';
    state._queueSzWarning=getSizeWarningForLoading(f,'video');
    try{
        var szVid=state._queueSzWarning;
        var sizeWarnV=szVid?szVid.label+'"'+szVid.fileName+'" 文件较大 ('+szVid.sizeMB.toFixed(1)+'MB), 请耐心等候...':null;
        showLoading('正在上传视频...', sizeWarnV);
        var fd=new FormData();fd.append('file',f);fd.append('confidence',state.config.confidence);fd.append('client_id',getClientId());
        var uR=await fetch('/api/detect/video',{method:'POST',body:fd});
        var uD=await uR.json();
        if(!uD.success){toast('error',uD.error);hideLoading();state.detecting=false;return}

        var jobId=uD.data.job_id;
        state._queueJobId=jobId;
        saveActiveJob(jobId,'video',f.name,szVid?szVid.sizeMB:0);
        hideAllInputs();
        document.getElementById('statsRow').style.display='none';
        document.getElementById('detailTable').style.display='none';
        document.getElementById('resultArea').style.display='none';

        // 初始状态: 排队或处理中
        var isQueued=uD.data.status==='queued';
        if(isQueued){
            showLoading('排队中...');
            var qPos=uD.data.queue_position||0;
            var estWait=uD.data.estimated_wait||0;
            document.getElementById('loadingText').textContent='排队中 · 前方 '+qPos+' 个任务';
            if(estWait>0){var wEl=document.getElementById('loadingWarning');if(wEl){wEl.textContent='⚠ 预计等待约 '+formatETA(estWait);wEl.style.display=''}}
            document.getElementById('loadingSpinner').style.display='';
            document.getElementById('loadingProgressWrap').style.display='none';
        }else{
            showLoadingProgress('正在分析视频...', false, sizeWarnV);
        }
        document.getElementById('loadingDismissBtn').style.display='';

        startVideoPolling(jobId, szVid);
    }catch(e){toast('error','视频检测失败: '+e.message);hideLoading();state.detecting=false}
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
    try{const r=await fetch('/api/detect/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:b64,confidence:state.config.confidence,client_id:getClientId()})});const d=await r.json();if(d.success){drawCameraResult(d.data);const td=d.data.total_detections;document.getElementById('cameraDets').textContent=td+' 目标';document.getElementById('cameraDets').style.color=td>0?'#34C759':'var(--text2)';state._lastCameraDets=d.data.detections;state._lastCameraB64=b64}}catch(e){console.error(e)}finally{state.detecting=false}
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
    try{const r=await fetch('/api/camera/snapshot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:state._lastCameraB64,detections:state._lastCameraDets||[],session_id:state.cameraSessionId,client_id:getClientId()})});const d=await r.json();if(d.success){state.cameraSessionId=d.data.session_id;state.cameraSnapshotCount=d.data.total_snapshots;document.getElementById('cameraSnaps').textContent=d.data.total_snapshots+' 抓拍';toast('success',d.message)}else toast('error',d.error)}catch(e){toast('error','抓拍失败')}
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
        // 时间戳格式化: "2025-05-31 14:23:15" → "05-31 14:23"
        var ts='';if(it.timestamp){var t=it.timestamp;var sp=t.indexOf(' ');ts=sp>0?t.substring(5,sp):t.substring(0,10)}
        // 设备标签
        var cid=it.client_id||'';var myCid=getClientId();
        var devTag='',devCls='';
        if(cid===myCid){devTag='本设备';devCls=' history-device-own'}
        else if(cid){devTag='设备 '+cid.substring(2,6);devCls=' history-device-other'}
        el.innerHTML='<div class="history-icon">'+ic+'</div><div class="history-info"><div class="history-type">'+it.type+'<span class="history-timestamp">'+ts+'</span><span class="history-device-tag'+devCls+'">'+devTag+'</span></div><div class="history-src">'+s+'</div></div><div class="history-meta"><div class="history-dets">'+it.detections+'</div><div class="history-time">'+it.elapsed_time+'s</div></div>';
        el.addEventListener('click',()=>{window.location.href='/history/'+it.id});l.appendChild(el);
    })
}

async function clearHistory(){if(!confirm('确定清空所有检测记录和结果文件？此操作不可撤销！'))return;try{await fetch('/api/history/clear',{method:'POST'});state.history=[];renderHistory();toast('success','所有记录已清空')}catch(e){toast('error','清空失败')}}
function closeAllModals(){closeModelModal();closeConfigModal()}

// ── Toast / Loading ─────────────────────────────────────────────────────

document.getElementById('modelModal').addEventListener('click',function(e){if(e.target===this)closeModelModal()});
document.getElementById('configModal').addEventListener('click',function(e){if(e.target===this)closeConfigModal()});
