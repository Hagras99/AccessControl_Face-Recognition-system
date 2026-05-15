/* ═══════════════════════════════════════════════════════════
   FaceGuard — Main Application Script
   Modules: TabManager, WebcamManager, FaceGuide,
            EnrollmentFlow, VerificationFlow, EvaluationFlow
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

// ── POSE DEFINITIONS ─────────────────────────────────────
const POSES = [
    { name: 'Front',      desc: 'Look straight at the camera',         transform: 'rotateY(0deg) rotateX(0deg) rotateZ(0deg)', arrow: null, mouth: 'M 80 145 Q 100 155, 120 145', eyeScale: 1 },
    { name: 'Left',       desc: 'Slowly turn your head to the LEFT',   transform: 'rotateY(30deg)',  arrow: { rotation: '90deg', pos: 'left' }, mouth: null, eyeScale: 1 },
    { name: 'Right',      desc: 'Slowly turn your head to the RIGHT',  transform: 'rotateY(-30deg)', arrow: { rotation: '-90deg', pos: 'right' }, mouth: null, eyeScale: 1 },
    { name: 'Up',         desc: 'Tilt your head slightly UP',          transform: 'rotateX(-20deg)', arrow: { rotation: '0deg', pos: 'top' }, mouth: null, eyeScale: 1 },
    { name: 'Down',       desc: 'Tilt your head slightly DOWN',        transform: 'rotateX(20deg)',  arrow: { rotation: '180deg', pos: 'bottom' }, mouth: null, eyeScale: 1 },
    { name: 'Tilt Left',  desc: 'Tilt head toward LEFT shoulder',      transform: 'rotateZ(15deg)',  arrow: { rotation: '90deg', pos: 'left' }, mouth: null, eyeScale: 1 },
    { name: 'Tilt Right', desc: 'Tilt head toward RIGHT shoulder',     transform: 'rotateZ(-15deg)', arrow: { rotation: '-90deg', pos: 'right' }, mouth: null, eyeScale: 1 },
    { name: 'Smile',      desc: 'Give a natural SMILE 😊',             transform: 'rotateY(0deg)',   arrow: null, mouth: 'M 78 142 Q 100 165, 122 142', eyeScale: 0.85 },
    { name: 'Eyes Closed', desc: 'Gently CLOSE your eyes',             transform: 'rotateY(0deg)',   arrow: null, mouth: null, eyeScale: 0.15 },
];

// ── REFS ─────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const navTabs = document.querySelectorAll('.nav-tab');
const tabContents = { enroll: $('enrollTab'), login: $('loginTab'), evaluation: $('evaluationTab') };
const tabIndicator = $('tabIndicator');

// Enroll refs
const enrollName = $('enrollName'), enrollVideo = $('enrollVideo'), enrollCanvas = $('enrollCanvas');
const startEnrollBtn = $('startEnrollBtn'), captureBtn = $('captureBtn'), resetEnrollBtn = $('resetEnrollBtn');
const enrollPoseLabel = $('enrollPoseLabel'), enrollPlaceholder = $('enrollPlaceholder');
const enrollFaceBox = $('enrollFaceBox'), enrollStatus = $('enrollStatus');
const enrollStatusMsg = $('enrollStatusMsg'), enrollStatusIcon = $('enrollStatusIcon');
const headSvg = $('headSvg'), directionArrow = $('directionArrow');
const poseTitle = $('poseTitle'), poseDesc = $('poseDescription');
const thumbGrid = $('thumbGrid'), poseProgress = $('poseProgress');
const nameInputGroup = $('nameInputGroup');

// Login refs
const loginVideo = $('loginVideo'), loginCanvas = $('loginCanvas');
const startLoginBtn = $('startLoginBtn'), verifyBtn = $('verifyBtn');
const loginPlaceholder = $('loginPlaceholder'), scanLine = $('scanLine');
const dashboardSection = $('dashboardSection'), accessCard = $('accessCard');
const accessIcon = $('accessIcon'), accessLabel = $('accessLabel');
const profileName = $('profileName'), profileId = $('profileId');
const profileThumb = $('profileThumb'), confidenceFill = $('confidenceFill');
const confidenceValue = $('confidenceValue'), enrolledDate = $('enrolledDate');
const profileSource = $('profileSource'), profileImageCount = $('profileImageCount');
const loginAgainBtn = $('loginAgainBtn'), loginStatusBadge = $('loginStatusBadge');
const loginWebcamContainer = $('loginWebcamContainer');

// Eval refs
const startEvalBtn = $('startEvalBtn'), evalLoading = $('evalLoading');
const evalResults = $('evalResults'), pcaStats = $('pcaStats'), lbpStats = $('lbpStats');
const resultsPlot = $('resultsPlot'), plotPlaceholder = $('plotPlaceholder');
const evalLoader = $('evalLoader');

// ── STATE ────────────────────────────────────────────────
let enrollStream = null, loginStream = null;
let currentPose = 0, capturedImages = [], detectInterval = null;
let faceDetectedState = false;

// ══════════════════════════════════════════════════════════
// TAB MANAGER
// ══════════════════════════════════════════════════════════
function initTabs() {
    updateIndicator(document.querySelector('.nav-tab.active'));
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            Object.values(tabContents).forEach(c => c.classList.remove('active'));
            tabContents[tab.dataset.tab].classList.add('active');
            updateIndicator(tab);
        });
    });
}

function updateIndicator(tab) {
    if (!tab) return;
    tabIndicator.style.width = tab.offsetWidth + 'px';
    tabIndicator.style.left = tab.offsetLeft + 'px';
}

// ══════════════════════════════════════════════════════════
// WEBCAM MANAGER
// ══════════════════════════════════════════════════════════
async function startWebcam(videoEl) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        videoEl.srcObject = stream;
        return stream;
    } catch (e) {
        alert('Camera access denied. Please allow camera permissions.');
        return null;
    }
}

function stopWebcam(stream, videoEl) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (videoEl) videoEl.srcObject = null;
}

function captureFrame(videoEl, canvasEl) {
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    const ctx = canvasEl.getContext('2d');
    ctx.drawImage(videoEl, 0, 0);
    return canvasEl.toDataURL('image/jpeg', 0.85);
}

// ══════════════════════════════════════════════════════════
// FACE GUIDE (3D wireframe head)
// ══════════════════════════════════════════════════════════
function updateFaceGuide(poseIdx) {
    const pose = POSES[poseIdx];
    headSvg.style.transform = pose.transform;
    poseTitle.textContent = pose.name;
    poseDesc.textContent = pose.desc;

    // Arrow
    if (pose.arrow) {
        directionArrow.classList.remove('hidden');
        const arrowSvg = directionArrow.querySelector('svg');
        arrowSvg.style.transform = `rotate(${pose.arrow.rotation})`;
        const positions = {
            left:   { top: '50%', left: '-10px', right: 'auto', bottom: 'auto', transform: 'translateY(-50%)' },
            right:  { top: '50%', right: '-10px', left: 'auto', bottom: 'auto', transform: 'translateY(-50%)' },
            top:    { top: '-10px', left: '50%', right: 'auto', bottom: 'auto', transform: 'translateX(-50%)' },
            bottom: { bottom: '-10px', left: '50%', right: 'auto', top: 'auto', transform: 'translateX(-50%)' },
        };
        Object.assign(directionArrow.style, positions[pose.arrow.pos]);
    } else {
        directionArrow.classList.add('hidden');
    }

    // Mouth (smile)
    const mouth = document.getElementById('mouth');
    if (pose.mouth) mouth.setAttribute('d', pose.mouth);
    else mouth.setAttribute('d', 'M 80 145 Q 100 155, 120 145');

    // Eyes (closed)
    const leftEye = $('leftEye'), rightEye = $('rightEye');
    leftEye.style.transform = `scaleY(${pose.eyeScale})`;
    rightEye.style.transform = `scaleY(${pose.eyeScale})`;
}

// ══════════════════════════════════════════════════════════
// FACE DETECTION (periodic)
// ══════════════════════════════════════════════════════════
function startFaceDetection(videoEl, canvasEl, faceBox, onDetectionChange) {
    if (detectInterval) clearInterval(detectInterval);
    detectInterval = setInterval(async () => {
        if (!videoEl.srcObject) return;
        try {
            const frame = captureFrame(videoEl, canvasEl);
            const resp = await fetch('/api/detect', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frame })
            });
            const data = await resp.json();
            if (data.face_detected && data.bbox) {
                const [x, y, w, h] = data.bbox;
                const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
                const cw = videoEl.clientWidth, ch = videoEl.clientHeight;
                // Mirror X because video is flipped
                const px = ((vw - x - w) / vw) * cw;
                const py = (y / vh) * ch;
                const pw = (w / vw) * cw;
                const ph = (h / vh) * ch;
                faceBox.style.display = 'block';
                faceBox.style.left = px + 'px';
                faceBox.style.top = py + 'px';
                faceBox.style.width = pw + 'px';
                faceBox.style.height = ph + 'px';
                faceDetectedState = true;
                if (onDetectionChange) onDetectionChange(true);
            } else {
                faceBox.style.display = 'none';
                faceDetectedState = false;
                if (onDetectionChange) onDetectionChange(false);
            }
        } catch {
            faceBox.style.display = 'none';
            faceDetectedState = false;
            if (onDetectionChange) onDetectionChange(false);
        }
    }, 500);
}

function stopFaceDetection() {
    if (detectInterval) { clearInterval(detectInterval); detectInterval = null; }
}

// ══════════════════════════════════════════════════════════
// ENROLLMENT FLOW
// ══════════════════════════════════════════════════════════
function initEnrollUI() {
    currentPose = 0; capturedImages = [];
    thumbGrid.innerHTML = '';
    for (let i = 0; i < 9; i++) {
        const div = document.createElement('div');
        div.className = 'thumb-item empty'; div.id = `thumb-${i}`;
        thumbGrid.appendChild(div);
    }
    updatePoseProgress();
    updateFaceGuide(0);
    enrollPoseLabel.textContent = 'Pose 1/9';
    hideStatus();
}

function updatePoseProgress() {
    poseProgress.querySelectorAll('.pose-dot').forEach((dot, i) => {
        dot.classList.remove('active', 'captured');
        if (i < currentPose) dot.classList.add('captured');
        else if (i === currentPose) dot.classList.add('active');
    });
}

function showStatus(type, msg) {
    enrollStatus.className = `enroll-status ${type}`;
    enrollStatusMsg.textContent = msg;
    enrollStatusIcon.innerHTML = type === 'success'
        ? '<svg viewBox="0 0 24 24" width="24" height="24" stroke="#10b981" stroke-width="2" fill="none"><polyline points="20 6 9 17 4 12"/></svg>'
        : '<svg viewBox="0 0 24 24" width="24" height="24" stroke="#ef4444" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    enrollStatus.classList.remove('hidden');
}

function hideStatus() { enrollStatus.classList.add('hidden'); }

startEnrollBtn.addEventListener('click', async () => {
    const name = enrollName.value.trim();
    if (!name) { enrollName.focus(); enrollName.style.borderColor = '#ef4444'; return; }
    enrollName.style.borderColor = '';
    
    enrollStream = await startWebcam(enrollVideo);
    if (!enrollStream) return;

    enrollPlaceholder.classList.add('hidden');
    startEnrollBtn.classList.add('hidden');
    captureBtn.classList.remove('hidden');
    resetEnrollBtn.classList.remove('hidden');
    nameInputGroup.style.opacity = '0.5';
    nameInputGroup.style.pointerEvents = 'none';
    
    initEnrollUI();
    startFaceDetection(enrollVideo, enrollCanvas, enrollFaceBox, (detected) => {
        if (captureBtn.classList.contains('hidden')) return;
        captureBtn.disabled = !detected;
        captureBtn.querySelector('span').textContent = detected ? 'Capture Pose' : 'No face detected';
    });
});

captureBtn.addEventListener('click', () => {
    if (currentPose >= 9 || !faceDetectedState) return;
    
    const frame = captureFrame(enrollVideo, enrollCanvas);
    capturedImages.push(frame);

    // Update thumbnail
    const thumb = document.getElementById(`thumb-${currentPose}`);
    thumb.classList.remove('empty');
    thumb.innerHTML = `<img src="${frame}" alt="Pose ${currentPose + 1}">`;

    currentPose++;
    updatePoseProgress();

    if (currentPose < 9) {
        enrollPoseLabel.textContent = `Pose ${currentPose + 1}/9`;
        updateFaceGuide(currentPose);
    } else {
        enrollPoseLabel.textContent = 'All Captured!';
        captureBtn.classList.add('hidden');
        submitEnrollment();
    }
});

async function submitEnrollment() {
    const name = enrollName.value.trim();
    startEnrollBtn.disabled = true;
    showStatus('success', 'Submitting enrollment...');

    try {
        const resp = await fetch('/api/enroll', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, images: capturedImages })
        });
        const data = await resp.json();

        if (data.status === 'success') {
            showStatus('success', `✅ ${data.message}`);
        } else if (data.status === 'duplicate') {
            showStatus('duplicate', `⚠️ ${data.message}`);
        } else {
            showStatus('error', `❌ ${data.message}`);
        }
    } catch (e) {
        showStatus('error', `❌ Error: ${e.message}`);
    }

    startEnrollBtn.disabled = false;
}

resetEnrollBtn.addEventListener('click', () => {
    stopFaceDetection();
    stopWebcam(enrollStream, enrollVideo);
    enrollStream = null;
    enrollPlaceholder.classList.remove('hidden');
    captureBtn.classList.add('hidden');
    resetEnrollBtn.classList.add('hidden');
    startEnrollBtn.classList.remove('hidden');
    nameInputGroup.style.opacity = '1';
    nameInputGroup.style.pointerEvents = 'auto';
    enrollFaceBox.style.display = 'none';
    initEnrollUI();
});

// ══════════════════════════════════════════════════════════
// VERIFICATION / LOGIN FLOW
// ══════════════════════════════════════════════════════════
startLoginBtn.addEventListener('click', async () => {
    loginStream = await startWebcam(loginVideo);
    if (!loginStream) return;

    loginPlaceholder.classList.add('hidden');
    startLoginBtn.classList.add('hidden');
    verifyBtn.classList.remove('hidden');
    dashboardSection.classList.add('hidden');
    loginStatusBadge.textContent = 'Camera Active';
    loginStatusBadge.className = 'badge green';
    startFaceDetection(loginVideo, loginCanvas, $('loginFaceBox'), null);
});

verifyBtn.addEventListener('click', async () => {
    verifyBtn.disabled = true;
    verifyBtn.querySelector('span').textContent = 'Scanning...';
    scanLine.classList.remove('hidden');
    loginStatusBadge.textContent = 'Scanning';
    loginStatusBadge.className = 'badge cyan';

    const frame = captureFrame(loginVideo, loginCanvas);

    try {
        const resp = await fetch('/api/verify', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frame })
        });
        const data = await resp.json();

        scanLine.classList.add('hidden');

        if (data.access === 'granted') {
            showDashboard(data, true);
        } else {
            showDashboard(data, false);
        }
    } catch (e) {
        alert('Verification failed: ' + e.message);
    }

    verifyBtn.disabled = false;
    verifyBtn.querySelector('span').textContent = 'Verify Identity';
});

function showDashboard(data, granted) {
    dashboardSection.classList.remove('hidden');

    if (granted) {
        accessCard.className = 'access-card glass-panel granted';
        accessIcon.className = 'access-icon granted';
        accessIcon.innerHTML = '<svg viewBox="0 0 24 24" width="64" height="64" stroke="currentColor" stroke-width="1.5" fill="none"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
        accessLabel.textContent = 'ACCESS GRANTED';
        accessLabel.className = 'access-label granted';
        loginStatusBadge.textContent = 'Granted';
        loginStatusBadge.className = 'badge green';

        const user = data.user;
        profileName.textContent = user.name;
        profileId.textContent = user.id.toUpperCase();
        confidenceValue.textContent = (data.score * 100).toFixed(1) + '%';
        confidenceFill.style.width = (data.score * 100) + '%';
        enrolledDate.textContent = user.enrolled_date ? new Date(user.enrolled_date).toLocaleDateString() : 'Pre-loaded (ORL)';
        profileSource.textContent = user.source === 'webcam' ? 'Webcam Enrollment' : 'ORL Dataset';
        profileImageCount.textContent = user.image_count;

        if (user.thumbnail) {
            profileThumb.src = 'data:image/jpeg;base64,' + user.thumbnail;
        } else {
            profileThumb.src = '';
        }
        $('profileCard').classList.remove('hidden');
    } else {
        accessCard.className = 'access-card glass-panel denied';
        accessIcon.className = 'access-icon denied';
        accessIcon.innerHTML = '<svg viewBox="0 0 24 24" width="64" height="64" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
        accessLabel.textContent = 'ACCESS DENIED';
        accessLabel.className = 'access-label denied';
        loginStatusBadge.textContent = 'Denied';
        loginStatusBadge.className = 'badge red';

        // Show diagnostic score info so the user can see what happened
        const score = data.best_score ?? data.score ?? null;
        const thr   = data.threshold ?? null;
        const reason = data.message  ?? '';
        let debugMsg = reason;
        if (score !== null) debugMsg += `  |  Score: ${(score*100).toFixed(1)}%`;
        if (thr   !== null) debugMsg += `  |  Required: ${(thr*100).toFixed(1)}%`;
        const existing = accessCard.querySelector('.deny-debug');
        if (existing) existing.remove();
        const dbg = document.createElement('p');
        dbg.className = 'deny-debug';
        dbg.style.cssText = 'font-size:0.8rem;opacity:0.7;margin-top:8px;font-family:monospace;';
        dbg.textContent = debugMsg;
        accessCard.appendChild(dbg);

        $('profileCard').classList.add('hidden');
    }
}

loginAgainBtn.addEventListener('click', () => {
    dashboardSection.classList.add('hidden');
    loginStatusBadge.textContent = 'Camera Active';
    loginStatusBadge.className = 'badge green';
});

// ══════════════════════════════════════════════════════════
// EVALUATION FLOW (preserved from original)
// ══════════════════════════════════════════════════════════
startEvalBtn.addEventListener('click', async () => {
    startEvalBtn.disabled = true;
    const btnText = startEvalBtn.querySelector('.eval-btn-text');
    btnText.textContent = 'Processing...';
    evalLoader.classList.remove('hidden');
    evalResults.classList.add('hidden');
    evalLoading.classList.remove('hidden');

    try {
        const resp = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        const result = await resp.json();

        if (resp.ok && result.status === 'success') {
            pcaStats.innerHTML = genStatsHTML(result.data.pca);
            lbpStats.innerHTML = genStatsHTML(result.data.lbp);
            resultsPlot.src = result.data.plot_url + '?t=' + Date.now();
            resultsPlot.onload = () => { plotPlaceholder.classList.add('hidden'); resultsPlot.classList.remove('hidden'); };
            evalResults.classList.remove('hidden');
        } else {
            throw new Error(result.message || 'Evaluation failed');
        }
    } catch (e) {
        alert('Evaluation error: ' + e.message);
    }

    startEvalBtn.disabled = false;
    btnText.textContent = 'Re-run Evaluation';
    evalLoader.classList.add('hidden');
    evalLoading.classList.add('hidden');
});

function genStatsHTML(m) {
    const pct = v => (v * 100).toFixed(2) + '%';
    const fl = v => v.toFixed(4);
    return `
        <div class="stat-item"><span class="stat-label">Equal Error Rate</span><span class="stat-value highlight">${pct(m.eer)}</span></div>
        <div class="stat-item"><span class="stat-label">Rank-1 Accuracy</span><span class="stat-value">${pct(m.rank1)}</span></div>
        <div class="stat-item"><span class="stat-label">D-Prime</span><span class="stat-value">${fl(m.d_prime)}</span></div>
        <div class="stat-item"><span class="stat-label">AUC</span><span class="stat-value">${fl(m.auc)}</span></div>
        <div class="stat-item"><span class="stat-label">TMR @ 1% FMR</span><span class="stat-value">${pct(m.tmr_1)}</span></div>
        <div class="stat-item"><span class="stat-label">TMR @ 0.01% FMR</span><span class="stat-value">${pct(m.tmr_001)}</span></div>
        <div class="stat-item"><span class="stat-label">FPIR @ EER</span><span class="stat-value">${pct(m.fpir)}</span></div>
        <div class="stat-item"><span class="stat-label">FNIR @ EER</span><span class="stat-value">${pct(m.fnir)}</span></div>`;
}

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════
initTabs();
initEnrollUI();

});
