import { DrawingUtils, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const btn = document.getElementById('btn');
const startDiv = document.getElementById('start');
// 双手 / 身体 (从 radio 读取)
function getMode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked?.value || 'hand';
}
// 两个 Worker URL (Vite 要求静态路径)
import HandWorker from './workers/hand-landmarker.worker.ts?worker';
import PoseWorker from './workers/pose-landmarker.worker.ts?worker';
function createWorker() {
    return getMode() === 'hand' ? new HandWorker() : new PoseWorker();
}
let worker;
let ready = false;
let running = false;
let busy = false;
let lastT = -1;
// 性能统计
const fpsEl = document.getElementById('fps');
const latEl = document.getElementById('lat');
const framesEl = document.getElementById('frames');
let frameCount = 0;
let fpsFrames = 0;
let fpsLastTime = performance.now();
// ══════════════════════════════════════════════════════════
// WebSocket
// ══════════════════════════════════════════════════════════
const wsRoomEl = document.getElementById('ws-room');
const wsHostEl = document.getElementById('ws-host');
const wsBtnEl = document.getElementById('ws-btn');
let ws = null;
let wsSeq = 0;
// userId: 时间戳低24位(6 hex) + 随机8位(2 hex) = 固定8字符，永不溢出
const userId = (() => {
    const ts = Date.now() & 0xFFFFFF; // 低24位, 0~16.7M
    const rnd = (Math.random() * 0x100) | 0; // 8位, 0~255
    return ts.toString(16).padStart(6, '0') + rnd.toString(16).padStart(2, '0');
})();
// 正则验证: room ≤ 6 ASCII, host = ws[s]://domain[:port]
const RE_ROOM = /^[\x21-\x7E]{1,6}$/;
const RE_HOST = /^wss?:\/\/[\w.-]+(:\d{2,5})?$/;
function validateWsInputs() {
    const r = wsRoomEl.value.trim();
    const h = wsHostEl.value.trim();
    if (!r || !RE_ROOM.test(r))
        return 'Room: 1-6 ASCII chars required';
    if (!h || !RE_HOST.test(h))
        return 'Host: ws://localhost:8080 or wss://example.com';
    return null;
}
function wsConnect() {
    if (ws) {
        ws.close();
        ws = null;
    }
    const err = validateWsInputs();
    if (err) {
        alert(err);
        return;
    }
    const r = wsRoomEl.value.trim();
    const h = wsHostEl.value.trim();
    const url = `${h}/ws?room=${encodeURIComponent(r)}&user=${encodeURIComponent(userId)}`;
    try {
        ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
            wsBtnEl.textContent = '\u{1F7E2}';
            wsBtnEl.className = 'connected';
            const mode = getMode();
            console.log(`%c\u{1F517} WS connected %c${url}`, 'color:#22c55e;font-weight:bold', 'color:#94a3b8');
            console.log(`   Mode: ${mode} | hand=33B body=146B COCO12`);
        };
        ws.onclose = () => {
            wsBtnEl.textContent = '\u{1F517}';
            wsBtnEl.className = '';
            ws = null;
            console.log('%c\u{1F50C} WS disconnected', 'color:#ef4444');
        };
        ws.onerror = () => {
            wsBtnEl.textContent = '\u{1F534}';
            wsBtnEl.className = 'error';
        };
    }
    catch (e) {
        alert('WebSocket error: ' + e.message);
    }
}
wsBtnEl.addEventListener('click', () => {
    if (ws) {
        ws.close();
        ws = null;
    }
    else
        wsConnect();
});
// ── 打包原始 2D 关键点: mode(1 byte) + 8 float32 = 33 bytes ──
function packLandmarks(pts) {
    const buf = new ArrayBuffer(33);
    const dv = new DataView(buf);
    dv.setUint8(0, getMode() === 'hand' ? 0x01 : 0x02);
    for (let i = 0; i < 8; i++)
        dv.setFloat32(1 + i * 4, pts[i] || 0, true);
    return new Uint8Array(buf);
}
function extractLandmarks(result) {
    const fx = (v) => v;
    if (getMode() === 'hand') {
        const lm0 = result.landmarks?.[0];
        const lm1 = result.landmarks?.[1];
        return [
            fx(lm0?.[0]?.x ?? 0), lm0?.[0]?.y ?? 0, // Left Wrist
            fx(lm0?.[8]?.x ?? 0), lm0?.[8]?.y ?? 0, // Left Index_TIP
            fx(lm1?.[0]?.x ?? 0), lm1?.[0]?.y ?? 0, // Right Wrist
            fx(lm1?.[8]?.x ?? 0), lm1?.[8]?.y ?? 0, // Right Index_TIP
        ];
    }
    else {
        const lm0 = result.landmarks?.[0];
        return [
            fx(lm0?.[13]?.x ?? 0), lm0?.[13]?.y ?? 0, // Left Elbow
            fx(lm0?.[15]?.x ?? 0), lm0?.[15]?.y ?? 0, // Left Wrist
            fx(lm0?.[14]?.x ?? 0), lm0?.[14]?.y ?? 0, // Right Elbow
            fx(lm0?.[16]?.x ?? 0), lm0?.[16]?.y ?? 0, // Right Wrist
        ];
    }
}
// ── COCO12 全身 12 关节 (body mode): 1B mode + 1B count + 12×(x,y,z) float32 LE = 146B ──
const COCO12_MP_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
function packBodyCOCO12(lm) {
    const buf = new ArrayBuffer(2 + 12 * 12);
    const dv = new DataView(buf);
    dv.setUint8(0, 0x02);
    dv.setUint8(1, 12);
    let off = 2;
    for (let i = 0; i < 12; i++) {
        const j = lm?.[COCO12_MP_IDX[i]];
        dv.setFloat32(off, j?.x ?? 0, true);
        dv.setFloat32(off + 4, j?.y ?? 0, true);
        dv.setFloat32(off + 8, j?.z ?? 0, true);
        off += 12;
    }
    return new Uint8Array(buf);
}
let lastWsLog = 0;
function sendWsMotion(result) {
    if (!ws || ws.readyState !== WebSocket.OPEN)
        return;
    let payload;
    if (getMode() === 'hand') {
        const pts = extractLandmarks(result);
        payload = packLandmarks(pts);
    }
    else {
        payload = packBodyCOCO12(result.landmarks?.[0]);
    }
    const wire = gatewayProtocol.encode({
        version: 0x02,
        pktType: gatewayProtocol.PKT_RAW_MOTION,
        tgtType: gatewayProtocol.TGT_BROADCAST,
        roomId: wsRoomEl.value.trim(),
        userId: userId,
        seq: wsSeq++,
        payload: payload,
    });
    ws.send(wire.buffer);
    if (frameCount - lastWsLog >= 30) {
        lastWsLog = frameCount;
        if (getMode() === 'hand') {
            const pts = extractLandmarks(result);
            console.log(`%c\u{1F4E1} WS #${wsSeq} %cHands %c| 57B`, 'color:#fbbf24;font-weight:bold', 'color:#94a3b8', 'color:#64748b');
        }
        else {
            console.log(`%c\u{1F4E1} WS #${wsSeq} %cBody COCO12 %c| 170B %c12joints`, 'color:#fbbf24;font-weight:bold', 'color:#22c55e', 'color:#64748b', 'color:#22c55e');
        }
    }
}
// ══════════════════════════════════════════════════════════
// Console 日志系统  (hl)
// ══════════════════════════════════════════════════════════
const HAND_LANDMARKS = {
    0: 'Wrist', 1: 'Thumb_CMC', 2: 'Thumb_MCP', 3: 'Thumb_IP', 4: 'Thumb_TIP',
    5: 'Index_MCP', 6: 'Index_PIP', 7: 'Index_DIP', 8: 'Index_TIP',
    9: 'Middle_MCP', 10: 'Middle_PIP', 11: 'Middle_DIP', 12: 'Middle_TIP',
    13: 'Ring_MCP', 14: 'Ring_PIP', 15: 'Ring_DIP', 16: 'Ring_TIP',
    17: 'Pinky_MCP', 18: 'Pinky_PIP', 19: 'Pinky_DIP', 20: 'Pinky_TIP',
};
const POSE_LANDMARKS = {
    0: 'Nose', 1: 'Left_Eye_Inner', 2: 'Left_Eye', 3: 'Left_Eye_Outer',
    4: 'Right_Eye_Inner', 5: 'Right_Eye', 6: 'Right_Eye_Outer',
    7: 'Left_Ear', 8: 'Right_Ear', 9: 'Left_Mouth', 10: 'Right_Mouth',
    11: 'Left_Shoulder', 12: 'Right_Shoulder', 13: 'Left_Elbow', 14: 'Right_Elbow',
    15: 'Left_Wrist', 16: 'Right_Wrist', 17: 'Left_Pinky', 18: 'Right_Pinky',
    19: 'Left_Index', 20: 'Right_Index', 21: 'Left_Thumb', 22: 'Right_Thumb',
    23: 'Left_Hip', 24: 'Right_Hip', 25: 'Left_Knee', 26: 'Right_Knee',
    27: 'Left_Ankle', 28: 'Right_Ankle', 29: 'Left_Heel', 30: 'Right_Heel',
    31: 'Left_Foot_Index', 32: 'Right_Foot_Index',
};
const HAND_EMOJI = ['\u{1F7E2}', '\u{1F534}'];
const RST = '\x1b[0m';
let tracked = new Map();
let throttle = 10;
let useCsv = false;
let prevValues = new Map();
function lmNames() {
    return getMode() === 'hand' ? HAND_LANDMARKS : POSE_LANDMARKS;
}
function resolveTracked(result) {
    const out = [];
    if (!result.landmarks)
        return out;
    const names = lmNames();
    if (getMode() === 'hand') {
        for (let hi = 0; hi < result.landmarks.length; hi++) {
            const side = result.handedness?.[hi]?.[0]?.displayName || `H${hi}`;
            for (const [idx, name] of Object.entries(names)) {
                const cb = tracked.get(name);
                if (!cb && !tracked.has('*'))
                    continue;
                const lm = result.landmarks[hi][parseInt(idx)];
                out.push({ label: `${HAND_EMOJI[hi]} ${side}:${name}`, x: lm.x, y: lm.y, z: lm.z, cb: cb || null });
            }
        }
    }
    else {
        for (let pi = 0; pi < result.landmarks.length; pi++) {
            const prefix = result.landmarks.length > 1 ? `P${pi}:` : '';
            for (const [idx, name] of Object.entries(names)) {
                const cb = tracked.get(name);
                if (!cb && !tracked.has('*'))
                    continue;
                const lm = result.landmarks[pi][parseInt(idx)];
                out.push({ label: `\u{1F9CD} ${prefix}${name}`, x: lm.x, y: lm.y, z: lm.z, cb: cb || null });
            }
        }
    }
    return out;
}
function colorDelta(cur, prev) {
    const d = Math.abs(cur - prev);
    if (d > 0.03)
        return '\x1b[38;5;196m';
    if (d > 0.01)
        return '\x1b[38;5;214m';
    if (d > 0.003)
        return '\x1b[38;5;228m';
    return '\x1b[38;5;250m';
}
function processFrame(result) {
    if (tracked.size === 0)
        return;
    const items = resolveTracked(result);
    if (items.length === 0)
        return;
    for (const item of items) {
        const prev = prevValues.get(item.label);
        const d = { label: item.label, x: item.x, y: item.y, z: item.z,
            dx: prev ? item.x - prev.x : 0, dy: prev ? item.y - prev.y : 0, dz: prev ? item.z - prev.z : 0 };
        if (item.cb)
            item.cb(d);
        prevValues.set(item.label, { x: item.x, y: item.y, z: item.z });
    }
    if (frameCount % throttle !== 0)
        return;
    if (useCsv) {
        const header = 'Frame,' + items.map(i => `${i.label}_x,${i.label}_y,${i.label}_z`).join(',');
        console.log(`%c[CSV] #${frameCount}`, 'color:#00bcd4;font-weight:bold');
        console.log('%c' + header, 'color:#888;font-size:0.7em');
        let row = '';
        items.forEach((item, idx) => {
            const prev = prevValues.get(item.label);
            for (const [j, cur] of [item.x, item.y, item.z].entries()) {
                const p = [prev?.x, prev?.y, prev?.z][j];
                row += (p !== undefined ? colorDelta(cur, p) : '\x1b[38;5;250m') + cur.toFixed(4) + RST;
                if (idx < items.length - 1 || j < 2)
                    row += ',';
            }
        });
        console.log(row);
    }
    else {
        console.log(`%c── #${frameCount}  %c${getMode() === 'hand' ? 'Hands' : 'Body'}  %c${items.length}pts ──`, 'color:#00bcd4', 'color:#888', 'color:#888');
        for (const item of items) {
            const prev = prevValues.get(item.label);
            let line = `  ${item.label.padEnd(26)}`;
            for (const [axis, cur] of [['x', item.x], ['y', item.y], ['z', item.z]]) {
                const j = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
                const p = [prev?.x, prev?.y, prev?.z][j];
                line += ` ${axis}:${(p !== undefined ? colorDelta(cur, p) : '\x1b[38;5;250m') + cur.toFixed(3)}${RST}`;
            }
            console.log(line);
        }
    }
}
const hl = {
    list() { console.table(Object.entries(lmNames()).map(([i, n]) => ({ index: +i, name: n }))); },
    on(name, cb) {
        const arr = Array.isArray(name) ? name : [name];
        arr.forEach(n => tracked.set(n, cb || null));
        tracked.delete('*');
        console.log(`✅ Tracking: ${arr.join(', ')} (${tracked.size} total)${cb ? ' [callback]' : ''}`);
    },
    off(name) {
        const arr = Array.isArray(name) ? name : [name];
        arr.forEach(n => tracked.delete(n));
        tracked.delete('*');
        console.log(`⏹ Stopped: ${arr.join(', ')}`);
    },
    onAll() { tracked.clear(); tracked.set('*', null); console.log('✅ Tracking all'); },
    offAll() { tracked.clear(); prevValues.clear(); console.log('⏹ Stopped all'); },
    csv(on) { useCsv = on; console.log(`CSV: ${on ? 'ON' : 'OFF'}`); },
    throttle(n) { throttle = Math.max(1, n); console.log(`Throttle: 1/${throttle}`); },
    status() {
        const m = getMode();
        const names = [...tracked.keys()].filter(k => k !== '*');
        console.log(`Mode: ${m === 'hand' ? 'Hands' : 'Body'} | CSV: ${useCsv ? 'ON' : 'OFF'} | Throttle: 1/${throttle} | FPS: ${fpsFrames} | Frames: ${frameCount} | Tracked(${names.length}): ${names.join(', ') || 'none'}`);
    },
    help() {
        console.log(`%c╔══════════════════════════════════════╗
║  hl Console Commands                         ║
╠══════════════════════════════════════╣
║  hl.list()          List all landmarks       ║
║  hl.on('Wrist')     Track a landmark         ║
║  hl.on('Wrist', fn) Track with callback      ║
║  hl.on(['A','B'])   Track multiple           ║
║  hl.off('Wrist')    Stop tracking            ║
║  hl.onAll()         Track all                ║
║  hl.offAll()        Stop all                 ║
║  hl.csv(true)       Toggle CSV mode          ║
║  hl.throttle(1)     Set sample rate (1=all)  ║
║  hl.status()        Show current state       ║
║  hl.help()          Show this help           ║
╚══════════════════════════════════════╝`, 'color:#00bcd4');
    },
};
window.hl = hl;
// ══════════════════════════════════════════════════════════
// 绘制
// ══════════════════════════════════════════════════════════
function draw(result) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!result.landmarks)
        return;
    const du = new DrawingUtils(ctx);
    if (getMode() === 'hand') {
        for (const lm of result.landmarks) {
            du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
            du.drawLandmarks(lm, { color: '#FF0000', lineWidth: 2 });
        }
    }
    else {
        for (const lm of result.landmarks) {
            du.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: '#FFFFFF', lineWidth: 4 });
            du.drawLandmarks(lm, { color: '#00BFFF', lineWidth: 3, radius: 5 });
        }
    }
}
// ══════════════════════════════════════════════════════════
// Worker
// ══════════════════════════════════════════════════════════
function initWorker() {
    worker = createWorker();
    worker.onmessage = (e) => {
        const { type, result, inferenceTime, loaded, total } = e.data;
        if (type === 'INIT_DONE') {
            ready = true;
            startCamera();
        }
        else if (type === 'LOAD_PROGRESS') {
            btn.textContent = `Loading ${Math.round(loaded / total * 100)}%`;
        }
        else if (type === 'DETECT_RESULT') {
            busy = false;
            frameCount++;
            fpsFrames++;
            latEl.textContent = `Latency: ${inferenceTime.toFixed(1)}ms`;
            framesEl.textContent = `Frames: ${frameCount}`;
            draw(result);
            processFrame(result);
            sendWsMotion(result);
        }
    };
}
// ══════════════════════════════════════════════════════════
// 启动
// ══════════════════════════════════════════════════════════
btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Loading model...';
    const modelUrl = getMode() === 'hand'
        ? 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        : 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
    const initParams = {
        type: 'INIT', modelAssetPath: modelUrl, delegate: 'GPU',
        baseUrl: import.meta.env.BASE_URL, runningMode: 'VIDEO',
    };
    if (getMode() === 'hand')
        initParams.numHands = 2;
    else {
        initParams.numPoses = 1;
        initParams.outputSegmentationMasks = false;
    }
    initWorker();
    worker.postMessage(initParams);
});
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    video.srcObject = stream;
    await video.play();
    startDiv.classList.add('hidden');
    running = true;
    requestAnimationFrame(loop);
}
// ══════════════════════════════════════════════════════════
// 主循环
// ══════════════════════════════════════════════════════════
function loop() {
    if (!running || !ready)
        return;
    const now = performance.now();
    if (now - fpsLastTime >= 1000) {
        fpsEl.textContent = `FPS: ${fpsFrames}`;
        fpsFrames = 0;
        fpsLastTime = now;
    }
    if (!busy && video.videoWidth > 0 && video.currentTime !== lastT) {
        lastT = video.currentTime;
        busy = true;
        createImageBitmap(video).then((bitmap) => {
            worker.postMessage({ type: 'DETECT_VIDEO', bitmap, timestampMs: performance.now() }, [bitmap]);
        }).catch(() => { busy = false; });
    }
    requestAnimationFrame(loop);
}
