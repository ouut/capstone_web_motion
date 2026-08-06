import { DrawingUtils, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';

declare const gatewayProtocol: {
  PKT_RAW_MOTION: number; TGT_BROADCAST: number;
  encode: (o: any) => Uint8Array; decode: (b: Uint8Array|ArrayBuffer) => any;
};

declare class LightsaberTracker {
  constructor(config?: { shoulderPos?: number[]; armLength?: number; handLength?: number; imageSize?: number[]; fov?: number; smoothing?: number });
  processHandLightsabers(lm: { leftWrist: number[]; leftIndex: number[]; rightWrist: number[]; rightIndex: number[] }):
    { left: { position: number[]; rotation: number[]; direction: number[] }; right: { position: number[]; rotation: number[]; direction: number[] } };
  processBodyLightsabers(lm: { leftElbow: number[]; leftWrist: number[]; rightElbow: number[]; rightWrist: number[] }):
    { left: { position: number[]; rotation: number[]; direction: number[] }; right: { position: number[]; rotation: number[]; direction: number[] } };
}

const video  = document.getElementById('webcam') as HTMLVideoElement;
const canvas = document.getElementById('output') as HTMLCanvasElement;
const ctx    = canvas.getContext('2d')!;
const btn    = document.getElementById('btn') as HTMLButtonElement;
const startDiv = document.getElementById('start')!;

// ══════════════════════════════════════════════════════════
// WebSocket
// ══════════════════════════════════════════════════════════

const wsUserEl = document.getElementById('ws-user') as HTMLInputElement;
const wsRoomEl = document.getElementById('ws-room') as HTMLInputElement;
const wsHostEl = document.getElementById('ws-host') as HTMLInputElement;
const wsBtnEl  = document.getElementById('ws-btn') as HTMLButtonElement;

let ws: WebSocket | null = null;
let wsSeq = 0;

// 光剑追踪器 (2D → 3D)
const lsTracker = new LightsaberTracker({
  shoulderPos: [0, -0.15, -0.4],
  armLength: 0.65,
  handLength: 0.18,
  imageSize: [640, 480],
  fov: 60,
  smoothing: 0.7,
});

// 正则验证: user ≤ 8 ASCII, room ≤ 6 ASCII, host = domain:port 或 ip:port
const RE_ASCII = /^[\x21-\x7E]{1,8}$/;   // user
const RE_ROOM  = /^[\x21-\x7E]{1,6}$/;   // room
const RE_HOST  = /^[\w.-]+:\d{2,5}$/;     // host:port

function validateWsInputs(): string | null {
  const u = wsUserEl.value.trim();
  const r = wsRoomEl.value.trim();
  const h = wsHostEl.value.trim();
  if (!u || !RE_ASCII.test(u)) return 'User: 1-8 ASCII chars required';
  if (!r || !RE_ROOM.test(r)) return 'Room: 1-6 ASCII chars required';
  if (!h || !RE_HOST.test(h)) return 'Host: e.g. 192.168.1.1:8080 or example.com:8080';
  return null;
}

function wsConnect() {
  if (ws) { ws.close(); ws = null; }

  const err = validateWsInputs();
  if (err) { alert(err); return; }

  const u = wsUserEl.value.trim();
  const r = wsRoomEl.value.trim();
  const h = wsHostEl.value.trim();

  const url = `ws://${h}/ws?room=${encodeURIComponent(r)}&user=${encodeURIComponent(u)}`;

  try {
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      wsBtnEl.textContent = '🟢';
      wsBtnEl.className = 'connected';
      console.log(`WS connected: ${url}`);
    };
    ws.onclose = () => {
      wsBtnEl.textContent = '🔗';
      wsBtnEl.className = '';
      ws = null;
    };
    ws.onerror = () => {
      wsBtnEl.textContent = '🔴';
      wsBtnEl.className = 'error';
    };
  } catch (e: any) {
    alert('WebSocket error: ' + e.message);
  }
}

wsBtnEl.addEventListener('click', () => {
  if (ws) { ws.close(); ws = null; }
  else wsConnect();
});

// 打包光剑数据: 每把剑 6个 float32 (剑柄 xyz + 剑尖 xyz), 2把 = 12 = 48 bytes
function packLightsabers(sabers: {hilt:number[]; tip:number[]}[]): Uint8Array {
  const buf = new ArrayBuffer(48);
  const dv = new DataView(buf);
  let off = 0;
  for (const s of sabers) {
    dv.setFloat32(off,      s.hilt[0], true); // little-endian
    dv.setFloat32(off + 4,  s.hilt[1], true);
    dv.setFloat32(off + 8,  s.hilt[2], true);
    dv.setFloat32(off + 12, s.tip[0],  true);
    dv.setFloat32(off + 16, s.tip[1],  true);
    dv.setFloat32(off + 20, s.tip[2],  true);
    off += 24;
  }
  return new Uint8Array(buf);
}

// 从检测结果 + LightsaberTracker 计算光剑 3D 坐标
function extractLightsabers(result: any): {hilt:number[]; tip:number[]}[] {
  const zero = { position: [0,0,0] as number[], direction: [0,0,1] as number[] };
  const handLen = 0.18, forearmLen = 0.29;

  if (getMode() === 'hand') {
    // Hands: Wrist(0) → Index_TIP(8)
    const lm0 = result.landmarks?.[0];
    const lm1 = result.landmarks?.[1];
    const pts: any = {
      leftWrist: lm0 ? [lm0[0].x, lm0[0].y] : [0,0],
      leftIndex: lm0 ? [lm0[8].x, lm0[8].y] : [0,0],
      rightWrist: lm1 ? [lm1[0].x, lm1[0].y] : [0,0],
      rightIndex: lm1 ? [lm1[8].x, lm1[8].y] : [0,0],
    };
    const ls = lsTracker.processHandLightsabers(pts);
    return [
      { hilt: ls.left.position,  tip: vecAdd(ls.left.position, vecMul(ls.left.direction, handLen)) },
      { hilt: ls.right.position, tip: vecAdd(ls.right.position, vecMul(ls.right.direction, handLen)) },
    ];
  } else {
    // Body: Elbow(13) → Wrist(15), L/R
    const lm0 = result.landmarks?.[0];
    const pts: any = {
      leftElbow:  lm0 ? [lm0[13].x, lm0[13].y] : [0,0],
      leftWrist:  lm0 ? [lm0[15].x, lm0[15].y] : [0,0],
      rightElbow: lm0 ? [lm0[14].x, lm0[14].y] : [0,0],
      rightWrist: lm0 ? [lm0[16].x, lm0[16].y] : [0,0],
    };
    const ls = lsTracker.processBodyLightsabers(pts);
    return [
      { hilt: ls.left.position,  tip: vecAdd(ls.left.position, vecMul(ls.left.direction, forearmLen)) },
      { hilt: ls.right.position, tip: vecAdd(ls.right.position, vecMul(ls.right.direction, forearmLen)) },
    ];
  }
}

function vecAdd(a: number[], b: number[]): number[] { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function vecMul(a: number[], s: number): number[] { return [a[0]*s, a[1]*s, a[2]*s]; }

function sendWsMotion(result: any) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const sabers = extractLightsabers(result);
  const payload = packLightsabers(sabers);
  const wire = gatewayProtocol.encode({
    version: 0x02,
    pktType: gatewayProtocol.PKT_RAW_MOTION,
    tgtType: gatewayProtocol.TGT_BROADCAST,
    roomId: wsRoomEl.value.trim(),
    userId: wsUserEl.value.trim(),
    seq: wsSeq++,
    payload,
  });
  ws.send(wire.buffer);
}

// 双手 / 身体 (从 radio 读取)
function getMode(): 'hand' | 'pose' {
  const checked = document.querySelector('input[name="mode"]:checked') as HTMLInputElement;
  return checked?.value as 'hand' | 'pose' || 'hand';
}

// 两个 Worker URL (Vite 要求静态路径)
const handWorkerUrl = new URL('./workers/hand-landmarker.worker.ts', import.meta.url);
const poseWorkerUrl = new URL('./workers/pose-landmarker.worker.ts', import.meta.url);

function createWorker(): Worker {
  return new Worker(
    getMode() === 'hand' ? handWorkerUrl : poseWorkerUrl,
    { type: 'module' }
  );
}

let worker: Worker;
let ready = false;
let running = false;
let busy = false;
let lastT = -1;

// 性能统计
const fpsEl    = document.getElementById('fps')!;
const latEl    = document.getElementById('lat')!;
const framesEl = document.getElementById('frames')!;
let frameCount = 0;
let fpsFrames  = 0;
let fpsLastTime = performance.now();

// ══════════════════════════════════════════════════════════
// Console 日志系统  (hl)
// ══════════════════════════════════════════════════════════

const HAND_LANDMARKS: Record<number,string> = {
  0:'Wrist',1:'Thumb_CMC',2:'Thumb_MCP',3:'Thumb_IP',4:'Thumb_TIP',
  5:'Index_MCP',6:'Index_PIP',7:'Index_DIP',8:'Index_TIP',
  9:'Middle_MCP',10:'Middle_PIP',11:'Middle_DIP',12:'Middle_TIP',
  13:'Ring_MCP',14:'Ring_PIP',15:'Ring_DIP',16:'Ring_TIP',
  17:'Pinky_MCP',18:'Pinky_PIP',19:'Pinky_DIP',20:'Pinky_TIP',
};

const POSE_LANDMARKS: Record<number,string> = {
  0:'Nose',1:'Left_Eye_Inner',2:'Left_Eye',3:'Left_Eye_Outer',
  4:'Right_Eye_Inner',5:'Right_Eye',6:'Right_Eye_Outer',
  7:'Left_Ear',8:'Right_Ear',9:'Left_Mouth',10:'Right_Mouth',
  11:'Left_Shoulder',12:'Right_Shoulder',13:'Left_Elbow',14:'Right_Elbow',
  15:'Left_Wrist',16:'Right_Wrist',17:'Left_Pinky',18:'Right_Pinky',
  19:'Left_Index',20:'Right_Index',21:'Left_Thumb',22:'Right_Thumb',
  23:'Left_Hip',24:'Right_Hip',25:'Left_Knee',26:'Right_Knee',
  27:'Left_Ankle',28:'Right_Ankle',29:'Left_Heel',30:'Right_Heel',
  31:'Left_Foot_Index',32:'Right_Foot_Index',
};

const HAND_EMOJI = ['🟢','🔴'];
const RST = '\x1b[0m';

type TrackCallback = (d: {label:string; x:number; y:number; z:number; dx:number; dy:number; dz:number}) => void;

let tracked: Map<string, TrackCallback | null> = new Map(); // name → callback (null=console)
let throttle = 10;
let useCsv = false;
let prevValues: Map<string,{x:number;y:number;z:number}> = new Map();

function lmNames(): Record<number,string> {
  return getMode() === 'hand' ? HAND_LANDMARKS : POSE_LANDMARKS;
}

type Resolved = {label:string; x:number; y:number; z:number; cb:TrackCallback|null};
function resolveTracked(result: any): Resolved[] {
  const out: Resolved[] = [];
  if (!result.landmarks) return out;
  const names = lmNames();

  if (getMode() === 'hand') {
    for (let hi = 0; hi < result.landmarks.length; hi++) {
      const side = result.handedness?.[hi]?.[0]?.displayName || `H${hi}`;
      for (const [idx, name] of Object.entries(names)) {
        const cb = tracked.get(name);
        if (!cb && !tracked.has('*')) continue;
        const lm = result.landmarks[hi]![parseInt(idx)];
        out.push({ label: `${HAND_EMOJI[hi]} ${side}:${name}`, x: lm.x, y: lm.y, z: lm.z, cb: cb || null });
      }
    }
  } else {
    for (let pi = 0; pi < result.landmarks.length; pi++) {
      const prefix = result.landmarks.length > 1 ? `P${pi}:` : '';
      for (const [idx, name] of Object.entries(names)) {
        const cb = tracked.get(name);
        if (!cb && !tracked.has('*')) continue;
        const lm = result.landmarks[pi]![parseInt(idx)];
        out.push({ label: `🧍 ${prefix}${name}`, x: lm.x, y: lm.y, z: lm.z, cb: cb || null });
      }
    }
  }
  return out;
}

function colorDelta(cur: number, prev: number): string {
  const d = Math.abs(cur - prev);
  if (d > 0.03) return '\x1b[38;5;196m';  // 红
  if (d > 0.01) return '\x1b[38;5;214m';  // 橙
  if (d > 0.003) return '\x1b[38;5;228m'; // 黄
  return '\x1b[38;5;250m';                 // 灰
}

function processFrame(result: any) {
  if (tracked.size === 0) return;
  const items = resolveTracked(result);
  if (items.length === 0) return;

  // 1. 每个帧都调用用户 callback (不受 throttle 限制)
  for (const item of items) {
    const prev = prevValues.get(item.label);
    const d = {
      label: item.label,
      x: item.x, y: item.y, z: item.z,
      dx: prev ? item.x - prev.x : 0,
      dy: prev ? item.y - prev.y : 0,
      dz: prev ? item.z - prev.z : 0,
    };
    if (item.cb) item.cb(d);  // 用户 callback，每帧触发
    prevValues.set(item.label, {x:item.x, y:item.y, z:item.z});
  }

  // 2. Console 日志按 throttle 节流
  if (frameCount % throttle !== 0) return;

  if (useCsv) {
    const header = 'Frame,' + items.map(i => `${i.label}_x,${i.label}_y,${i.label}_z`).join(',');
    console.log(`%c[CSV] #${frameCount}`, 'color:#00bcd4;font-weight:bold');
    console.log('%c' + header, 'color:#888;font-size:0.7em');
    let row = '';
    items.forEach((item, idx) => {
      const prev = prevValues.get(item.label);
      for (const [j, cur] of [item.x, item.y, item.z].entries()) {
        const p = [prev?.x, prev?.y, prev?.z][j];
        row += (p !== undefined ? colorDelta(cur, p!) : '\x1b[38;5;250m') + cur.toFixed(4) + RST;
        if (idx < items.length - 1 || j < 2) row += ',';
      }
    });
    console.log(row);
  } else {
    console.log(`%c── #${frameCount}  %c${getMode()==='hand'?'Hands':'Body'}  %c${items.length}pts ──`,
      'color:#00bcd4', 'color:#888', 'color:#888');
    for (const item of items) {
      const prev = prevValues.get(item.label);
      let line = `  ${item.label.padEnd(26)}`;
      for (const [axis, cur] of [['x',item.x],['y',item.y],['z',item.z]] as [string,number][]) {
        const j = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
        const p = [prev?.x, prev?.y, prev?.z][j];
        line += ` ${axis}:${(p !== undefined ? colorDelta(cur, p!) : '\x1b[38;5;250m') + cur.toFixed(3)}${RST}`;
      }
      console.log(line);
    }
  }
}

interface HL {
  list(): void;
  on(name: string|string[], cb?: TrackCallback): void;
  off(name: string|string[]): void;
  onAll(): void;
  offAll(): void;
  csv(on: boolean): void;
  throttle(n: number): void;
  status(): void;
  help(): void;
}

const hl: HL = {
  list() {
    console.table(Object.entries(lmNames()).map(([i,n]) => ({index:+i, name:n})));
  },
  on(name: string|string[], cb?: TrackCallback) {
    const arr = Array.isArray(name) ? name : [name];
    arr.forEach(n => tracked.set(n, cb || null));
    tracked.delete('*');
    console.log(`✅ Tracking: ${arr.join(', ')} (${tracked.size} total)${cb ? ' [callback]' : ''}`);
  },
  off(name: string|string[]) {
    const arr = Array.isArray(name) ? name : [name];
    arr.forEach(n => tracked.delete(n));
    tracked.delete('*');
    console.log(`⏹ Stopped: ${arr.join(', ')}`);
  },
  onAll()  { tracked.clear(); tracked.set('*', null); console.log('✅ Tracking all'); },
  offAll() { tracked.clear(); prevValues.clear(); console.log('⏹ Stopped all'); },
  csv(on: boolean)  { useCsv = on; console.log(`CSV: ${on?'ON':'OFF'}`); },
  throttle(n: number) { throttle = Math.max(1, n); console.log(`Throttle: 1/${throttle}`); },
  status() {
    const m = getMode();
    const names = [...tracked.keys()].filter(k => k !== '*');
    console.log(`Mode: ${m==='hand'?'Hands':'Body'} | CSV: ${useCsv?'ON':'OFF'} | Throttle: 1/${throttle} | FPS: ${fpsFrames} | Frames: ${frameCount} | Tracked(${names.length}): ${names.join(', ') || 'none'}`);
  },
  help() {
    console.log(
`%c╔══════════════════════════════════════════════╗
║  hl Console Commands                         ║
╠══════════════════════════════════════════════╣
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
╚══════════════════════════════════════════════╝`, 'color:#00bcd4');
  },
};
(window as any).hl = hl;

// ══════════════════════════════════════════════════════════
// 绘制
// ══════════════════════════════════════════════════════════

function draw(result: any) {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result.landmarks) return;

  const du = new DrawingUtils(ctx);
  if (getMode() === 'hand') {
    for (const lm of result.landmarks) {
      du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
      du.drawLandmarks(lm, { color: '#FF0000', lineWidth: 2 });
    }
  } else {
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
    } else if (type === 'LOAD_PROGRESS') {
      btn.textContent = `Loading ${Math.round(loaded / total * 100)}%`;
    } else if (type === 'DETECT_RESULT') {
      busy = false;
      frameCount++; fpsFrames++;
      latEl.textContent    = `Latency: ${inferenceTime.toFixed(1)}ms`;
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
  btn.disabled = true; btn.textContent = 'Loading model...';

  const modelUrl = getMode() === 'hand'
    ? 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    : 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

  const initParams: Record<string, any> = {
    type: 'INIT',
    modelAssetPath: modelUrl,
    delegate: 'GPU',
    baseUrl: import.meta.env.BASE_URL,
    runningMode: 'VIDEO',
  };

  if (getMode() === 'hand') {
    initParams.numHands = 2;
  } else {
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
  await new Promise<void>(r => { if (video.videoWidth) r(); else video.addEventListener('loadedmetadata', () => r(), { once: true }); });
  (lsTracker as any).imageSize = [video.videoWidth, video.videoHeight];
  startDiv.classList.add('hidden');
  running = true;
  requestAnimationFrame(loop);
}

// ══════════════════════════════════════════════════════════
// 主循环
// ══════════════════════════════════════════════════════════

function loop() {
  if (!running || !ready) return;

  const now = performance.now();
  if (now - fpsLastTime >= 1000) {
    fpsEl.textContent = `FPS: ${fpsFrames}`;
    fpsFrames = 0; fpsLastTime = now;
  }

  if (!busy && video.videoWidth > 0 && video.currentTime !== lastT) {
    lastT = video.currentTime;
    busy = true;
    createImageBitmap(video).then((bitmap) => {
      worker.postMessage(
        { type: 'DETECT_VIDEO', bitmap, timestampMs: performance.now() },
        [bitmap]
      );
    }).catch(() => { busy = false; });
  }

  requestAnimationFrame(loop);
}
