import { DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('webcam') as HTMLVideoElement;
const canvas = document.getElementById('output') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const btn = document.getElementById('btn') as HTMLButtonElement;
const startDiv = document.getElementById('start')!;

const worker = new Worker(
  new URL('./workers/hand-landmarker.worker.ts', import.meta.url),
  { type: 'module' }
);

let ready = false;
let running = false;
let busy = false;
let lastT = -1;

// ─── 性能统计 ────────────────────────────────────────────
const fpsEl = document.getElementById('fps')!;
const latEl = document.getElementById('lat')!;
const framesEl = document.getElementById('frames')!;
let frameCount = 0;
let fpsFrames = 0;
let fpsLastTime = performance.now();

// ─── Worker 消息 ────────────────────────────────────────
worker.onmessage = (e) => {
  const { type, result, inferenceTime, loaded, total } = e.data;

  if (type === 'INIT_DONE') {
    ready = true;
    startCamera();
  } else if (type === 'LOAD_PROGRESS') {
    btn.textContent = `加载中 ${Math.round(loaded / total * 100)}%`;
  } else if (type === 'DETECT_RESULT') {
    busy = false;
    frameCount++;
    fpsFrames++;

    latEl.textContent = `延迟: ${inferenceTime.toFixed(1)}ms`;
    framesEl.textContent = `帧数: ${frameCount}`;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (result.landmarks) {
      const du = new DrawingUtils(ctx);
      for (const lm of result.landmarks) {
        du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        du.drawLandmarks(lm, { color: '#FF0000', lineWidth: 2 });
      }
    }
  }
};

// ─── 启动 ────────────────────────────────────────────────
btn.addEventListener('click', async () => {
  btn.disabled = true; btn.textContent = '加载模型中...';
  worker.postMessage({
    type: 'INIT',
    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    delegate: 'GPU',
    baseUrl: import.meta.env.BASE_URL,
    numHands: 2,
    runningMode: 'VIDEO',
  });
});

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
  video.srcObject = stream;
  await video.play();
  startDiv.classList.add('hidden');
  running = true;
  requestAnimationFrame(loop);
}

// ─── 主循环 (busy 锁防卡死) ──────────────────────────────
function loop() {
  if (!running || !ready) return;

  // FPS 每秒更新
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
      worker.postMessage(
        { type: 'DETECT_VIDEO', bitmap, timestampMs: performance.now() },
        [bitmap]
      );
    }).catch(() => { busy = false; });
  }

  requestAnimationFrame(loop);
}
