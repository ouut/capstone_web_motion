# Hand / Pose Landmarker Demo

Real-time hand skeleton and body pose tracking using MediaPipe, rendered on a webcam overlay with a programmable console API for landmark data extraction.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173/` in a browser that supports WebGL and Web Workers.

## Project Structure

```
demo/
├── index.html                          # Entry page (UI)
├── package.json                        # Dependencies & scripts
├── tsconfig.json                       # TypeScript config
├── vite.config.ts                      # Vite config (ES module workers)
├── copy-wasm.js                        # Copies WASM files to public/ on install
├── .gitignore
├── README.md
└── src/
    ├── main.ts                         # Main thread: UI, drawing, console API
    └── workers/
        ├── base-worker.ts              # Abstract worker base class
        ├── hand-landmarker.worker.ts   # HandLandmarker worker (21 landmarks × 2 hands)
        └── pose-landmarker.worker.ts   # PoseLandmarker worker (33 landmarks)
```

## Architecture

```
┌─────────────────────────┐      postMessage      ┌────────────────────────┐
│  Main Thread            │ ◄──────────────────► │  Web Worker             │
│                         │                       │                         │
│  main.ts                │   INIT                │  base-worker.ts         │
│  ├─ UI (index.html)     │   ─────────────────►  │  ├─ loadModelAsset()    │
│  ├─ Camera (getUserMedia)│  INIT_DONE           │  ├─ getVisionFileset()  │
│  ├─ DrawingUtils        │   ◄─────────────────  │  └─ diagnoseWebGL()     │
│  └─ hl console API      │                       │                         │
│                         │   DETECT_VIDEO(bitmap)│  hand-landmarker.worker  │
│   requestAnimationFrame │   ─────────────────►  │  └─ detectForVideo()    │
│   → createImageBitmap   │                       │     or                   │
│   → worker.postMessage  │   DETECT_RESULT       │  pose-landmarker.worker  │
│   ← draw(result)        │   ◄─────────────────  │  └─ detectForVideo()    │
└─────────────────────────┘                       └────────────────────────┘
```

### Key Design Decisions

- **Web Worker** offloads ML inference from the UI thread, preventing jank
- **Busy flag** (`isBusy`) ensures only one frame is in-flight at a time — prevents ImageBitmap backlog that caused freezes in earlier versions
- **WASM files** are copied from `node_modules/@mediapipe/tasks-vision/wasm/` to `public/wasm/` by `copy-wasm.js` at dev/build time, served as static assets
- **CSS Grid** (`grid-area: 1 / 1`) stacks `<video>` and `<canvas>` at the same position — resize-proof alignment, no manual coordinate math

## UI

On load, a start card appears with:

```
┌─────────────────────┐
│  ✋ Hands  │ 🧍 Body │  ← Radio toggle (default: Hands)
│                     │
│  📷 Start Camera    │
└─────────────────────┘
```

After camera starts, the card hides. Performance stats (FPS, latency, frame count) appear semi-transparent in the bottom-right corner.

| Mode   | Model                                     | Landmarks      | Skeleton Style               |
|--------|-------------------------------------------|----------------|------------------------------|
| Hands  | `hand_landmarker.task` (float16)          | 21 pts × 2 hands | Green lines, red dots        |
| Body   | `pose_landmarker_lite.task` (float16)     | 33 pts × 1 body  | White lines, cyan-blue dots  |

## Console API (`hl`)

Open DevTools (F12) and use the `hl` global object. All commands work in both Hands and Body modes.

### List Landmarks

```js
hl.list()
// ┌───────┬─────────────────┐
// │ index │ name            │
// ├───────┼─────────────────┤
// │ 0     │ Wrist           │
// │ 1     │ Thumb_CMC       │
// │ ...   │ ...             │
// └───────┴─────────────────┘
```

### Track Landmarks (Console Output)

```js
hl.on('Wrist')                     // track one
hl.on(['Index_TIP', 'Thumb_TIP'])  // track multiple
hl.onAll()                         // track everything
```

Output (every N frames, default N=10):

```
── #140  Hands  2pts ──
  🟢 Left:Wrist              x:0.512 y:0.823 z:-0.103
  🔴 Right:Index_TIP         x:0.401 y:0.765 z:-0.052
```

Color coding by per-coordinate delta vs previous frame:

| Δ range    | Color  |
|------------|--------|
| ≤ 0.003    | Gray   |
| 0.003–0.01 | Yellow |
| 0.01–0.03  | Orange |
| > 0.03     | Red    |

### Track Landmarks (Callback — every frame)

Callbacks fire on **every frame** regardless of throttle setting:

```js
hl.on('Wrist', (d) => {
  // d = { label, x, y, z, dx, dy, dz }
  if (Math.abs(d.dx) > 0.05) {
    console.warn(`${d.label} jumped! Δx=${d.dx.toFixed(4)}`);
  }
});

// Multiple points, same callback
hl.on(['Index_TIP', 'Thumb_TIP'], (d) => {
  // d fires once per tracked point per frame
});
```

Callback data shape:

```ts
{
  label: string;   // e.g. "🟢 Left:Wrist" or "🧍 Left_Shoulder"
  x: number;       // Normalized x (0–1)
  y: number;       // Normalized y (0–1)
  z: number;       // Normalized z (relative depth)
  dx: number;      // Δx from previous frame
  dy: number;      // Δy from previous frame
  dz: number;      // Δz from previous frame
}
```

### Stop Tracking

```js
hl.off('Wrist')    // stop one
hl.offAll()        // stop all
```

### CSV Mode

```js
hl.csv(true)       // enable CSV output
hl.csv(false)      // back to labeled text
```

CSV output format:

```
[CSV] #150
Frame,🟢 Left:Wrist_x,🟢 Left:Wrist_y,🟢 Left:Wrist_z,...
0.5120,0.8230,-0.1030,...
```

Values are color-coded by delta (same scale as text mode).

### Throttle

```js
hl.throttle(1)     // console print every frame (default: 10)
hl.throttle(30)    // every 30 frames
```

Only affects console output — callbacks always fire every frame.

### Status

```js
hl.status()
// Mode: Hands | CSV: OFF | Throttle: 1/10 | FPS: 28 | Frames: 847 | Tracked(2): Wrist, Index_TIP
```

### Help

```js
hl.help()
```

### Hands Mode Landmarks (21)

| # | Name | Description |
|---|------|-------------|
| 0 | Wrist | Wrist center |
| 1–4 | Thumb_CMC, MCP, IP, TIP | Thumb joints |
| 5–8 | Index_MCP, PIP, DIP, TIP | Index finger |
| 9–12 | Middle_MCP, PIP, DIP, TIP | Middle finger |
| 13–16 | Ring_MCP, PIP, DIP, TIP | Ring finger |
| 17–20 | Pinky_MCP, PIP, DIP, TIP | Pinky |

The worker returns `handedness` classification (Left/Right) with confidence score. Left hand is green 🟢, right hand is red 🔴.

### Body Mode Landmarks (33)

Face (0–10): Nose, eyes (inner/center/outer), ears, mouth corners

Upper body (11–22): Shoulders, elbows, wrists, pinkies, indices, thumbs

Lower body (23–32): Hips, knees, ankles, heels, foot indices

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `@mediapipe/tasks-vision` | ^1.0.1 | HandLandmarker, PoseLandmarker, DrawingUtils, FilesetResolver |
| `typescript` | ^5.7 | Dev: type checking |
| `vite` | ^6.0 | Dev: bundler, dev server, worker compilation |

No other runtime dependencies. Models are downloaded once from Google Cloud Storage and cached by the browser.

## Browser Requirements

- WebGL 2.0 (GPU delegate; falls back to CPU automatically)
- Web Workers (module type)
- `navigator.mediaDevices.getUserMedia` (camera)
- `OffscreenCanvas` (for worker rendering)
- `createImageBitmap` (frame extraction)

## Standalone Verification

This project is fully self-contained. To verify:

```bash
# 1. Copy the demo folder anywhere
cp -r demo /tmp/hand-demo
cd /tmp/hand-demo

# 2. Install and run
npm install
npm run dev

# 3. Open http://localhost:5173
```

No files reference the parent `mediapipe-samples-web` project.
