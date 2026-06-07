# YOLO Vision X — Real-Time Edge Object Tracking

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="YOLO Vision X Dashboard" width="100%">
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api">API Reference</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="OpenVINO" src="https://img.shields.io/badge/OpenVINO-2024.x-0071C5?style=flat-square&logo=intel&logoColor=white">
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8n--seg-Ultralytics-FF6F00?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

## Overview

**YOLO Vision X** is a production-grade, real-time object detection and segmentation tracking system built for deployment on CPU-only edge hardware. It combines Intel's **OpenVINO inference runtime** with the **YOLOv8n-seg** segmentation model, a custom centroid-based multi-object tracker, and a live telemetry dashboard delivered entirely over server-sent events (SSE) and WebSockets.

The system detects and segments 80 COCO object classes in real time, assigns persistent tracking IDs to each object across frames, and streams annotated video, telemetry metrics, and structured event logs to any browser without additional plugins or client-side SDKs.

Designed as a portfolio-grade demonstration of modern edge AI architecture, the project emphasises low-latency inference, a stable professional UI, and clean production engineering practices.

---

## Features

### Detection & Segmentation
- Full **YOLOv8n-seg** instance segmentation with pixel-accurate mask overlays
- All **80 COCO class labels** active (person, chair, bottle, cup, TV, clock, cell phone, and more)
- Configurable confidence and IoU thresholds, adjustable at runtime via WebSocket control channel

### Object Tracking
- **Centroid-based multi-object tracker** with Euclidean-distance nearest-neighbour matching
- Persistent track IDs assigned per unique object
- Track history with optional movement trail rendering
- Intrusion Zone detection: configurable polygon region with real-time alert overlay

### Live Dashboard
- **SSE-streamed telemetry** updating the UI at ~20Hz without polling overhead
- FPS gauge, inference latency readout, object count, and queue-depth diagnostics
- Low-pass exponential smoothing filter on FPS and latency for stable display values
- Hardware badge (CPU/GPU) and model identification header

### Track Registry
- **Permanent event registry** maintaining up to 1,000 unique track entries
- Per-entry columns: **ID · CLASS · STATUS (Active / Lost) · FIRST SEEN · LAST SEEN**
- Real-time summary counters: Total Seen, Active, Lost
- Fixed-height container with internal scrollbar — zero layout shift during operation
- `CLR` button clears the registry in-memory and updates all counters immediately
- Optimised rendering via `DocumentFragment` — no full DOM rebuilds per frame

### Snapshot Gallery
- Automatic snapshot captured when a new unique object is tracked for >3 frames
- Per-class cooldown (5 seconds) prevents duplicate captures on ID reassignment
- Snapshots displayed as a 12-card rolling gallery with hover-to-download actions
- Base64-encoded JPEGs embedded directly in telemetry stream — no separate file storage

### CSV Export
- Exports the complete in-memory Track Registry to CSV on demand
- Columns: `ID, CLASS, STATUS, FIRST_SEEN, LAST_SEEN`
- Uses Blob `URL.createObjectURL` — no server round-trip, no browser memory blowout

### Camera Switching
- Safe **hardware-validation-first** camera switching
- Backend probes the requested index via `VideoCapture.isOpened()` and a live frame grab before switching
- Active camera stream is **never destroyed** until the new stream is validated
- Returns `"Camera index unavailable"` warning to the client on failure — no black screen

### Digital Enhancement Controls
- **Digital Boost** (brightness amplifier, `cv2.convertScaleAbs`)
- **Movement Trails** (centroid path history overlay)
- **Intrusion Zone** (polygon region, real-time containment check)
- **JPEG Quality** slider (controls stream bandwidth vs. visual fidelity)
- All controls wired directly to the inference pipeline — zero placebo controls

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser Client                        │
│                                                              │
│   ┌──────────────┐  SSE   ┌──────────────┐                  │
│   │  Video Feed  │◄──────►│  Telemetry   │                  │
│   │  MJPEG Stream│        │  Dashboard   │                  │
│   └──────────────┘        └──────────────┘                  │
│                            WS ▲                              │
│                               │  Control Channel            │
└──────────────────────────────┼──────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────┐
│                        FastAPI Backend                       │
│                                                              │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐  │
│  │  CameraStream │──►│ TrackerEngine │──►│   main.py     │  │
│  │  (Thread)     │   │               │   │  (async loop) │  │
│  └───────────────┘   └───────┬───────┘   └───────────────┘  │
│                              │                               │
│                    ┌─────────▼──────────┐                   │
│                    │ NativeOpenVINOEngine│                   │
│                    │  (OpenVINO Runtime) │                   │
│                    └────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow:**

1. `CameraStream` captures frames in a dedicated daemon thread, writing to a bounded `deque(maxlen=8)` queue.
2. The async inference loop in `main.py` dequeues the latest frame via `ThreadPoolExecutor(max_workers=1)`.
3. `TrackerEngine` delegates inference to `NativeOpenVINOEngine`, which runs synchronised OpenVINO IR inference and returns bounding boxes, confidence scores, class IDs, and segmentation mask polygons.
4. The centroid tracker matches detections to existing track IDs by minimum Euclidean distance (threshold: 150px).
5. The annotated frame is JPEG-encoded and pushed to `state.jpeg` for MJPEG streaming.
6. The structured telemetry payload (detections, snapshots, timing metrics) is serialised to JSON and streamed to all SSE consumers.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference Runtime | Intel OpenVINO 2024.x |
| Detection Model | YOLOv8n-seg (Ultralytics export → OpenVINO IR) |
| Backend Framework | FastAPI + Uvicorn |
| Async I/O | Python asyncio + ThreadPoolExecutor |
| Video Capture | OpenCV (cv2.VideoCapture, DSHOW backend on Windows) |
| Real-Time Streaming | Server-Sent Events (SSE) + MJPEG over HTTP |
| Control Channel | WebSocket (bidirectional, with send-queue and exponential backoff reconnect) |
| Frontend Framework | Vanilla JavaScript (ES2022, strict mode, class-based OO architecture) |
| UI Styling | Tailwind CSS |
| Mask Rendering | OpenCV `cv2.findContours` + `cv2.fillPoly` + alpha blend |
| Logging | Loguru |
| Configuration | Pydantic Settings (`config.py`) |

---

## Performance Optimizations

The following optimizations were applied iteratively to reduce end-to-end latency from ~270ms to ~132ms:

1. **Native OpenVINO Engine** — Replaced the Ultralytics `model.predict()` wrapper with a direct OpenVINO Core `compile_model()` → `infer_request.infer()` pipeline, eliminating Python-level overhead from the Ultralytics postprocessing chain.

2. **Contour Polygon Extraction** — Instead of rendering full binary segmentation masks `(H, W)` through `fillPoly`, the engine calls `cv2.findContours()` on each mask slice and returns compact polygon coordinates `(-1, 2)`, cutting memory overhead significantly.

3. **Bounded Frame Queue** — `CameraStream` maintains a `deque(maxlen=8)`, popping and clearing on each read to always serve the freshest available frame, avoiding stale frame accumulation.

4. **Single Overlay Blend** — All segmentation masks and the intrusion zone are accumulated into a single `overlay` array and blended with one `cv2.addWeighted()` call rather than per-object blending passes.

5. **JPEG Quality Slider** — Runtime-adjustable JPEG encoding quality (`cv2.IMWRITE_JPEG_QUALITY`) lets operators trade bandwidth for visual fidelity without restarting the pipeline.

6. **SSE at 20Hz** — The telemetry generator yields every 50ms. The frontend uses `requestAnimationFrame`-gated rendering to avoid reflow storms between telemetry ticks.

7. **DocumentFragment DOM Updates** — The Track Registry renders all rows into a detached `DocumentFragment` before a single `innerHTML` swap, reducing browser reflow/repaint cycles.

---

## Performance Benchmarks

Tested on Intel Core i7-10th Gen, 16GB RAM, integrated GPU, Windows 11.

| Pipeline Version | End-to-End Latency | Effective FPS |
|---|---|---|
| Original Ultralytics (ONNX fallback) | ~270 ms | ~3.7 FPS |
| Native OpenVINO Engine | ~132 ms | ~7.5 FPS |

> **Note:** FPS figures represent true end-to-end inference FPS measured as `1 / (t_postprocess - t_infer_start)`, inclusive of postprocessing, mask rendering, and JPEG encoding. UI display FPS may differ as it reflects telemetry stream frequency, not inference cadence.

---

## Screenshots

> *(Replace placeholders with actual screenshots after deployment)*

| Dashboard Overview | Track Registry | Snapshot Gallery |
|---|---|---|
| `docs/screenshots/dashboard.png` | `docs/screenshots/registry.png` | `docs/screenshots/snapshots.png` |

---

## Installation

### Prerequisites

- Python 3.11+
- Git
- A connected webcam (index 0)
- Windows 10/11 (Linux supported with `CAP_ANY` backend — remove `CAP_DSHOW` flag in `video_stream.py`)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/object-tracking-service.git
cd object-tracking-service/object-tracking-service
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv backend/.venv
# Windows
backend\.venv\Scripts\activate
# Linux / macOS
source backend/.venv/bin/activate
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Export the YOLOv8 Model to OpenVINO IR

If you do not already have the exported model:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt').export(format='openvino', imgsz=640)"
```

This produces `yolov8n-seg_openvino_model/` in the current directory. Move or symlink it to `backend/models/`.

### 5. Verify Configuration

Edit `backend/config.py` to confirm:

```python
OPENVINO_MODEL_PATH = "models/yolov8n-seg_openvino_model"
CAMERA_INDEX        = 0
INFER_RESOLUTION    = 640
```

---

## Usage

### Start the Backend

```bash
cd backend
python main.py
```

The server starts at `http://127.0.0.1:8000`.

### Open the Dashboard

Navigate to `http://127.0.0.1:8000/` in any modern browser (Chrome, Edge, Firefox recommended).

### Runtime Controls

All controls apply changes instantly via the WebSocket control channel — no page reload required.

| Control | Effect |
|---|---|
| Confidence Threshold | Filters detections below the set confidence level |
| IoU Threshold | Adjusts Non-Maximum Suppression aggressiveness |
| JPEG Quality | Trades stream bandwidth for visual fidelity |
| Digital Boost | Amplifies frame brightness before inference |
| Movement Trails | Renders centroid path history as overlay lines |
| Intrusion Zone | Draws a fixed polygon; detections inside are highlighted red |
| Camera Switch | Validates new camera hardware before switching; preserves current stream on failure |

---

## Folder Structure

```
object-tracking-service/
├── backend/
│   ├── main.py              # FastAPI application, inference loop, WebSocket control
│   ├── tracker_engine.py    # TrackerEngine: centroid tracker + annotation pipeline
│   ├── native_ov_engine.py  # NativeOpenVINOEngine: direct OpenVINO IR inference
│   ├── video_stream.py      # CameraStream: threaded frame capture with bounded queue
│   ├── config.py            # Pydantic settings (model paths, resolution, FPS)
│   ├── requirements.txt
│   └── models/
│       └── yolov8n-seg_openvino_model/
│           ├── yolov8n-seg.xml
│           ├── yolov8n-seg.bin
│           └── metadata.yaml
└── frontend/
    ├── index.html           # Main production dashboard
    ├── app.js               # VisionDashboardApp orchestrator (class-based OO)
    ├── edge.html            # Browser-only ONNX prototype (experimental)
    └── edge.js              # Browser-side ONNX Runtime + JS tracker
```

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | System health: camera state, tracker state, active model |
| `GET` | `/api/camera/status` | Camera index, resolution, target FPS |
| `GET` | `/api/model/info` | Active model filename |
| `GET` | `/stream/video` | MJPEG annotated video stream |
| `GET` | `/stream/telemetry` | SSE telemetry stream (JSON payload, ~20Hz) |
| `WS` | `/ws/control` | WebSocket control channel for runtime configuration |

### Telemetry Payload Schema

```jsonc
{
  "detections": [
    {
      "box":         [x1, y1, x2, y2],
      "confidence":  0.87,
      "class_label": "person",
      "tracking_id": 3,
      "intrusion":   false      // present only if Intrusion Zone is active
    }
  ],
  "count":                  2,
  "fps":                    7.4,
  "inference_latency_ms":   131.8,
  "queue_depth":            2,
  "frames_dropped":         0,
  "engine_timing": {
    "infer_ms":   98.2,
    "post_ms":    33.6,
    "total_ms":   131.8
  },
  "snapshots": [
    {
      "track_id":    3,
      "class_label": "person",
      "image":       "<base64-encoded JPEG>"
    }
  ]
}
```

### WebSocket Control Messages

Send JSON to `/ws/control`:

```jsonc
{ "action": "set_conf",       "value": 0.45 }
{ "action": "set_iou",        "value": 0.50 }
{ "action": "set_quality",    "value": 70   }
{ "action": "set_brightness", "value": 1.3  }
{ "action": "set_trails",     "value": true }
{ "action": "set_tripwire",   "value": true }
{ "action": "switch_camera",  "value": 1    }
```

Responses:

```jsonc
{ "ack": "set_conf", "value": 0.45 }           // success
{ "warning": "Camera index unavailable" }        // camera switch failure
```

---

## Track Registry

The Track Registry is a **permanent in-memory event log** of every unique object detected since the dashboard was last cleared.

**State Machine per Entry:**
- `Active` — tracking ID was present in the most recent telemetry frame
- `Lost` — tracking ID was not present in the most recent telemetry frame (but remains in registry)

**Columns:** `ID · CLASS · STATUS · FIRST SEEN · LAST SEEN`

**Counters:** `Total Seen · Active · Lost` update in real time.

**Capacity:** Capped at 1,000 entries. When the cap is exceeded, the oldest entries by `first_seen` timestamp are removed. This prevents unbounded browser memory growth during extended operation.

**Rendering:** Built using `DocumentFragment` for efficient batched DOM updates. No full DOM rebuild occurs per telemetry frame — only a single `innerHTML` swap after the fragment is fully constructed.

**CLR Button:** Clears the in-memory Map entirely and resets all counters.

---

## Snapshot Gallery

The Snapshot Gallery automatically captures a cropped JPEG of a detected object under the following conditions:

1. The object has been continuously tracked for more than **3 consecutive frames** (avoids spurious one-shot detections)
2. No snapshot has been taken for this **class label within the last 5 seconds** (prevents duplicate captures during brief ID reassignment events)

Snapshots are embedded as base64-encoded JPEGs directly in the telemetry stream. The frontend renders a rolling gallery of up to **12 cards**. Each card displays the track ID, class label, and a hover-to-download button that saves the image as a local JPEG without any server round-trip.

---

## CSV Export

The CSV Export button triggers a client-side download of the complete Track Registry. Data is sourced directly from the in-memory `Map` — not scraped from the DOM — ensuring the export is always complete and accurate regardless of the scroll position or visible rows.

**CSV Columns:** `ID, CLASS, STATUS, FIRST_SEEN, LAST_SEEN`

The export uses `Blob` + `URL.createObjectURL()` for memory-safe downloads with no server involvement.

---

## Camera Switching

Camera switching implements a **validate-before-commit** protocol:

1. The requested camera index is opened with `cv2.VideoCapture(new_index)` in a **temporary test context**.
2. A single frame is grabbed to confirm the hardware is physically producing frames.
3. Only if both checks pass is the active camera stream stopped and the new stream started.
4. If either check fails, the active stream is preserved without interruption, and the client receives a `{"warning": "Camera index unavailable"}` message surfaced in the System Log.

This guarantees the dashboard never displays a black screen due to a failed camera switch request.

---

## Mobile Support

The dashboard layout uses Tailwind CSS responsive grid classes targeting `md` and `lg` breakpoints. On mobile viewports:

- The camera switcher UI is conditionally visible only on mobile devices (detected via `navigator.userAgent`)
- Fixed-height containers (`Track Registry`, `Class Histogram`, `System Log`) prevent layout shifts when content grows
- All interactive controls remain accessible via vertical scroll

---

## Known Limitations

### 1. Single-Camera Architecture
The backend maintains a single global camera stream. Multi-camera support would require a connection manager pool replacing the current global state pattern.

### 2. Centroid Tracking Limitations
The tracker uses Euclidean distance between detection centroids to match objects across frames (threshold: 150px). This approach is computationally lightweight but has known weaknesses:
- **Occlusion:** If two objects cross paths, their IDs may swap or a new ID may be assigned when they separate.
- **Fast Motion:** Objects moving more than 150px between inference frames will lose their ID and receive a new one.
- **ID Fragmentation:** Brief disappearances (e.g. object partially out of frame) generate a new track ID on re-entry.

### 3. Possible ID Reassignment Under Heavy Occlusion
In scenes with many overlapping objects (e.g. crowded rooms), the centroid matcher may incorrectly associate a detection with the nearest previous centroid rather than the correct object, leading to ID swaps. This is a fundamental property of nearest-neighbour tracking without appearance features.

### 4. CPU-Only Inference at ~7.5 FPS
The pipeline is optimised for CPU inference via OpenVINO. GPU acceleration (via OpenVINO GPU plugin or CUDA) is not configured by default and would require additional driver setup.

### 5. No Authentication
The WebSocket control channel and all API endpoints are unauthenticated. For deployment beyond a local network, a reverse proxy with authentication (e.g. Nginx + JWT) is strongly recommended.

---

## Future Improvements

- [ ] **Kalman Filter Tracker** — Replace centroid matching with a Kalman-filter-based tracker for improved occlusion robustness and motion prediction
- [ ] **Multi-Camera Support** — Refactor global state to a connection manager pool supporting N simultaneous camera streams
- [ ] **GPU Inference** — Enable OpenVINO GPU plugin for discrete and integrated GPU acceleration
- [ ] **Authentication Layer** — Add JWT-gated WebSocket and SSE endpoints
- [ ] **Object Re-identification** — Integrate lightweight appearance features (e.g. colour histograms) to improve ID persistence across occlusion events
- [ ] **Alert System** — Configurable webhook or notification trigger when a specific class enters the Intrusion Zone
- [ ] **Recording Mode** — On-demand video recording with annotated overlays to disk

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with OpenVINO · FastAPI · YOLOv8 · JavaScript
</p>
