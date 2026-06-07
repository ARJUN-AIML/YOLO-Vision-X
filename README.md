# YOLO Vision X 🚀

A real-time, high-performance object tracking and inference dashboard powered by FastAPI, YOLOv8, and ByteTrack. Designed for seamless edge deployment with a beautiful, fully responsive TailwindCSS frontend.

## 🌟 Features
- **Real-Time Video Streaming**: Ultra-low latency MJPEG streaming with real-time AI bounding box overlays.
- **YOLOv8 Segmentation & Tracking**: Powered by Ultralytics YOLOv8 and robust ByteTrack ID assignments.
- **Dynamic Snapshot Gallery**: Automatically captures, de-duplicates, and displays high-quality snapshots of detected objects with a one-click local save option.
- **Live Telemetry & Diagnostics**: Real-time server health and performance monitoring via Server-Sent Events (SSE).
- **Mobile Optimized**: Fully responsive interface with device-specific hardware controls.

---

## 🎛 Dashboard Controls & Configuration

The dashboard provides a powerful control panel to tweak inference and camera settings on the fly without restarting the server.

### Inference Config
* **Confidence Threshold (0.05 - 0.95)**: Filters out detections below the chosen confidence score. Higher values reduce false positives; lower values detect objects that are harder to see or occluded.
* **IoU NMS Threshold (0.05 - 0.95)**: Controls the Intersection over Union for Non-Maximum Suppression. Lower values aggressively merge overlapping bounding boxes for the same object.
* **MJPEG Quality (20 - 95)**: Adjusts the compression quality of the live video stream sent to your browser. Lower to `20` for fast streaming over poor networks, or raise to `95` for crystal clear visuals.
* **CAMERA SOURCE / SWITCH CAMERA (Mobile Only)**: Seamlessly toggle between front, rear, and wide-angle cameras on mobile devices.
* **Digital Boost Box (1.0x - 3.0x)**: Artificially enhances the brightness and contrast of the video feed *before* running it through the AI. Excellent for low-light, night vision, or infrared environments.
* **Movement Trails (0 / 1)**: Toggles visual path trails behind moving objects to track their trajectory and speed over time.
* **Intrusion Zone (0 / 1)**: Draws a customizable virtual boundary on the camera feed. Objects that enter this tripwire zone will trigger an intrusion alert and turn their bounding box red.
* **APPLY**: Sends the updated configuration directly to the backend AI engine over WebSockets for instant application.

### System Diagnostics
The diagnostics panel gives you a real-time health check of the backend AI engine:
* **Camera Index**: The hardware ID or stream URL of the active camera being processed.
* **Actual Capture FPS**: The raw hardware frame capture rate directly from the camera sensor.
* **Frames Dropped**: The number of frames discarded because the AI queue was full. A high number indicates your hardware is bottlenecking and cannot keep up with the camera's capture rate.
* **Queue Depth**: The current number of frames sitting in memory waiting to be processed by the YOLO model.
* **Total Detections**: The lifetime number of objects successfully detected and tracked since the server started.
* **Avg Infer (ms)**: The average time taken by the AI to process a single frame (Pre-process + Inference + Post-process).
* **WS Connection**: Real-time status of the bidirectional WebSocket control channel (`connected` / `disconnected` / `error`).
* **SSE Connection**: Real-time status of the Server-Sent Events telemetry channel (`connected` / `error`).
* **↻ Refresh Health Check**: Manually pings the backend to verify the API is alive and forces a refresh of the diagnostic data.

---

## 🚀 Deployment Instructions

> **⚠️ Note on Vercel Deployment:**
> Vercel is a Serverless platform. While you can deploy the `frontend` folder to Vercel, the **Python backend will not work on Vercel**. The backend requires persistent WebSockets, continuous camera access, and heavy GPU/CPU compute loops, meaning it must be deployed on a VPS (like Render, Railway, AWS EC2, or DigitalOcean) or run locally. If you deploy the frontend to Vercel, make sure to update the backend URLs in `app.js` to point to your live backend server.

### Running Locally
1. Clone the repository.
2. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn ultralytics opencv-python-headless loguru
   ```
3. Navigate to the `backend` directory and start the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Open your browser and navigate to `http://localhost:8000` to view the dashboard!
