import asyncio, cv2, uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from config import settings
from video_stream import CameraStream
from tracker_engine import TrackerEngine
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.camera = CameraStream(settings.CAMERA_INDEX, 640, 640, 60, 8)
    state.camera.start()
    state.tracker = TrackerEngine([settings.OPENVINO_MODEL_PATH, settings.ONNX_MODEL_PATH, settings.YOLO_MODEL])
    inference_task = asyncio.create_task(inference())
    yield
    if state.camera:
        state.camera.stop()
    inference_task.cancel()
    executor.shutdown(wait=False)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = type('State', (), {'camera': None, 'tracker': None, 'jpeg': None, 'telemetry': {}})()

import time
# Inference loop
async def inference():
    last_time = time.time()
    while True:
        if state.camera and state.tracker:
            frame = state.camera.read()
            if frame is not None:
                try:
                    t0 = time.time()
                    annotated, payload = await asyncio.get_event_loop().run_in_executor(executor, state.tracker.process, frame)
                    t1 = time.time()
                    
                    payload["inference_latency_ms"] = (t1 - t0) * 1000
                    dt = time.time() - last_time
                    payload["fps"] = 1.0 / dt if dt > 0 else 0
                    payload["queue_depth"] = len(state.camera._queue)
                    payload["frames_dropped"] = getattr(state.camera, '_dropped', 0)
                    last_time = time.time()
                    
                    state.telemetry = payload
                    t2 = time.time()
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, getattr(state, "jpeg_quality", 60)])
                    t3 = time.time()
                    payload["jpeg_ms"] = (t3 - t2) * 1000
                    state.jpeg = buf.tobytes()
                except Exception as e:
                    import traceback
                    print("INFERENCE CRASHED:", traceback.format_exc())
        await asyncio.sleep(0.01)



# --- API Endpoints ---
@app.get("/api/health")
async def health(): 
    return {
        "status": "ok",
        "camera_ready": state.camera is not None and state.camera.is_running(),
        "tracker_ready": state.tracker is not None,
        "model": os.path.basename(state.tracker.active_model_path) if state.tracker and hasattr(state.tracker, 'active_model_path') else os.path.basename(settings.YOLO_MODEL),
        "device": "CPU"
    }

@app.get("/api/camera/status")
async def camera_status():
    return {
        "index": settings.CAMERA_INDEX,
        "resolution": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
        "target_fps": settings.TARGET_FPS
    }

@app.get("/api/model/info")
async def model_info(): return {"model_path": os.path.basename(settings.YOLO_MODEL)}

import json
@app.get("/stream/telemetry")
async def telemetry():
    async def gen():
        while True:
            yield f"data: {json.dumps(state.telemetry)}\n\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/stream/video")
async def video():
    async def gen():
        while True:
            if state.jpeg: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + state.jpeg + b'\r\n')
            await asyncio.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if "action" in msg:
                action = msg["action"]
                val = msg.get("value")
                if action == "set_conf" and state.tracker:
                    state.tracker.conf_thresh = float(val)
                elif action == "set_iou" and state.tracker:
                    state.tracker.iou_thresh = float(val)
                elif action == "set_quality":
                    state.jpeg_quality = int(val)
                elif action == "set_brightness" and state.tracker:
                    state.tracker.digital_boost = float(val)
                elif action == "set_trails" and state.tracker:
                    state.tracker.draw_trails = bool(val)
                elif action == "set_tripwire" and state.tracker:
                    state.tracker.draw_tripwire = bool(val)
                elif action == "switch_camera":
                    new_idx = int(val)
                    if state.camera and getattr(state.camera, '_index', None) != new_idx:
                        state.camera.stop()
                        state.camera = CameraStream(new_idx, settings.FRAME_WIDTH, settings.FRAME_HEIGHT, settings.TARGET_FPS, 8)
                        state.camera.start()
                await ws.send_json({"ack": action, "value": val})
    except: pass

# Serve static files last
app.mount("/", StaticFiles(directory=os.path.abspath("../frontend"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT)