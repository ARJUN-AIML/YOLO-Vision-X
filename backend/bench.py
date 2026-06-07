import time
import cv2
import numpy as np
from tracker_engine import TrackerEngine
from config import settings

print(f"Loading {settings.OPENVINO_MODEL_PATH} at {settings.INFER_RESOLUTION}x{settings.INFER_RESOLUTION}...")
engine = TrackerEngine([settings.OPENVINO_MODEL_PATH])

# Use a real image (black screen, or a simple scene) instead of pure noise
# Pure noise generates 8400 false positives which crushes NMS performance
dummy_frame = np.zeros((settings.INFER_RESOLUTION, settings.INFER_RESOLUTION, 3), dtype=np.uint8)

print("Warming up with empty frame...")
for _ in range(5):
    engine.process(dummy_frame)

print("Running benchmark on empty frame...")
infer_ms = []
post_ms = []
total_ms = []

for _ in range(30):
    annotated, payload = engine.process(dummy_frame)
    timings = payload.get("engine_timing", {})
    infer_ms.append(timings.get("infer_ms", 0))
    post_ms.append(timings.get("post_ms", 0))
    total_ms.append(timings.get("total_ms", 0))

avg_total = np.mean(total_ms)
print(f"Resolution: {settings.INFER_RESOLUTION}x{settings.INFER_RESOLUTION}")
print(f"Total Latency: Avg {avg_total:.2f} ms")
print(f"FPS: Avg {1000/avg_total:.2f}")
print(f"  - Inference Latency: {np.mean(infer_ms):.2f} ms")
print(f"  - Tracker/Mask Rendering: {np.mean(post_ms):.2f} ms")
