import time
import numpy as np
import cv2
import warnings
import os
import shutil
warnings.filterwarnings('ignore')

from config import settings

def run_benchmark(resolution):
    print(f"\n--- BENCHMARKING {resolution}x{resolution} ---")
    settings.INFER_RESOLUTION = resolution
    
    target_dir = "yolov8n-seg_openvino_model"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    print(f"Exporting OpenVINO model at {resolution}...")
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt", task="segment")
    model.export(format='openvino', imgsz=resolution)
    
    # Initialize Tracker Engine
    from tracker_engine import TrackerEngine
    engine = TrackerEngine([target_dir])
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        engine.process(dummy_frame)
        
    # Benchmark
    print("Running benchmark...")
    infer_ms = []
    post_ms = []
    total_ms = []
    jpeg_ms = []
    
    for _ in range(30):
        annotated, payload = engine.process(dummy_frame)
        
        timings = payload.get("engine_timing", {})
        infer_ms.append(timings.get("infer_ms", 0))
        post_ms.append(timings.get("post_ms", 0))
        total_ms.append(timings.get("total_ms", 0))
        
        # Test JPEG
        t0 = time.perf_counter()
        cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        t1 = time.perf_counter()
        jpeg_ms.append((t1 - t0) * 1000)
        
    # Calculate Results
    avg_total = np.mean(total_ms) + np.mean(jpeg_ms)
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Total Latency: Avg {avg_total:.2f} ms")
    print(f"FPS: Avg {1000/avg_total:.2f}")
    print(f"  - Inference Latency: {np.mean(infer_ms):.2f} ms")
    print(f"  - Tracker/Mask Rendering: {np.mean(post_ms):.2f} ms")
    print(f"  - JPEG Encoding: {np.mean(jpeg_ms):.2f} ms")


if __name__ == "__main__":
    run_benchmark(640)
    run_benchmark(512)
    run_benchmark(416)
