import cv2
import numpy as np
import time
from tracker_engine import TrackerEngine
from native_ov_engine import NativeOpenVINOEngine
from config import settings
import psutil
import os

def generate_test_image(res=640):
    # A scene with a solid background and some colored squares simulating objects
    img = np.ones((res, res, 3), dtype=np.uint8) * 40
    cv2.rectangle(img, (100, 100), (200, 300), (0, 0, 255), -1)
    cv2.rectangle(img, (400, 200), (500, 500), (0, 255, 0), -1)
    cv2.rectangle(img, (250, 50), (350, 150), (255, 0, 0), -1)
    # add some noise
    noise = np.random.randint(0, 20, (res, res, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

def main():
    print("Initializing Engines...")
    model_path = settings.OPENVINO_MODEL_PATH
    
    # 1. Init Production Engine
    prod_engine = TrackerEngine([model_path])
    
    # 2. Init Native Engine
    native_engine = NativeOpenVINOEngine(model_path, infer_res=settings.INFER_RESOLUTION)
    
    # Wait for psutil to settle
    time.sleep(1)
    process = psutil.Process(os.getpid())
    
    img = cv2.imread("bus.jpg")
    if img is None:
        print("Failed to load bus.jpg")
        return
    img = cv2.resize(img, (640, 480))
    
    print("\nWarming up engines...")
    for _ in range(3):
        prod_engine.process(img.copy())
        native_engine.process(img.copy())
        
    print("\n--- BENCHMARK RUN ---")
    num_runs = 20
    
    prod_latency = []
    native_latency = []
    
    print("Running Production (Ultralytics OpenVINO)...")
    for _ in range(num_runs):
        t0 = time.perf_counter()
        annotated, payload = prod_engine.process(img.copy())
        t1 = time.perf_counter()
        prod_latency.append((t1 - t0)*1000)
        
    prod_detections = payload.get("detections", [])
        
    print("Running Native OpenVINO...")
    for _ in range(num_runs):
        t0 = time.perf_counter()
        boxes, scores, classes, masks, timing = native_engine.process(img.copy())
        t1 = time.perf_counter()
        native_latency.append((t1 - t0)*1000)

    avg_prod = np.mean(prod_latency)
    avg_native = np.mean(native_latency)
    
    print(f"\n--- RESULTS ---")
    print(f"Production Latency: {avg_prod:.2f} ms ({1000/avg_prod:.2f} FPS)")
    print(f"Native Latency:     {avg_native:.2f} ms ({1000/avg_native:.2f} FPS)")
    print(f"Improvement:        {avg_prod / avg_native:.2f}x")
    
    print("\n--- PARITY CHECK ---")
    print(f"Production Detections Found: {len(prod_detections)}")
    print(f"Native Detections Found:     {len(boxes)}")
    
    if len(boxes) > 0 and len(prod_detections) > 0:
        # compare top score
        prod_top = prod_detections[0]
        print("\nTop Detection Production:")
        print(f"  Class: {prod_top.get('class_label', prod_top.get('class_name', 'unknown'))}")
        print(f"  Conf:  {prod_top.get('confidence', 0):.4f}")
        print(f"  Bbox:  {prod_top['bbox']}")
        
        print("\nTop Detection Native:")
        print(f"  Class index: {classes[0]}")
        print(f"  Conf:        {scores[0]:.4f}")
        print(f"  Bbox:        {boxes[0]}")
        
    print(f"\nRAM Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    
if __name__ == "__main__":
    main()
