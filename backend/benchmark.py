import time
import cv2
import numpy as np
from ultralytics import YOLO

def benchmark():
    print("Loading model...")
    model = YOLO("yolov8n-seg_openvino_model", task="segment")
    
    # Dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Warming up...")
    for _ in range(5):
        model.predict(frame, verbose=False)
        
    print("Benchmarking 100 frames...")
    start_time = time.time()
    for _ in range(100):
        model.predict(frame, verbose=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / 100) * 1000
    fps = 100 / total_time
    
    print(f"Total time for 100 frames: {total_time:.2f}s")
    print(f"Average Inference Latency: {avg_latency:.2f}ms")
    print(f"Theoretical Max FPS: {fps:.2f}")

if __name__ == "__main__":
    benchmark()
