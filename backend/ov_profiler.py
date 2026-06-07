import time
import cv2
import numpy as np
import psutil
import os
import shutil
from ultralytics import YOLO

def export_ov(res):
    model = YOLO("yolov8n-seg.pt", task="segment")
    target_dir = f"ov_{res}"
    
    if not os.path.exists(target_dir):
        print(f"Exporting OpenVINO FP32 for {res}...")
        try:
            # ultralytics forces name to yolov8n-seg_openvino_model
            if os.path.exists("yolov8n-seg_openvino_model"):
                shutil.rmtree("yolov8n-seg_openvino_model")
            
            model.export(format="openvino", half=False, imgsz=res)
            
            # rename to target_dir
            if os.path.exists("yolov8n-seg_openvino_model"):
                shutil.move("yolov8n-seg_openvino_model", target_dir)
        except Exception as e:
            print(f"OV export failed: {e}")

def profile_model(model_path, res, engine_name):
    try:
        model = YOLO(model_path, task="segment")
        w, h = res, res
        print(f"Running {engine_name} at {w}x{h}...")
        
        raw_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            model.predict(raw_frame, verbose=False, imgsz=res)
            
        latencies = []
        cpu_usages = []
        ram_usages = []
        
        process = psutil.Process(os.getpid())
        
        for _ in range(50):
            cpu_before = psutil.cpu_percent(interval=None)
            
            t0 = time.perf_counter()
            model.predict(raw_frame, verbose=False, imgsz=res)
            t1 = time.perf_counter()
            
            cpu_after = psutil.cpu_percent(interval=None)
            ram = process.memory_info().rss / (1024 * 1024)
            
            latencies.append((t1 - t0) * 1000)
            cpu_usages.append((cpu_before + cpu_after) / 2.0)
            ram_usages.append(ram)
            
        arr = np.array(latencies)
        avg = np.mean(arr)
        median = np.median(arr)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        fps = 1000.0 / avg if avg > 0 else 0
        
        avg_cpu = np.mean(cpu_usages)
        avg_ram = np.mean(ram_usages)
        
        print(f"RESULT|{engine_name}|{res}|{avg:.2f}|{median:.2f}|{p95:.2f}|{p99:.2f}|{fps:.2f}|{avg_cpu:.1f}%|{avg_ram:.0f}MB")
        
    except Exception as e:
        print(f"Error profiling {engine_name} at {res}: {e}")

if __name__ == "__main__":
    resolutions = [416, 512, 640]
    
    print("Preparing models...")
    for r in resolutions:
        export_ov(r)
        
    print("\nStarting OpenVINO Benchmarks...\n")
    print("FORMAT: RESULT|ENGINE|RES|AVG|MEDIAN|P95|P99|FPS|CPU|RAM")
    
    for r in resolutions:
        ov_dir = f"ov_{r}"
        if os.path.exists(ov_dir):
            profile_model(ov_dir, r, "OpenVINO (FP32)")
