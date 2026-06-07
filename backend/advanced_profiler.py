import time
import cv2
import numpy as np
import psutil
import os
from ultralytics import YOLO

def export_if_needed(res):
    model = YOLO("yolov8n-seg.pt", task="segment")
    
    onnx_path = "yolov8n-seg.onnx"
    if not os.path.exists(onnx_path):
        print("Exporting ONNX...")
        try:
            model.export(format="onnx", dynamic=True)
        except Exception as e:
            print(f"ONNX export failed: {e}")

    ov_path_fp32 = f"ov_fp32_{res}"
    if not os.path.exists(ov_path_fp32):
        print(f"Exporting OpenVINO FP32 for {res}...")
        try:
            model.export(format="openvino", half=False, imgsz=res, name=ov_path_fp32)
        except Exception as e:
            print(f"OV FP32 export failed: {e}")

    ov_path_fp16 = f"ov_fp16_{res}"
    if not os.path.exists(ov_path_fp16):
        print(f"Exporting OpenVINO FP16 for {res}...")
        try:
            model.export(format="openvino", half=True, imgsz=res, name=ov_path_fp16)
        except Exception as e:
            print(f"OV FP16 export failed: {e}")

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
        export_if_needed(r)
        
    print("\nStarting Benchmarks...\n")
    print("FORMAT: RESULT|ENGINE|RES|AVG|MEDIAN|P95|P99|FPS|CPU|RAM")
    
    for r in resolutions:
        # PyTorch
        profile_model("yolov8n-seg.pt", r, "PyTorch (FP32)")
        
        # ONNX
        profile_model("yolov8n-seg.onnx", r, "ONNX")
        
        # OV FP32
        # ultralytics export creates folder named ov_fp32_res_openvino_model
        ov32 = f"ov_fp32_{r}_openvino_model"
        if os.path.exists(ov32):
            profile_model(ov32, r, "OpenVINO (FP32)")
            
        # OV FP16
        ov16 = f"ov_fp16_{r}_openvino_model"
        if os.path.exists(ov16):
            profile_model(ov16, r, "OpenVINO (FP16)")
