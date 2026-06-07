import time
import cv2
import numpy as np
from ultralytics import YOLO

def profile_model(model_path, task, resolutions):
    try:
        print(f"\n{'='*50}\nProfiling Model: {model_path} (Task: {task})\n{'='*50}")
        model = YOLO(model_path, task=task)
        
        for res in resolutions:
            w, h = res
            print(f"\n--- Resolution: {w}x{h} ---")
            
            # Dummy frame
            raw_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            
            # Warmup
            for _ in range(3):
                model.predict(raw_frame, verbose=False)
            
            times = {
                "prep": [],
                "infer": [],
                "post": [],
                "render": [],
                "encode": []
            }
            
            for _ in range(50):
                # 1. Prep
                t0 = time.perf_counter()
                frame = cv2.resize(raw_frame, (w, h))
                t1 = time.perf_counter()
                
                # 2. Infer
                results = model.predict(frame, verbose=False, imgsz=w)
                t2 = time.perf_counter()
                
                # 3. Post (masks, boxes, tracking sim)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    if hasattr(results[0], 'masks') and results[0].masks is not None:
                        masks = results[0].masks.xy
                    else:
                        masks = None
                t3 = time.perf_counter()
                
                # 4. Render
                annotated = frame.copy()
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if masks and len(masks) > i:
                            poly = np.array(masks[i], dtype=np.int32)
                            cv2.polylines(annotated, [poly], True, (255, 0, 0), 2)
                t4 = time.perf_counter()
                
                # 5. Encode
                _, _ = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
                t5 = time.perf_counter()
                
                times["prep"].append((t1 - t0) * 1000)
                times["infer"].append((t2 - t1) * 1000)
                times["post"].append((t3 - t2) * 1000)
                times["render"].append((t4 - t3) * 1000)
                times["encode"].append((t5 - t4) * 1000)
                
            total_times = [sum(x) for x in zip(times["prep"], times["infer"], times["post"], times["render"], times["encode"])]
            
            for stage in ["prep", "infer", "post", "render", "encode"]:
                arr = np.array(times[stage])
                avg = np.mean(arr)
                p95 = np.percentile(arr, 95)
                pct = (avg / np.mean(total_times)) * 100
                print(f"{stage.upper():<10} | Avg: {avg:>6.2f}ms | P95: {p95:>6.2f}ms | {pct:>5.1f}%")
                
            avg_total = np.mean(total_times)
            fps = 1000.0 / avg_total if avg_total > 0 else 0
            print(f"TOTAL      | Avg: {avg_total:>6.2f}ms | Est FPS: {fps:>5.1f}")

    except Exception as e:
        print(f"Error profiling {model_path}: {e}")

if __name__ == "__main__":
    resolutions = [(416, 416), (512, 512), (640, 640)]
    models = [
        ("yolov8n.pt", "detect"),
        ("yolov8n-seg.pt", "segment"),
        ("yolov8n_openvino_model", "detect"),
        ("yolov8n-seg_openvino_model", "segment")
    ]
    
    for path, task in models:
        profile_model(path, task, resolutions)
