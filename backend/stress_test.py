import cv2, time, os, psutil
from tracker_engine import TrackerEngine
from native_ov_engine import NativeOpenVINOEngine

def main():
    print("Initializing TrackerEngine with OpenVINO...")
    engine = TrackerEngine(["yolov8n-seg_openvino_model"])
    
    img = cv2.imread("bus.jpg")
    if img is None:
        print("Failed to load test image.")
        return
    img = cv2.resize(img, (640, 480))
    
    process = psutil.Process(os.getpid())
    print("Initial RAM:", process.memory_info().rss / 1024 / 1024, "MB")
    
    # 15 minutes of video at 10 FPS = 9000 frames
    # Let's run 1000 frames to verify memory stability
    
    latencies = []
    
    print("\nRunning Stress Test (1000 frames)...")
    for i in range(1000):
        t0 = time.perf_counter()
        annotated, payload = engine.process(img.copy())
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
        if (i + 1) % 100 == 0:
            ram = process.memory_info().rss / 1024 / 1024
            print(f"Frame {i+1} - RAM: {ram:.2f} MB - Avg Latency: {sum(latencies[-100:])/100:.2f} ms")
            
    print("\nStress Test Completed.")
    print("Final RAM:", process.memory_info().rss / 1024 / 1024, "MB")
    
    # P95 latency
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"Average Latency: {sum(latencies)/len(latencies):.2f} ms")
    
if __name__ == "__main__":
    main()
