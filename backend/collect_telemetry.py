import urllib.request
import json
import numpy as np

def run(duration=10):
    url = "http://127.0.0.1:8000/stream/telemetry"
    req = urllib.request.Request(url)
    
    metrics = {
        "fps": [],
        "latency": [],
        "infer_ms": [],
        "post_ms": [],
        "jpeg_ms": []
    }
    
    with urllib.request.urlopen(req) as response:
        for _ in range(duration * 2): # Just sample roughly duration number of updates
            line = response.readline()
            if not line:
                break
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = json.loads(line[6:])
                metrics["fps"].append(data.get("fps", 0))
                metrics["latency"].append(data.get("inference_latency_ms", 0))
                metrics["jpeg_ms"].append(data.get("jpeg_ms", 0))
                
                engine_timing = data.get("engine_timing", {})
                metrics["infer_ms"].append(engine_timing.get("infer_ms", 0))
                metrics["post_ms"].append(engine_timing.get("post_ms", 0))

                if len(metrics["fps"]) >= 100:
                    break

    print("--- RAW TELEMETRY DATA (Last 100 frames) ---")
    print("Dashboard FPS Pipeline: ", [round(x, 1) for x in metrics['fps'][-10:]], "... (last 10)")
    print("Inference Latency: ", [round(x, 1) for x in metrics['latency'][-10:]], "... (last 10)")
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total Latency: Avg {np.mean(metrics['latency']):.2f} ms | P95 {np.percentile(metrics['latency'], 95):.2f} ms")
    print(f"FPS: Avg {np.mean(metrics['fps']):.2f}")
    print(f"  - Inference Latency: {np.mean(metrics['infer_ms']):.2f} ms")
    print(f"  - Tracker/Mask Rendering: {np.mean(metrics['post_ms']):.2f} ms")
    print(f"  - JPEG Encoding: {np.mean(metrics['jpeg_ms']):.2f} ms")
    
if __name__ == "__main__":
    run()
