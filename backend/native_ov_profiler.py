import time
import numpy as np
import openvino as ov

def profile_native_ov():
    core = ov.Core()
    # Read the model
    model = core.read_model("yolov8n-seg_openvino_model/yolov8n-seg.xml")
    
    # Compile model for CPU
    compiled_model = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    infer_request = compiled_model.create_infer_request()
    
    # Get input shape
    input_layer = compiled_model.input(0)
    print(f"Input shape: {input_layer.shape}")
    
    # Create dummy input [1, 3, 416, 416]
    dummy_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        infer_request.infer({input_layer.any_name: dummy_input})
        
    # Profile pure OpenVINO runtime
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        infer_request.infer({input_layer.any_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
    avg = np.mean(latencies)
    print(f"Native OpenVINO 416x416 Pure Inference Latency: {avg:.2f} ms")
    print(f"Native OpenVINO FPS: {1000/avg:.2f}")

if __name__ == "__main__":
    profile_native_ov()
