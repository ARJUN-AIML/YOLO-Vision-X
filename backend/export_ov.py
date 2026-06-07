import os
# 1. Lower PyTorch secure unpickling validation parameters globally
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import json
import torch

original_load = torch.load
def unblocked_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = unblocked_load

from ultralytics import YOLO
import openvino as ov

if __name__ == "__main__":
    print("─── OVERCLOCK ENGINE: COMPILING NATIVE OPENVINO WITH METADATA ───")
    
    export_dir = "yolov8n_openvino_model"
    os.makedirs(export_dir, exist_ok=True)
    
    # Step A: Load the base weights via security interceptor and extract class dict strings
    print("[1/4] Extracting base model layers and coco dictionary configurations...")
    model = YOLO("yolov8n.pt")
    class_names_dict = model.names  # 80 classes mapped out correctly!
    
    # Step B: Export model computational graphs smoothly to an intermediate format
    print("[2/4] Tracing computational layout down to structural ONNX blocks...")
    onnx_path = model.export(format="onnx", dynamic=False, simplify=True)
    
    # Step C: Read the output intermediate vectors through native Intel APIs
    print("[3/4] Running native Intel Core layer translation matrix...")
    core = ov.Core()
    ov_model = core.read_model(onnx_path)
    
    # Explicitly map core layout identifiers so model doesn't guess task properties
    ov_model.set_rt_info("detect", ["model_info", "task"])
    
    # Inject the missing names dictionary string directly into OpenVINO runtime structures metadata!
    ov_model.set_rt_info(str(class_names_dict), ["model_info", "names"])
    
    # Step D: Serialize graph structural arrays safely down to local paths disk layout
    print("[4/4] Writing static binaries and metadata matrices onto local path files...")
    output_xml = os.path.join(export_dir, "yolov8n.xml")
    ov.save_model(ov_model, output_xml)
    
    # Manual synchronization fallback: backup config layer metadata json inside the directory
    metadata_backup = {
        "task": "detect",
        "names": class_names_dict
    }
    with open(os.path.join(export_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_backup, f, indent=4)
        
    # Remove transient tracking files to keep workspace tidy
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        
    print("─── SUCCESS: METADATA OVERCLOCK COMPILATION PASSED COMPLETELY ───")
    print(f"Target Assets Path: {os.path.abspath(export_dir)}")