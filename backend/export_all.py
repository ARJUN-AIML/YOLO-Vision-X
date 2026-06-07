import os
import torch
# Patch torch.load for PyTorch 2.6+
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt", task="segment")
# Export FP16
print("Exporting FP16...")
model.export(format="openvino", half=True, imgsz=416, int8=False)
# Export INT8
print("Exporting INT8...")
model.export(format="openvino", half=False, imgsz=416, int8=True)
