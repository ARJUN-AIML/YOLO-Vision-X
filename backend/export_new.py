from ultralytics import YOLO
# Original .pt file-ai vachu pudhu folder create pannuvom
model = YOLO("yolov8n.pt") 
model.export(format="openvino")