from ultralytics import YOLO
try:
    # Direct load to check if library and model are healthy
    model = YOLO("yolov8n_openvino_model")
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR FOUND: {e}")