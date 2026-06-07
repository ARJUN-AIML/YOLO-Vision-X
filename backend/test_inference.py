import cv2
from ultralytics import YOLO

def test():
    model = YOLO(r"D:\object-tracking-service\object-tracking-service\backend\yolov8n-seg_openvino_model")
    frame = cv2.imread(r"D:\object-tracking-service\object-tracking-service\frontend\assets\person.jpg")
    if frame is None:
        frame = cv2.imread(r"D:\object-tracking-service\object-tracking-service\frontend\assets\sample.jpg")
    if frame is None:
        import numpy as np
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    print("Running inference...")
    try:
        results = model.track(frame, persist=True, verbose=False, conf=0.25, iou=0.45, tracker="bytetrack.yaml", imgsz=320, half=True)
        print("Inference successful!")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test()
