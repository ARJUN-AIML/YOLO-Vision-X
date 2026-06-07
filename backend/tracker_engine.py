"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRICK 3 — TrackerEngine (Final Production Ready)                            ║
║  File: object-tracking-service/backend/tracker_engine.py                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json, os, base64
import cv2, numpy as np
from collections import deque
from loguru import logger
from ultralytics import YOLO

class TrackerEngine:
    def __init__(self, model_paths: str | list) -> None:
        if isinstance(model_paths, str):
            model_paths = [model_paths]
            
        logger.info(f"Attempting to load models from fallback list: {model_paths}")
        self._model = None
        self.active_model_path = ""
        
        from config import settings
        self.infer_res = settings.INFER_RESOLUTION
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    if "openvino" in path.lower():
                        from native_ov_engine import NativeOpenVINOEngine
                        self._native_engine = NativeOpenVINOEngine(path)
                        self._model = self._native_engine # dummy assignment
                    else:
                        self._model = YOLO(path, task="segment")
                        self._model.predict(np.zeros((self.infer_res, self.infer_res, 3), dtype=np.uint8), verbose=False, imgsz=self.infer_res)
                    logger.success(f"Successfully loaded YOLO engine: {path}")
                    self.active_model_path = path
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    
        if self._model is None:
            fallback = model_paths[-1]
            logger.warning(f"All local engines failed. Falling back to basic load: {fallback}")
            self._model = YOLO(fallback, task="segment")
            self.active_model_path = fallback

        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.digital_boost = 1.0
        self.draw_trails = False
        self.draw_tripwire = False
        self.track_history = {}
        self.snapshot_taken_ids = set()
        
        # Load metadata if it exists
        self.class_names = {}
        metadata_path = os.path.join(self.active_model_path, "metadata.json") if os.path.isdir(self.active_model_path) else self.active_model_path.replace(".pt", "").replace(".onnx", "") + "_metadata.json"
        
        if not os.path.exists(metadata_path) and os.path.isdir(self.active_model_path):
            pass # ignore
        elif os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "names" in meta:
                        self.class_names = {int(k): v for k, v in meta["names"].items()}
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        if not self.class_names:
            self.class_names = getattr(self._model, 'names', {})
            
        if not self.class_names:
            self.class_names = {0: "person", 67: "cell phone", 74: "clock", 73: "notebook / book", 24: "backpack"}

        # Student-focused aliases
        aliases = {
            73: "notebook / book",
            24: "backpack",
            39: "water bottle",
            76: "pen / scissors / stationary",
            63: "laptop",
            67: "mobile phone"
        }
        for k, v in aliases.items():
            if k in self.class_names:
                self.class_names[k] = v

        logger.success("TrackerEngine ready.")

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        if getattr(self, 'digital_boost', 1.0) > 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=self.digital_boost, beta=0)
            
        # Dynamically update predictor arguments since persist=True caches them
        if hasattr(self._model, 'predictor') and self._model.predictor is not None:
            self._model.predictor.args.conf = self.conf_thresh
            self._model.predictor.args.iou = self.iou_thresh

        import time
        t_infer_start = time.perf_counter()
        
        native_boxes, native_scores, native_classes, native_masks, native_timing = [], [], [], [], {}
        if hasattr(self, '_native_engine') and self._native_engine is not None:
            native_boxes, native_scores, native_classes, native_masks, native_timing = self._native_engine.process(frame)
            t_infer_end = time.perf_counter()
            results = None
        else:
            results = self._model.predict(frame, verbose=False, conf=self.conf_thresh, iou=self.iou_thresh, imgsz=self.infer_res)
            t_infer_end = time.perf_counter()

        if not hasattr(self, 'all_snapshots'):
            self.all_snapshots = deque(maxlen=20)
        if not hasattr(self, '_my_tracker'):
            self._my_tracker = {}
            self._next_id = 1
            
        detections = []
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Define and draw Intrusion Zone if enabled
        if getattr(self, 'draw_tripwire', False):
            intrusion_poly = np.array([
                [int(w * 0.3), int(h * 0.3)],
                [int(w * 0.7), int(h * 0.3)],
                [int(w * 0.8), int(h * 0.9)],
                [int(w * 0.2), int(h * 0.9)]
            ], np.int32)
            cv2.polylines(annotated_frame, [intrusion_poly], True, (0, 0, 200), 2)
        else:
            intrusion_poly = None
            
        overlay = annotated_frame.copy()
        
        new_tracker = {}
        
        num_detections = len(native_boxes) if hasattr(self, '_native_engine') and self._native_engine is not None else (len(results[0].boxes) if results and results[0].boxes else 0)
        
        if num_detections > 0:
            masks = None if hasattr(self, '_native_engine') and self._native_engine is not None else (results[0].masks if hasattr(results[0], 'masks') else None)
            
            for idx in range(num_detections):
                if hasattr(self, '_native_engine') and self._native_engine is not None:
                    xyxy = native_boxes[idx]
                    conf = float(native_scores[idx])
                    cls_id = int(native_classes[idx])
                else:
                    box = results[0].boxes[idx]
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                
                cx, cy = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                cls_name = self.class_names.get(cls_id, f"class{cls_id}")
                
                # Match track_id
                best_id = None
                best_dist = 150
                for tid, t_pos in self._my_tracker.items():
                    dist = ((cx - t_pos[0])**2 + (cy - t_pos[1])**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid
                        
                if best_id is not None:
                    track_id = best_id
                    del self._my_tracker[best_id]
                else:
                    track_id = getattr(self, '_next_id', 1)
                    self._next_id = track_id + 1
                    
                new_tracker[track_id] = (cx, cy)
                
                det = {
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "class_label": cls_name
                }
                if track_id is not None:
                    det["tracking_id"] = track_id
                
                # Custom drawing
                x1, y1, x2, y2 = map(int, xyxy)
                
                is_intruding = False
                if intrusion_poly is not None:
                    is_intruding = cv2.pointPolygonTest(intrusion_poly, (cx, cy), False) >= 0
                
                if is_intruding:
                    det["intrusion"] = True
                detections.append(det)
                
                if track_id is not None and track_id not in self.snapshot_taken_ids:
                    import time
                    if not hasattr(self, '_last_snap_time'):
                        self._last_snap_time = {}
                    
                    age = len(self.track_history.get(track_id, []))
                    last_time = self._last_snap_time.get(cls_name, 0)
                    current_time = time.time()
                    
                    # Require object to be tracked for at least 3 frames, and avoid identical class snapshots within 5 seconds (prevents drop-and-reacquire duplicates)
                    if age > 3 and (current_time - last_time) > 5.0:
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            b64_img = base64.b64encode(buffer).decode('utf-8')
                            self.all_snapshots.append({
                                "track_id": track_id,
                                "class_label": cls_name,
                                "image": b64_img
                            })
                            self.snapshot_taken_ids.add(track_id)
                            self._last_snap_time[cls_name] = current_time
                
                if track_id is not None:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=30)
                    self.track_history[track_id].append((cx, cy))
                    
                    if getattr(self, 'draw_trails', False):
                        points = list(self.track_history[track_id])
                        for i in range(1, len(points)):
                            thickness = int(np.sqrt(float(i + 1)) * 1.5)
                            cv2.line(annotated_frame, points[i - 1], points[i], (50, 255, 50), thickness)
                            
                color = (0, 0, 255) if is_intruding else (255, 100, 50) # Red if intruding, Blue otherwise
                
                # Draw Mask Polygon if available, otherwise fallback to bounding box
                has_mask = False
                if hasattr(self, '_native_engine') and self._native_engine is not None:
                    if len(native_masks) > idx:
                        polygon = np.array(native_masks[idx], dtype=np.int32).reshape(-1, 1, 2)
                        if len(polygon) > 2:
                            has_mask = True
                            cv2.fillPoly(overlay, [polygon], color)
                            cv2.polylines(annotated_frame, [polygon], isClosed=True, color=color, thickness=2)
                else:
                    if masks is not None and masks.xy and len(masks.xy) > idx:
                        polygon = np.array(masks.xy[idx], dtype=np.int32)
                        if len(polygon) > 2:
                            has_mask = True
                            cv2.fillPoly(overlay, [polygon], color)
                            cv2.polylines(annotated_frame, [polygon], isClosed=True, color=color, thickness=2)
                
                if not has_mask:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                id_str = f"{track_id}" if track_id is not None else "--"
                label = f"ID:{id_str}  {cls_name}  {int(conf * 100)}%"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw filled rectangle for text background
                cv2.rectangle(annotated_frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                # Draw text in black
                cv2.putText(annotated_frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Cleanup stale tracking data to prevent memory leaks
        active_ids = set(new_tracker.keys())
        stale_ids = [tid for tid in self.track_history.keys() if tid not in active_ids]
        for tid in stale_ids:
            del self.track_history[tid]
            if tid in self.snapshot_taken_ids:
                self.snapshot_taken_ids.remove(tid)
                
        # Update tracker state
        self._my_tracker = new_tracker
        
        # Apply the single overlay blend for all masks + intrusion zone
        if intrusion_poly is not None:
            cv2.fillPoly(overlay, [intrusion_poly], (0, 0, 255))
        cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
        
        t_post_end = time.perf_counter()
        timing_metrics = {
            "infer_ms": (t_infer_end - t_infer_start) * 1000,
            "post_ms": (t_post_end - t_infer_end) * 1000,
            "total_ms": (t_post_end - t_infer_start) * 1000
        }
        
        return annotated_frame, {"detections": detections, "count": len(detections), "snapshots": list(getattr(self, 'all_snapshots', [])), "engine_timing": timing_metrics}