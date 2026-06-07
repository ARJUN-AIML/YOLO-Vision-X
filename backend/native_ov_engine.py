import cv2
import numpy as np
import openvino as ov
import time

class NativeOpenVINOEngine:
    def __init__(self, model_path, infer_res=416, conf_thresh=0.25, iou_thresh=0.45):
        self.infer_res = infer_res
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        self.core = ov.Core()
        model = self.core.read_model(model_path + "/yolov8n-seg.xml")
        self.compiled_model = self.core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        self.infer_request = self.compiled_model.create_infer_request()
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer_boxes = self.compiled_model.output(0)
        self.output_layer_masks = self.compiled_model.output(1)
        
    def letterbox(self, img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
            
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
            
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def process(self, frame):
        t0 = time.perf_counter()
        
        # 1. Preprocess
        img, ratio, pad = self.letterbox(frame, (self.infer_res, self.infer_res))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        t1 = time.perf_counter()
        
        # 2. Infer
        self.infer_request.infer([img])
        preds = self.infer_request.get_output_tensor(0).data
        protos = self.infer_request.get_output_tensor(1).data
        
        t2 = time.perf_counter()
        
        # 3. Postprocess NMS
        preds = preds[0].T # Shape (8400, 116)
        
        bboxes = preds[:, :4]
        scores = preds[:, 4:84]
        mask_coeffs = preds[:, 84:]
        
        max_scores = np.max(scores, axis=1)
        max_class_indices = np.argmax(scores, axis=1)
        
        # Filter by confidence
        valid_mask = max_scores > self.conf_thresh
        valid_bboxes = bboxes[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_indices = max_class_indices[valid_mask]
        valid_coeffs = mask_coeffs[valid_mask]
        
        # Convert xywh to xyxy for NMS
        if len(valid_bboxes) > 0:
            x, y, w, h = valid_bboxes[:, 0], valid_bboxes[:, 1], valid_bboxes[:, 2], valid_bboxes[:, 3]
            valid_bboxes_xyxy = np.stack([x - w/2, y - h/2, w, h], axis=1) # cv2.dnn.NMSBoxes expects x,y,w,h
            
            indices = cv2.dnn.NMSBoxes(valid_bboxes_xyxy.tolist(), valid_scores.tolist(), self.conf_thresh, self.iou_thresh)
            
            if len(indices) > 0:
                indices = indices.flatten()
                final_bboxes = valid_bboxes[indices]
                final_scores = valid_scores[indices]
                final_classes = valid_class_indices[indices]
                final_coeffs = valid_coeffs[indices]
                
                # 4. Mask Decoding
                # protos shape: (1, 32, 104, 104) -> (32, 10816)
                p_c, p_h, p_w = protos.shape[1:]
                protos_flat = protos[0].reshape(p_c, -1)
                
                # coeffs shape: (N, 32)
                # mask = coeffs @ protos_flat -> (N, 10816) -> (N, 104, 104)
                masks = final_coeffs @ protos_flat
                masks = 1 / (1 + np.exp(-masks)) # sigmoid
                masks = masks.reshape(-1, p_h, p_w)
                
                # Rescale boxes to original frame
                rx, ry = ratio
                dw, dh = pad
                
                # xywh to xyxy
                x1 = final_bboxes[:, 0] - final_bboxes[:, 2]/2
                y1 = final_bboxes[:, 1] - final_bboxes[:, 3]/2
                x2 = final_bboxes[:, 0] + final_bboxes[:, 2]/2
                y2 = final_bboxes[:, 1] + final_bboxes[:, 3]/2
                
                x1 = (x1 - dw) / rx
                x2 = (x2 - dw) / rx
                y1 = (y1 - dh) / ry
                y2 = (y2 - dh) / ry
                
                final_boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
                
                # clip
                final_boxes_xyxy[:, [0, 2]] = np.clip(final_boxes_xyxy[:, [0, 2]], 0, frame.shape[1])
                final_boxes_xyxy[:, [1, 3]] = np.clip(final_boxes_xyxy[:, [1, 3]], 0, frame.shape[0])
                
                # Resize masks to original frame
                final_masks = []
                orig_h, orig_w = frame.shape[:2]
                for i in range(len(masks)):
                    # resize mask to 416x416
                    mask_resized = cv2.resize(masks[i], (self.infer_res, self.infer_res), interpolation=cv2.INTER_LINEAR)
                    # crop padding
                    mask_cropped = mask_resized[int(dh):int(self.infer_res-dh), int(dw):int(self.infer_res-dw)]
                    # resize back to original image
                    mask_final = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    mask_bool = mask_final > 0.5
                    mask_uint8 = (mask_bool * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        largest_contour = max(contours, key=cv2.contourArea)
                        final_masks.append(largest_contour.reshape(-1, 2))
                    else:
                        final_masks.append(np.array([]))
                
                # We no longer convert final_masks to np.array(final_masks) because they have different sizes.
                # final_masks is a list of ndarrays.
                
                t3 = time.perf_counter()
                
                timing = {
                    "pre_ms": (t1 - t0) * 1000,
                    "infer_ms": (t2 - t1) * 1000,
                    "post_ms": (t3 - t2) * 1000,
                    "total_ms": (t3 - t0) * 1000
                }
                
                return final_boxes_xyxy, final_scores, final_classes, final_masks, timing
                
        t3 = time.perf_counter()
        timing = {
            "pre_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000
        }
        return np.array([]), np.array([]), np.array([]), np.array([]), timing
