# src/modeling/predict.py
from ultralytics import YOLO
import cv2
import numpy as np

# --- Helper Function from your notebook ---
def calculate_iou(box1, box2):
    # ... (same IoU calculation logic as before) ...
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# --- Main Prediction Class ---
class HierarchicalPredictor:
    def __init__(self, coarse_model_path, fine_model_path):
        self.model_coarse = YOLO(coarse_model_path)
        self.model_fine = YOLO(fine_model_path)
        self.iou_threshold = 0.8

    def run_inference(self, image_path):
        results_coarse = self.model_coarse(image_path, verbose=False)[0]
        results_fine = self.model_fine(image_path, verbose=False)[0]

        predictions = []

        for fine_box in results_fine.boxes:
            fine_xyxy = fine_box.xyxy[0].cpu().numpy().tolist()
            fine_class_name = self.model_fine.names[int(fine_box.cls)]
            confidence = float(fine_box.conf)

            best_match_coarse_name = "Unknown"
            max_iou = 0

            for coarse_box in results_coarse.boxes:
                iou = calculate_iou(fine_xyxy, coarse_box.xyxy[0].cpu().numpy())
                if iou > max_iou:
                    max_iou = iou
                    best_match_coarse_name = self.model_coarse.names[int(coarse_box.cls)]

            if max_iou > self.iou_threshold:
                predictions.append({
                    "box": fine_xyxy,
                    "coarse_label": best_match_coarse_name,
                    "fine_label": fine_class_name,
                    "confidence": confidence
                })

        return predictions