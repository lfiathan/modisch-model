import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import pathlib

# make `src` importable when running this file directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from config import Config
from modeling.common import resolve_device

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def _class_name(model, idx: int) -> str:
    names = model.names
    if isinstance(names, dict):
        return names.get(int(idx), str(int(idx)))
    try:
        return names[int(idx)]
    except Exception:
        return str(int(idx))

def hierarchical_predict(image_path,
                         coarse_weights=Config.MODEL_PHASE1_WEIGHTS,
                         fine_weights=Config.MODEL_PHASE2_WEIGHTS,
                         iou_threshold=Config.IOU_THRESHOLD,
                         device=None):
    yolo_device = resolve_device(device or Config.DEVICE)

    model_coarse = YOLO(coarse_weights)
    model_fine = YOLO(fine_weights)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    res_coarse = model_coarse.predict(source=img, device=yolo_device, verbose=False)[0]
    res_fine = model_fine.predict(source=img, device=yolo_device, verbose=False)[0]

    out_img = img.copy()

    # extract arrays safely
    if hasattr(res_coarse, "boxes") and len(res_coarse.boxes) > 0:
        coarse_xyxy = res_coarse.boxes.xyxy.cpu().numpy()
        coarse_cls = res_coarse.boxes.cls.cpu().numpy().astype(int)
    else:
        coarse_xyxy = np.zeros((0, 4))
        coarse_cls = np.zeros((0,), dtype=int)

    if hasattr(res_fine, "boxes") and len(res_fine.boxes) > 0:
        fine_xyxy = res_fine.boxes.xyxy.cpu().numpy()
        fine_cls = res_fine.boxes.cls.cpu().numpy().astype(int)
    else:
        fine_xyxy = np.zeros((0, 4))
        fine_cls = np.zeros((0,), dtype=int)

    for i in range(len(fine_xyxy)):
        fbox = fine_xyxy[i]
        fine_name = _class_name(model_fine, int(fine_cls[i]))

        best_match = "Unknown"
        max_iou = 0.0
        for j in range(len(coarse_xyxy)):
            iou = calculate_iou(fbox, coarse_xyxy[j])
            if iou > max_iou:
                max_iou = iou
                best_match = _class_name(model_coarse, int(coarse_cls[j]))

        label = f"{best_match}: {fine_name}" if max_iou > iou_threshold else fine_name
        x1, y1, x2, y2 = map(int, fbox)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out_img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out_img

if __name__ == '__main__':
    images_dir = Config.IMAGES_DIR
    sample_candidates = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample = random.choice(sample_candidates)
    img_path = os.path.join(images_dir, sample)
    out = hierarchical_predict(img_path)
    cv2.imwrite('hierarchical_demo.jpg', out)
    print("Wrote hierarchical_demo.jpg")