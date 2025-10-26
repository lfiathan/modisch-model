import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import pathlib
import logging
from typing import Optional, Union
from .common import resolve_device
from ..config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # (Ensure box format is [x1, y1, x2, y2])
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0  # Avoid division by zero

    iou = inter_area / union_area
    return iou


def _get_class_name(model: YOLO, class_idx: int) -> str:
    """Safely get class name from YOLO model."""
    try:
        return model.names[int(class_idx)]
    except (KeyError, IndexError, TypeError):
        return f"Class_{class_idx}"


def hierarchical_predict(
    image_path: str,
    coarse_weights_path: str = str(Config.MODEL_PHASE1_WEIGHTS),
    fine_weights_path: str = str(Config.MODEL_PHASE2_WEIGHTS),
    iou_threshold: float = Config.IOU_THRESHOLD,
    device: Optional[Union[str, int]] = None,
    conf_threshold: float = 0.25,  # Add confidence threshold
) -> Optional[np.ndarray]:
    """Runs hierarchical prediction and returns the annotated image array."""

    yolo_device = resolve_device(device=device)
    logging.info("Using device: %s for prediction", yolo_device)

    if not os.path.exists(image_path):
        logging.error(f"Input image not found: {image_path}")
        return None
    if not os.path.exists(coarse_weights_path):
        logging.error(f"Coarse model weights not found: {coarse_weights_path}")
        return None
    if not os.path.exists(fine_weights_path):
        logging.error(f"Fine model weights not found: {fine_weights_path}")
        return None

    try:
        model_coarse = YOLO(coarse_weights_path)
        model_fine = YOLO(fine_weights_path)

        # Use predict method directly for potentially better performance/consistency
        results = model_fine.predict(
            source=image_path,
            device=yolo_device,
            verbose=False,
            conf=conf_threshold,  # Apply confidence threshold
        )

        if not results or len(results) == 0:
            logging.warning(f"No results returned by fine model for {image_path}")
            return None

        res_fine = results[0]  # Assuming single image prediction

        # Run coarse prediction only if fine prediction found something
        res_coarse = None
        if hasattr(res_fine, "boxes") and len(res_fine.boxes) > 0:
            results_coarse = model_coarse.predict(
                source=image_path,  # Use original image path
                device=yolo_device,
                verbose=False,
                conf=conf_threshold,  # Apply confidence threshold
            )
            if results_coarse and len(results_coarse) > 0:
                res_coarse = results_coarse[0]

        # Prepare an output image (res_fine.plot returns an image array)
        try:
            out_img = res_fine.plot(labels=False, line_width=2)
            # Ensure it's a mutable array (uint8) for cv2 drawing
            out_img = out_img.copy() if isinstance(out_img, np.ndarray) else None
        except Exception:
            out_img = None

        # If plotting failed, load the image manually as a fallback
        if out_img is None:
            out_img = cv2.imread(image_path)
            if out_img is None:
                logging.error(f"Failed to create output image for {image_path}")
                return None

        # Extract fine results
        fine_boxes = getattr(res_fine, "boxes", None)
        if fine_boxes is None or len(fine_boxes) == 0:
            logging.warning(f"No fine detections found in {image_path}")
            return out_img  # Return image with no boxes if none found

        fine_xyxy = fine_boxes.xyxy.cpu().numpy()
        fine_cls = fine_boxes.cls.cpu().numpy().astype(int)
        # Some versions may not expose conf; guard access
        try:
            fine_conf = fine_boxes.conf.cpu().numpy()
        except Exception:
            fine_conf = np.ones(len(fine_xyxy), dtype=float)

        # Extract coarse results if they exist
        coarse_xyxy = np.zeros((0, 4))
        coarse_cls = np.zeros((0,), dtype=int)
        if res_coarse and hasattr(res_coarse, "boxes") and len(res_coarse.boxes) > 0:
            coarse_xyxy = res_coarse.boxes.xyxy.cpu().numpy()
            coarse_cls = res_coarse.boxes.cls.cpu().numpy().astype(int)

        # Map fine boxes to coarse boxes and draw labels
        for i in range(len(fine_xyxy)):
            fbox = fine_xyxy[i]
            fcls_idx = fine_cls[i]
            fconf = float(fine_conf[i]) if i < len(fine_conf) else 0.0
            fine_name = _get_class_name(model_fine, fcls_idx)

            best_match_name = "Unknown"
            max_iou = 0.0
            for j in range(len(coarse_xyxy)):
                iou = calculate_iou(fbox, coarse_xyxy[j])
                if iou > max_iou:
                    max_iou = iou
                    best_match_name = _get_class_name(model_coarse, coarse_cls[j])

            # Decide label based on IoU threshold
            if max_iou >= iou_threshold:
                label = f"{best_match_name}: {fine_name} ({fconf:.2f})"
            else:
                label = f"{fine_name} ({fconf:.2f})"

            x1, y1, _, _ = map(int, fbox)

            # Use CV2 for text drawing on the plotted image
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # Make sure coordinates are in-bounds
            top_left = (max(0, x1), max(0, y1 - text_height - baseline - 5))
            bottom_right = (min(out_img.shape[1] - 1, x1 + text_width), min(out_img.shape[0] - 1, y1))
            cv2.rectangle(out_img, top_left, bottom_right, (0, 255, 0), -1)  # Background rect
            cv2.putText(out_img, label, (top_left[0], bottom_right[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text
        return out_img

    except Exception as e:
        logging.error(f"Error during prediction for {image_path}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    images_dir = str(Config.IMAGES_DIR)  # Ensure it's a string
    if not os.path.isdir(images_dir):
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    sample_candidates = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not sample_candidates:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    sample_image_name = random.choice(sample_candidates)
    img_path = os.path.join(images_dir, sample_image_name)

    print(f"Running hierarchical prediction on: {img_path}")
    annotated_image = hierarchical_predict(img_path)

    if annotated_image is not None:
        output_filename = "hierarchical_prediction_output.jpg"
        cv2.imwrite(output_filename, annotated_image)
        print(f"Annotated image saved as: {output_filename}")
    else:
        print("Prediction failed.")