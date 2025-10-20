import os
import shutil
from ultralytics import YOLO
import argparse
import sys
import pathlib

# make `src` importable when running this file directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from config import Config
from modeling.common import resolve_device, switch_labels

# --- Dataset-specific mappings for optional prepare ---
original_classes = {
    0: 'sunglass', 1: 'hat', 2: 'jacket', 3: 'shirt', 4: 'pants',
    5: 'shorts', 6: 'skirt', 7: 'dress', 8: 'bag', 9: 'shoe'
}
phase1_classes = {'TOP': 0, 'BOTTOM': 1, 'SHOES': 2}
phase1_map = {
    0: None, 1: None, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: None, 9: 2
}

def process_original_annotations(original_annotations_path, labels_phase1_dir):
    os.makedirs(labels_phase1_dir, exist_ok=True)
    processed = 0
    for filename in os.listdir(original_annotations_path):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(original_annotations_path, filename), 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        new_lines = []
        for line in lines:
            parts = line.split()
            try:
                original_class_id = int(parts[0])
            except Exception:
                continue
            coords = ' '.join(parts[1:])
            if original_class_id in phase1_map:
                nc = phase1_map[original_class_id]
                if nc is not None:
                    new_lines.append(f"{nc} {coords}")
        with open(os.path.join(labels_phase1_dir, filename), 'w') as f:
            if new_lines:
                f.write("\n".join(new_lines))
        processed += 1
    print(f"Processed {processed} annotation files")

def train_phase1(epochs=Config.EPOCHS, imgsz=Config.IMG_SIZE, batch=Config.BATCH_SIZE,
                 workers=Config.WORKERS, patience=Config.PATIENCE, cache=True, device=None,
                 prepare=False, original_annotations_path=None):
    # Optional label preparation
    if prepare:
        if original_annotations_path is None:
            original_annotations_path = Config.ORIGINAL_ANNOTATIONS_PATH
        if os.path.exists(original_annotations_path):
            print("Preparing labels_phase1 from original annotations...")
            process_original_annotations(original_annotations_path, Config.LABELS_PHASE1_DIR)
        else:
            print(f"Original annotations not found: {original_annotations_path}", file=sys.stderr)

    yolo_device = resolve_device(device or Config.DEVICE)
    is_gpu = isinstance(yolo_device, int)
    print("Using device:", yolo_device)

    print("\n--- STARTING PHASE 1 TRAINING ---")
    with switch_labels(Config.LABELS_PHASE1_DIR, Config.LABELS_DIR):
        model = YOLO('yolov8n.pt')
        model.train(
            data=Config.PHASE1_DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=yolo_device,
            cache=cache,
            half=bool(is_gpu),
            patience=patience,
            name='yolov8_phase1_coarse',
            exist_ok=True
        )
    print("--- PHASE 1 TRAINING COMPLETE ---")

    best_weights = os.path.join('runs', 'detect', 'yolov8_phase1_coarse', 'weights', 'best.pt')
    return best_weights if os.path.exists(best_weights) else None

def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 (coarse)")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--imgsz', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--batch', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--workers', type=int, default=Config.WORKERS)
    parser.add_argument('--patience', type=int, default=Config.PATIENCE)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default=None, help="cpu | mps | 0 | 1 ...")
    parser.add_argument('--prepare', action='store_true', help="Prepare labels_phase1 from original annotations")
    parser.add_argument('--original_annotations', type=str, default=None, help="Path to original annotations")
    args = parser.parse_args()

    best = train_phase1(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        device=args.device,
        prepare=args.prepare,
        original_annotations_path=args.original_annotations
    )
    print("Phase1 best weights:", best)

if __name__ == '__main__':
    main()