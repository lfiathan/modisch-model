import os
from ultralytics import YOLO
import argparse
import sys
import pathlib

# make `src` importable when running this file directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from config import Config
from modeling.common import resolve_device, switch_labels

def train_phase2(seed_weights=None, epochs=Config.EPOCHS, imgsz=Config.IMG_SIZE, batch=Config.BATCH_SIZE,
                 workers=Config.WORKERS, patience=Config.PATIENCE, cache=True, device=None):
    yolo_device = resolve_device(device or Config.DEVICE)
    is_gpu = isinstance(yolo_device, int)
    print("Using device:", yolo_device)

    if not os.path.exists(Config.LABELS_PHASE2_DIR):
        raise FileNotFoundError(f"Phase2 labels not found: {Config.LABELS_PHASE2_DIR}")

    # choose weights: provided -> Config phase1 best -> YOLO nano
    chosen_weights = seed_weights if (seed_weights and os.path.exists(seed_weights)) else (
        Config.MODEL_PHASE1_WEIGHTS if os.path.exists(Config.MODEL_PHASE1_WEIGHTS) else 'yolov8n.pt'
    )

    print("\n--- STARTING PHASE 2 TRAINING ---")
    with switch_labels(Config.LABELS_PHASE2_DIR, Config.LABELS_DIR):
        model = YOLO(chosen_weights)
        model.train(
            data=Config.PHASE2_DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=yolo_device,
            cache=cache,
            half=bool(is_gpu),
            patience=patience,
            name='yolov8_phase2_fine',
            exist_ok=True
        )
    print("--- PHASE 2 TRAINING COMPLETE ---")

    best_weights = os.path.join('runs', 'detect', 'yolov8_phase2_fine', 'weights', 'best.pt')
    return best_weights if os.path.exists(best_weights) else None

def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 (fine)")
    parser.add_argument('--seed_weights', type=str, default=None, help="Path to seed weights (e.g. Phase1 best.pt)")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--imgsz', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--batch', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--workers', type=int, default=Config.WORKERS)
    parser.add_argument('--patience', type=int, default=Config.PATIENCE)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default=None, help="cpu | mps | 0 | 1 ...")
    args = parser.parse_args()

    best = train_phase2(
        seed_weights=args.seed_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        device=args.device
    )
    print("Phase2 best weights:", best)

if __name__ == '__main__':
    main()