import argparse
import os
import logging
from ultralytics import YOLO
from ..config import Config # Use relative import within src
from .common import resolve_device, switch_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_common_train_args(parser):
    """Adds common training arguments to an argparse parser."""
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--imgsz', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--batch', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--workers', type=int, default=Config.WORKERS)
    parser.add_argument('--patience', type=int, default=Config.PATIENCE)
    parser.add_argument('--cache', action='store_true', default=True, help="Use dataset caching.")
    parser.add_argument('--no-cache', action='store_false', dest='cache', help="Disable dataset caching.")
    parser.add_argument('--device', type=str, default=None, help="cpu | mps | 0 | 1 ...")
    return parser

def run_training(model_weights, data_yaml, labels_src_dir, run_name, args):
    """Runs a YOLO training phase."""
    yolo_device = resolve_device(args.device)
    is_gpu = isinstance(yolo_device, int) or yolo_device == 'mps'
    logging.info(f"Using device: {yolo_device}")

    if not labels_src_dir.exists():
        logging.error(f"Labels source directory not found: {labels_src_dir}")
        return None
        
    logging.info(f"Starting training for {run_name} using weights: {model_weights}")
    
    # Use context manager to handle label switching
    with switch_labels(str(labels_src_dir), str(Config.LABELS_DIR)):
        try:
            model = YOLO(str(model_weights)) # Ensure weights path is a string
            model.train(
                data=str(data_yaml), # Ensure YAML path is a string
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=yolo_device,
                cache=args.cache,
                half=bool(is_gpu), # Use half precision if on GPU/MPS
                patience=args.patience,
                name=run_name,
                exist_ok=True,
                project=str(Config.RUNS_DIR / 'detect') # Explicitly set project dir
            )
            logging.info(f"--- Training complete for {run_name} ---")

            best_weights_path = Config.RUNS_DIR / 'detect' / run_name / 'weights' / 'best.pt'
            if best_weights_path.exists():
                logging.info(f"Best weights saved at: {best_weights_path}")
                return str(best_weights_path)
            else:
                logging.warning("Training finished, but best.pt not found.")
                return None
        except Exception as e:
            logging.error(f"Error during training for {run_name}: {e}", exc_info=True)
            return None