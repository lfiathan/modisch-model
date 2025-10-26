import argparse
import logging
from pathlib import Path
from ..config import Config
from .trainer import add_common_train_args, run_training

def train_phase2(args):
    # Determine seed weights
    seed_weights_path = args.seed_weights or Config.MODEL_PHASE1_WEIGHTS
    if not Path(seed_weights_path).exists():
        logging.warning(f"Seed weights not found at {seed_weights_path}. Starting from yolov8n.pt.")
        seed_weights_path = 'yolov8n.pt'
        
    return run_training(
        model_weights=seed_weights_path,
        data_yaml=Config.PHASE2_DATA_YAML,
        labels_src_dir=Config.LABELS_PHASE2_DIR,
        run_name=Config.PHASE2_RUN_NAME,
        args=args
    )

def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 (fine)")
    parser = add_common_train_args(parser)
    parser.add_argument('--seed_weights', type=str, default=str(Config.MODEL_PHASE1_WEIGHTS), 
                        help="Path to seed weights (e.g., Phase1 best.pt)")
    args = parser.parse_args()
    best_weights = train_phase2(args)
    if best_weights:
        logging.info(f"Phase 2 finished. Best weights: {best_weights}")
    else:
        logging.error("Phase 2 training failed.")

if __name__ == '__main__':
    main()