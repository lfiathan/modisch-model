import argparse
import logging
from ..config import Config
from .trainer import add_common_train_args, run_training
# ... (keep process_original_annotations if needed) ...

def train_phase1(args):
    # ... (add --prepare logic if kept) ...
    return run_training(
        model_weights='yolov8n.pt', # Always start phase 1 from scratch or base
        data_yaml=Config.PHASE1_DATA_YAML,
        labels_src_dir=Config.LABELS_PHASE1_DIR,
        run_name=Config.PHASE1_RUN_NAME,
        args=args
    )

def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 (coarse)")
    parser = add_common_train_args(parser)
    # ... (add --prepare args if kept) ...
    args = parser.parse_args()
    best_weights = train_phase1(args)
    if best_weights:
        logging.info(f"Phase 1 finished. Best weights: {best_weights}")
    else:
        logging.error("Phase 1 training failed.")

if __name__ == '__main__':
    main()