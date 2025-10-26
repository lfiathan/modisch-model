import os
from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv() # Loads variables from .env file

class Config:
    # Use pathlib for all paths
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # Allow overriding dataset root via environment variable
    DATASET_ROOT_STR = os.getenv('DATASET_ROOT', str(REPO_ROOT / 'dataset' / 'colorful_fashion_dataset_for_object_detection'))
    DATASET_ROOT = Path(DATASET_ROOT_STR)

    IMAGES_DIR = DATASET_ROOT / 'JPEGImages'
    # Use Path objects directly
    ORIGINAL_ANNOTATIONS_PATH = DATASET_ROOT / 'Annotations_txt'

    LABELS_BASE = DATASET_ROOT # Base directory for label variations
    LABELS_PHASE1_DIR = LABELS_BASE / 'labels_phase1'
    LABELS_PHASE2_DIR = LABELS_BASE / 'labels_phase2'
    # Standard labels dir used temporarily by training scripts
    LABELS_DIR = LABELS_BASE / 'labels' 

    # Use Path objects for list files
    TRAIN_LIST_TXT = DATASET_ROOT / 'train_full.txt'
    VAL_LIST_TXT = DATASET_ROOT / 'val_full.txt'

    # Centralize model/run locations relative to repo root
    MODELS_DIR = REPO_ROOT / 'models'
    RUNS_DIR = REPO_ROOT / 'runs'

    # YAML paths using the centralized location
    PHASE1_DATA_YAML = MODELS_DIR / 'phase1-data.yaml'
    PHASE2_DATA_YAML = MODELS_DIR / 'phase2-data.yaml'

    # Define model run names/subdirs
    PHASE1_RUN_NAME = 'yolov8_phase1_coarse'
    PHASE2_RUN_NAME = 'yolov8_phase2_fine'

    # Construct weights paths relative to RUNS_DIR
    MODEL_PHASE1_WEIGHTS = RUNS_DIR / 'detect' / PHASE1_RUN_NAME / 'weights' / 'best.pt'
    MODEL_PHASE2_WEIGHTS = RUNS_DIR / 'detect' / PHASE2_RUN_NAME / 'weights' / 'best.pt'

    # Training Hyperparameters (Keep as is or adjust)
    EPOCHS = 50
    IMG_SIZE = 512
    BATCH_SIZE = 16
    WORKERS = 8 # Set based on your CPU cores
    PATIENCE = 10
    IOU_THRESHOLD = 0.7 # Used in prediction

    # Device configuration (moved to common.py, but can be kept here as default)
    # DEVICE = 'cuda' if torch.cuda.is_available() else (
    #     'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
    # )

    # Add any Supabase/API keys here, loaded from .env
    # SUPABASE_URL = os.getenv("SUPABASE_URL")
    # SUPABASE_KEY = os.getenv("SUPABASE_KEY")