import os
from pathlib import Path
import torch

class Config:
    # repo root (one level above src/)
    REPO_ROOT = Path(__file__).resolve().parents[1]

    # dataset lives in repo_root/dataset/colorful_fashion_dataset_for_object_detection
    DATASET_ROOT = str(REPO_ROOT / 'dataset' / 'colorful_fashion_dataset_for_object_detection')

    IMAGES_DIR = os.path.join(DATASET_ROOT, 'JPEGImages')
    # fixed: Annotations_txt sits directly under DATASET_ROOT in your tree
    ORIGINAL_ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, 'Annotations_txt')

    # fixed: labels folders are directly under DATASET_ROOT
    LABELS_PHASE1_DIR = os.path.join(DATASET_ROOT, 'labels_phase1')
    LABELS_PHASE2_DIR = os.path.join(DATASET_ROOT, 'labels_phase2')
    LABELS_DIR = os.path.join(DATASET_ROOT, 'labels')

    TRAIN_LIST_TXT = os.path.abspath(os.path.join(DATASET_ROOT, 'train_full.txt'))
    VAL_LIST_TXT = os.path.abspath(os.path.join(DATASET_ROOT, 'val_full.txt'))

    # keep model yaml paths relative to repo root
    PHASE1_DATA_YAML = str(REPO_ROOT / 'models' / 'phase1-data.yaml')
    PHASE2_DATA_YAML = str(REPO_ROOT / 'models' / 'phase2-data.yaml')

    MODEL_PHASE1_WEIGHTS = 'runs/detect/yolov8_phase1_coarse/weights/best.pt'
    MODEL_PHASE2_WEIGHTS = 'runs/detect/yolov8_phase2_fine/weights/best.pt'

    EPOCHS = 50
    IMG_SIZE = 512
    BATCH_SIZE = 16
    WORKERS = 8
    PATIENCE = 10
    IOU_THRESHOLD = 0.7

    DEVICE = 'cuda' if torch.cuda.is_available() else (
        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
    )