import os
import torch

class Config:
    DATASET_ROOT = './data'
    IMAGES_DIR = os.path.join(DATASET_ROOT, 'raw', 'JPEGImages')
    ORIGINAL_ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, 'raw', 'Annotations_txt')
    LABELS_PHASE1_DIR = os.path.join(DATASET_ROOT, 'processed', 'labels_phase1')
    LABELS_PHASE2_DIR = os.path.join(DATASET_ROOT, 'processed', 'labels_phase2')
    LABELS_DIR = os.path.join(DATASET_ROOT, 'processed', 'labels')
    TRAIN_LIST_TXT = os.path.abspath(os.path.join(DATASET_ROOT, 'train_full.txt'))
    VAL_LIST_TXT = os.path.abspath(os.path.join(DATASET_ROOT, 'val_full.txt'))
    PHASE1_DATA_YAML = 'models/phase1-data.yaml'
    PHASE2_DATA_YAML = 'models/phase2-data.yaml'
    MODEL_PHASE1_WEIGHTS = 'runs/detect/yolov8_phase1_coarse/weights/best.pt'
    MODEL_PHASE2_WEIGHTS = 'runs/detect/yolov8_phase2_fine/weights/best.pt'
    EPOCHS = 50
    IMG_SIZE = 512
    BATCH_SIZE = 16
    WORKERS = 8
    PATIENCE = 10
    IOU_THRESHOLD = 0.7
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')