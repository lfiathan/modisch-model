from contextlib import contextmanager
import os
import shutil
import torch

def auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def resolve_device(device=None):
    # Normalize into what Ultralytics accepts: int GPU index, 'cpu', or 'mps'
    if device is None:
        device = auto_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        s = device.strip().lower()
        if s.isdigit():
            return int(s)
        if s.startswith('cuda'):
            return 0 if torch.cuda.is_available() else 'cpu'
        if s in ('mps', 'cpu'):
            return s
    return device

@contextmanager
def switch_labels(src_dir, active_dir):
    """
    Temporarily move labels from src_dir into active_dir, then restore.
    Ensures cleanup/restoration even on exceptions.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Labels source not found: {src_dir}")
    os.makedirs(os.path.dirname(active_dir), exist_ok=True)
    if os.path.exists(active_dir):
        shutil.rmtree(active_dir)
    # move src -> active
    shutil.move(src_dir, active_dir)
    try:
        yield
    finally:
        # restore active -> src
        os.makedirs(os.path.dirname(src_dir), exist_ok=True)
        if os.path.exists(active_dir):
            shutil.move(active_dir, src_dir)