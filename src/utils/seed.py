"""
Deterministic Seed Locker for Full Reproducibility.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets seeds across random, numpy, and torch for CPU & GPU determinism."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] Deterministic random seed set to: {seed}")

if __name__ == "__main__":
    set_seed(42)
