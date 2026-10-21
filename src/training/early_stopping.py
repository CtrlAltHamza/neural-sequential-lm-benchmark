"""
Early Stopping Handler based on Validation Perplexity.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)
"""

import copy
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience: int = 2, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_metric: float, model: torch.nn.Module) -> bool:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

        return False

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
