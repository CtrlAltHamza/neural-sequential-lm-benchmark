"""
Unified Model Trainer with Early Stopping and Metrics Logging.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)
"""

import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.training.early_stopping import EarlyStopping
from src.evaluation.metrics import evaluate_model

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        patience: int = 2,
        model_name: str = "model"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()

            # Gradient clipping to stabilize recurrent updates
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

        mean_loss = total_loss / total_samples
        ppl = math.exp(min(mean_loss, 20.0))
        acc = (total_correct / total_samples) * 100.0
        return {"loss": mean_loss, "perplexity": ppl, "accuracy": acc}

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        history = {
            "epoch": [],
            "train_loss": [], "train_ppl": [], "train_acc": [],
            "val_loss": [], "val_ppl": [], "val_acc": [],
            "epoch_times": []
        }

        best_val_ppl = float("inf")
        stopped_epoch = epochs

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_loss, val_ppl, val_acc = evaluate_model(self.model, val_loader, self.criterion, self.device)
            elapsed = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_ppl"].append(train_metrics["perplexity"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["val_acc"].append(val_acc)
            history["epoch_times"].append(elapsed)

            if verbose:
                print(f"[{self.model_name}] Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
                      f"Train Loss: {train_metrics['loss']:.4f} | Train PP: {train_metrics['perplexity']:.2f} | Train Acc: {train_metrics['accuracy']:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val PP: {val_ppl:.2f} | Val Acc: {val_acc:.2f}%")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl

            # Check early stopping on validation perplexity
            if self.early_stopping(val_ppl, self.model):
                if verbose:
                    print(f"[{self.model_name}] Early stopping triggered at epoch {epoch} (Patience={self.patience})")
                stopped_epoch = epoch
                break

        # Restore optimal weights
        self.early_stopping.restore_best_weights(self.model)
        history["best_val_ppl"] = best_val_ppl
        history["stopped_epoch"] = stopped_epoch
        return history
