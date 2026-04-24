"""
Training loop for GraspIntegrityLSTM.

BCEWithLogitsLoss (or focal) + automatic pos_weight, Adam/RMSprop, early
stopping on val_loss, best-checkpoint saving. Seed is fixed so the run is
reproducible against the same Mosaico catalog.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import GraspDataset
from .lstm import GraspIntegrityLSTM

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary focal loss: BCE * (1-p_t)^gamma. Kept as an alternative — the
    released run uses BCEWithLogitsLoss + pos_weight instead."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select best available compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Training orchestrator for GraspIntegrityLSTM.

    pos_weight=None -> computed from the training labels (n_neg / n_pos).
    Early stopping watches val_loss and saves the best checkpoint.
    """

    def __init__(
        self,
        model: GraspIntegrityLSTM,
        train_dataset: GraspDataset,
        val_dataset: GraspDataset,
        batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adam",
        pos_weight: Optional[float] = None,
        checkpoint_dir: str = "checkpoints/",
        early_stopping_patience: int = 10,
        seed: int = 42,
        loss_type: str = "bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_lr_scheduler: bool = False,
    ) -> None:
        set_seed(seed)
        self.device = get_device()
        logger.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.patience = early_stopping_patience

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # pos_weight = n_neg/n_pos. Without it the model collapses to always
        # predicting success and still scores ~85% acc on the imbalanced set.
        _pos_weight = pos_weight
        if _pos_weight is None:
            labels = train_dataset._y
            n_pos = labels.sum().item()
            n_neg = len(labels) - n_pos
            # guard /0 if train has no failures
            _pos_weight = n_neg / max(n_pos, 1)
            logger.info(
                "Auto-computed pos_weight=%.3f (n_pos=%d, n_neg=%d)",
                _pos_weight, n_pos, n_neg,
            )

        if loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, pos_weight=_pos_weight
            )
            logger.info("Using FocalLoss(alpha=%.2f, gamma=%.1f, pos_weight=%.3f)",
                        focal_alpha, focal_gamma, _pos_weight)
        else:
            pw = torch.tensor([_pos_weight], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        if optimizer_name.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        # LR halved when val_loss stalls for 3 epochs.
        self.scheduler = None
        if use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3,
            )
            logger.info("Using ReduceLROnPlateau scheduler (factor=0.5, patience=3)")

    def train(self) -> Dict[str, list]:
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_acc = self._eval_epoch()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                epoch, self.num_epochs, train_loss, val_loss, val_acc,
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # LSTM starts memorizing train after ~20-30 epochs, so early
            # stopping really matters here.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best_model.pt")
                logger.info("  -> Saved best model (val_loss=%.4f)", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        return history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            # clip grads: BPTT loves to explode
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.shape[0]

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_best_checkpoint(self) -> None:
        """Reload the best saved weights into the model."""
        # weights_only=True required by PyTorch 2.6+; the state_dicts are
        # plain tensors so it doesn't change anything either way.
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        logger.info("Loaded best checkpoint from %s", path)
