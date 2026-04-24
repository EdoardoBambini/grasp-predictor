"""
GraspDataset: thin Dataset wrapper around TensorBuilder's output, plus a
temporal-split helper.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class GraspDataset(Dataset):
    """Wraps the window tensors from TensorBuilder. X is (n, seq_len, n_features),
    y is (n,) float32 with 0=success, 1=failure."""

    def __init__(self, X: Tensor, y: Tensor) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X={X.shape[0]}, y={y.shape[0]}"
            )
        self._X = X.float()
        self._y = y.float()

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self._X[idx], self._y[idx]

    @property
    def n_features(self) -> int:
        return self._X.shape[2]

    @property
    def sequence_length(self) -> int:
        return self._X.shape[1]

    @classmethod
    def temporal_split(
        cls,
        X: Tensor,
        y: Tensor,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple["GraspDataset", "GraspDataset", "GraspDataset"]:
        """Split by index position: max(train_t) < min(val_t) < min(test_t)."""
        # Not used by cross-dataset training (which stratifies per-dataset in
        # run_training_crossdataset.build_split_tensors). Kept for legacy
        # single-dataset runs.
        n = X.shape[0]
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

        return cls(X_train, y_train), cls(X_val, y_val), cls(X_test, y_test)
