"""
Dataset wrapping pre-computed window tensors (kin float32, cnn fp16, label
float32). Pairs with CachedFeatureLSTM / LateFusionLSTM and run_training_cached.py.

Optional kinematic-side noise augmentation: small Gaussian on X_kin only.
The CNN cache is read-only on disk, so visual augmentation is not available
without recomputing it. The cache trades augmentation flexibility for the
ability to train the temporal head in minutes on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """In-memory dataset over pre-windowed tensors.

    X_kin: (N, T, 15) float32, already z-score normalized using train stats.
    X_cnn: (N, T, cnn_dim) fp16 from disk; cast to float32 at __getitem__.
           Default cnn_dim is 360 for spatial MobileNetV3 Small features
           (intermediate layer 6 + AdaptiveMaxPool2d(3)).
    y:     (N,) float32 in {0, 1}.
    """

    def __init__(
        self,
        X_kin: np.ndarray,
        X_cnn: np.ndarray,
        y: np.ndarray,
        kin_noise_std: float = 0.0,
    ) -> None:
        if X_kin.shape[0] != X_cnn.shape[0] or X_kin.shape[0] != y.shape[0]:
            raise ValueError(
                f"length mismatch: kin={X_kin.shape[0]} cnn={X_cnn.shape[0]} y={y.shape[0]}"
            )
        if X_kin.shape[1] != X_cnn.shape[1]:
            raise ValueError(
                f"seq_len mismatch: kin={X_kin.shape[1]} cnn={X_cnn.shape[1]}"
            )
        self._X_kin = X_kin.astype(np.float32, copy=False)
        self._X_cnn = X_cnn  # keep fp16; cast at __getitem__
        self._y = y.astype(np.float32, copy=False)
        self._kin_noise_std = float(kin_noise_std)

    def __len__(self) -> int:
        return self._X_kin.shape[0]

    def __getitem__(self, idx: int):
        x_kin = self._X_kin[idx]
        if self._kin_noise_std > 0.0:
            x_kin = x_kin + np.random.normal(
                0.0, self._kin_noise_std, size=x_kin.shape
            ).astype(np.float32)
        x_cnn = self._X_cnn[idx].astype(np.float32, copy=False)
        return (
            torch.from_numpy(x_kin),
            torch.from_numpy(x_cnn),
            torch.tensor(self._y[idx], dtype=torch.float32),
        )

    @property
    def labels(self) -> np.ndarray:
        return self._y
