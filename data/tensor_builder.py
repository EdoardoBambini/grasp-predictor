"""
TensorBuilder: labeled+synchronized DataFrames in, sliding-window tensors
out. X shape (n_windows, seq_len, n_features), y shape (n_windows,).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_TIMESTAMP_COL = "timestamp_ns"
_LABEL_COL = "grasp_failure"


class TensorBuilder:
    """Builds sliding-window tensors. feature_columns=None -> all numeric
    columns (minus timestamp/label) resolved at the first build() call.
    normalize=True fits a StandardScaler in build_from_chunks."""

    def __init__(
        self,
        sequence_length: int = 50,
        feature_columns: Optional[List[str]] = None,
        label_column: str = _LABEL_COL,
        stride: int = 1,
        normalize: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.stride = stride
        self.normalize = normalize

        # Resolved at first call
        self._resolved_features: Optional[List[str]] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def n_features(self) -> Optional[int]:
        """Return the number of resolved features, or None before the first build()."""
        if self._resolved_features is None:
            return None
        return len(self._resolved_features)

    def build(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Tensor, Tensor]:
        """Build (X, y) for a single dense df. Must have grasp_failure + timestamp_ns."""
        features = self._resolve_features(df)
        self._validate_df(df, features)

        # NaN -> 0 on residual holes left by SyncTransformer at chunk boundaries.
        values = df[features].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(values, copy=False, nan=0.0)
        labels = df[self.label_column].to_numpy(dtype=np.float32)

        n_rows = len(values)
        # Shorter than one window -> skip (happens on very short Fractal seqs).
        if n_rows < self.sequence_length:
            logger.debug(
                "Chunk has %d rows, less than sequence_length=%d, skipping.",
                n_rows,
                self.sequence_length,
            )
            return torch.empty(0), torch.empty(0)

        indices = range(0, n_rows - self.sequence_length + 1, self.stride)
        windows_x = np.stack([values[i : i + self.sequence_length] for i in indices])
        # Window label = max over the 50 samples. One failure taints the
        # window — better a false positive than a false negative here.
        windows_y = np.array(
            [labels[i : i + self.sequence_length].max() for i in indices],
            dtype=np.float32,
        )

        X = torch.from_numpy(windows_x)   # (n_windows, seq_len, n_features)
        y = torch.from_numpy(windows_y)   # (n_windows,)

        logger.debug(
            "Built tensors: X=%s, y=%s from chunk with %d rows",
            tuple(X.shape),
            tuple(y.shape),
            n_rows,
        )
        return X, y

    def build_from_chunks(
        self,
        chunks: List[pd.DataFrame],
    ) -> Tuple[Tensor, Tensor]:
        """Build tensors from a list of chunks and concat the results."""
        all_X: List[Tensor] = []
        all_y: List[Tensor] = []

        for chunk in chunks:
            X, y = self.build(chunk)
            if X.numel() > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            raise ValueError(
                "No valid windows extracted. "
                "Check sequence_length against the real chunk size."
            )

        X_out = torch.cat(all_X, dim=0)
        y_out = torch.cat(all_y, dim=0)

        if self.normalize:
            # Z-score, per feature. Must be fitted on the train split only;
            # val/test reuse _mean/_std without refitting (see run_training).
            N, T, F = X_out.shape
            flat = X_out.reshape(-1, F).numpy()
            np.nan_to_num(flat, copy=False, nan=0.0)
            self._mean = flat.mean(axis=0)
            self._std = flat.std(axis=0)
            # Constant feature -> std=0 -> inf everywhere. Keep it untouched.
            self._std[self._std < 1e-8] = 1.0
            normalized = ((flat - self._mean) / self._std).reshape(N, T, F).astype(np.float32)
            # Extreme samples can produce inf after normalization, zero them.
            np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            X_out = torch.from_numpy(normalized)
            logger.info(
                "Features normalized: mean range [%.3f, %.3f], std range [%.3f, %.3f]",
                self._mean.min(), self._mean.max(),
                self._std.min(), self._std.max(),
            )

        return X_out, y_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        """Resolve feature columns at the first call, then cache them for consistency."""
        if self._resolved_features is not None:
            return self._resolved_features

        if self.feature_columns is not None:
            self._resolved_features = self.feature_columns
        else:
            # Auto-detect: all numeric columns except timestamp and label.
            exclude = {_TIMESTAMP_COL, self.label_column}
            self._resolved_features = sorted([
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ])
            logger.info(
                "Auto-detected %d feature columns: %s",
                len(self._resolved_features),
                self._resolved_features,
            )

        return self._resolved_features

    def _validate_df(self, df: pd.DataFrame, features: List[str]) -> None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in the DataFrame: {missing}")
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in the DataFrame."
            )
        if df[features].isnull().any().any():
            n_nan = df[features].isnull().sum().sum()
            logger.warning(
                "Detected %d NaN values in the feature columns after synchronization. "
                "Consider adjusting the sync policy or checking sensor data integrity.",
                n_nan,
            )
