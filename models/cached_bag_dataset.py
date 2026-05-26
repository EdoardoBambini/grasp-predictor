"""Bag-level dataset for MIL training (one bag = one sequence) with optional tail-fraction window filtering on failure bags."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedSequenceBagDataset(Dataset):
    """Groups per-window cached features into per-sequence bags.

    Args:
      Xk:   (N_windows, T, kin_dim) float
      Xc:   (N_windows, T, cnn_dim) float
      y:    (N_windows,) float window labels (each window inherits its bag label)
      seq_ids: (N_windows,) sequence id per window (groups windows into bags)
      bag_dataset_ids: list[str] dataset id per bag (parallel to sorted unique
                       seq_ids); required for tail_fraction_for.
      tail_fraction_for: {dataset_id: frac}. For FAILURE bags of that dataset,
                       keep only the last round(frac * n_windows) windows.
      bag_nl_emb: (n_bags, 512) per-bag natural-language embedding (Fractal USE);
                  zeros for DROID (null token handled in the model).
      bag_is_droid: (n_bags,) bool per-bag DROID flag (routes the null NL token).
    """

    def __init__(
        self,
        Xk: np.ndarray,
        Xc: np.ndarray,
        y: np.ndarray,
        seq_ids: np.ndarray,
        bag_dataset_ids: Optional[List[str]] = None,
        tail_fraction_for: Optional[Dict[str, float]] = None,
        bag_nl_emb: Optional[np.ndarray] = None,
        bag_is_droid: Optional[np.ndarray] = None,
    ) -> None:
        self._Xk = np.asarray(Xk, dtype=np.float32)
        self._Xc = np.asarray(Xc, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        seq_ids = np.asarray(seq_ids)

        # Stable unique sequence order; window index lists per sequence.
        self._unique_seqs = list(dict.fromkeys(seq_ids.tolist()))
        idx_by_seq: Dict[object, List[int]] = {s: [] for s in self._unique_seqs}
        for i, s in enumerate(seq_ids.tolist()):
            idx_by_seq[s].append(i)
        self._windows_per_seq: List[List[int]] = [idx_by_seq[s] for s in self._unique_seqs]
        # Bag label = max over its window labels (failure if any window is failure).
        self._bag_labels = np.array(
            [float(np.max(y[ws])) if ws else 0.0 for ws in self._windows_per_seq],
            dtype=np.float32,
        )

        n = len(self._unique_seqs)
        if bag_nl_emb is None:
            bag_nl_emb = np.zeros((n, 512), dtype=np.float32)
        self._bag_nl_emb = bag_nl_emb.astype(np.float32, copy=False)
        if bag_is_droid is None:
            bag_is_droid = np.zeros((len(self._unique_seqs),), dtype=bool)
        elif bag_is_droid.shape != (len(self._unique_seqs),):
            raise ValueError(
                f"bag_is_droid shape {bag_is_droid.shape} != (n_bags,)",
            )
        self._bag_is_droid = np.asarray(bag_is_droid, dtype=bool)

        # Optional v10h+ temporal window filtering on failure bags.
        self._filter_stats = {"applied": False}
        if bag_dataset_ids is not None and tail_fraction_for:
            if len(bag_dataset_ids) != len(self._unique_seqs):
                raise ValueError(
                    f"bag_dataset_ids length {len(bag_dataset_ids)} != "
                    f"unique_seqs length {len(self._unique_seqs)}",
                )
            n_filtered = 0
            sum_before = 0
            sum_after = 0
            for i, ws in enumerate(self._windows_per_seq):
                ds_id = bag_dataset_ids[i]
                frac = float(tail_fraction_for.get(ds_id, 1.0))
                if frac >= 1.0 or self._bag_labels[i] <= 0.5:
                    continue
                n_keep = max(1, int(round(len(ws) * frac)))
                if n_keep >= len(ws):
                    continue
                sum_before += len(ws)
                self._windows_per_seq[i] = ws[-n_keep:]
                sum_after += n_keep
                n_filtered += 1
            self._filter_stats = {
                "applied": True,
                "n_bags_filtered": n_filtered,
                "avg_len_before": (sum_before / n_filtered) if n_filtered else 0.0,
                "avg_len_after": (sum_after / n_filtered) if n_filtered else 0.0,
                "tail_fraction_for": dict(tail_fraction_for),
            }

    def __len__(self) -> int:
        return len(self._unique_seqs)

    def __getitem__(self, idx: int):
        ws = self._windows_per_seq[idx]
        xk = torch.from_numpy(self._Xk[ws])            # (M, T, kin_dim)
        xc = torch.from_numpy(self._Xc[ws])            # (M, T, cnn_dim)
        y = torch.tensor(float(self._bag_labels[idx]), dtype=torch.float32)
        nl = torch.from_numpy(self._bag_nl_emb[idx])   # (512,)
        is_droid = torch.tensor(bool(self._bag_is_droid[idx]))
        return xk, xc, y, nl, is_droid

    @property
    def bag_labels(self) -> np.ndarray:
        return self._bag_labels

    @property
    def filter_stats(self) -> dict:
        return self._filter_stats


def collate_bags(batch):
    """Pad a list of (xk, xc, y, nl, is_droid) bags to the max #windows in the
    batch and build the validity mask. Returns, in the order the trainer and
    LateFusionLSTMMIL.forward(X_kin, X_cnn, mask, nl_emb, is_droid) expect:
      Xk (B,M,T,kin), Xc (B,M,T,cnn), mask (B,M) bool, y (B,),
      nl_emb (B,512), is_droid (B,).
    """
    M = max(b[0].shape[0] for b in batch)
    B = len(batch)
    T = batch[0][0].shape[1]
    kin_dim = batch[0][0].shape[2]
    cnn_dim = batch[0][1].shape[2]
    Xk = torch.zeros((B, M, T, kin_dim), dtype=torch.float32)
    Xc = torch.zeros((B, M, T, cnn_dim), dtype=torch.float32)
    mask = torch.zeros((B, M), dtype=torch.bool)
    y = torch.zeros((B,), dtype=torch.float32)
    nl = torch.zeros((B, 512), dtype=torch.float32)
    is_droid = torch.zeros((B,), dtype=torch.bool)
    for i, (xk, xc, yy, nlb, isd) in enumerate(batch):
        m = xk.shape[0]
        Xk[i, :m] = xk
        Xc[i, :m] = xc
        mask[i, :m] = True
        y[i] = yy
        nl[i] = nlb
        is_droid[i] = isd
    return Xk, Xc, mask, y, nl, is_droid
