"""Train the LateFusionLSTM jointly on all datasets in the CNN feature cache,
evaluate per dataset. Stratified 70/15/15 split, seed 42."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))

from models.cached_dataset import CachedFeatureDataset
from models.cached_lstm import CachedFeatureLSTM, LateFusionLSTM
from models.cached_bag_dataset import CachedSequenceBagDataset, collate_bags
from models.mil_attention import LateFusionLSTMMIL
from models.trainer import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("training.cached")

SEED = 42
SEQ_LEN = 50
STRIDE = 10  # default; overridable via --stride
TRAIN_RATIO_INDIST = 0.70
VAL_RATIO_INDIST = 0.15
VAL_RATIO_ZEROSHOT = 0.15
KIN_DIM = 15
CNN_DIM = 576  # default for MobileNetV3-small; overridable via --cnn-dim


# --------------------------------------------------------------------------
# Cache index + window building
# --------------------------------------------------------------------------

def index_cache(cache_dir: str, datasets: List[str],
                overrides: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """{dsid: [npz_path, ...]} for each requested dataset.

    `overrides` lets a specific dataset point to a different cache directory
    than the global cache_dir. Used in v10i+ where DROID stays in
    cnn_cache_spatial/droid but Fractal pulls from cnn_cache_v10k_fractal/fractal_rt1.
    """
    overrides = overrides or {}
    out: Dict[str, List[str]] = {}
    for dsid in datasets:
        ds_root = overrides.get(dsid, os.path.join(cache_dir, dsid))
        paths = sorted(glob.glob(os.path.join(ds_root, "*.npz")))
        if not paths:
            log.warning("[%s] no .npz found in %s", dsid, ds_root)
        out[dsid] = paths
    return out


def windows_from_seq(kin: np.ndarray, cnn: np.ndarray, label: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding window seq_len=SEQ_LEN stride=STRIDE. Window label = max over
    window (one failure tick contaminates the whole window)."""
    n = kin.shape[0]
    if n < SEQ_LEN:
        return (np.empty((0, SEQ_LEN, KIN_DIM), dtype=np.float32),
                np.empty((0, SEQ_LEN, CNN_DIM), dtype=np.float16),
                np.empty((0,), dtype=np.float32))
    starts = list(range(0, n - SEQ_LEN + 1, STRIDE))
    W = len(starts)
    X_kin = np.empty((W, SEQ_LEN, KIN_DIM), dtype=np.float32)
    X_cnn = np.empty((W, SEQ_LEN, CNN_DIM), dtype=np.float16)
    y = np.empty((W,), dtype=np.float32)
    for w, s in enumerate(starts):
        X_kin[w] = kin[s:s + SEQ_LEN]
        X_cnn[w] = cnn[s:s + SEQ_LEN]
        y[w] = float(label[s:s + SEQ_LEN].max())
    return X_kin, X_cnn, y


def load_split_windows(paths: List[str], return_seq_id: bool = False):
    """Concat windows from a list of .npz paths.

    kin is padded/clipped to the module-level KIN_DIM so heterogeneous caches
    mix safely: DROID is 15-D and zero-pads to 16; Fractal is already 16-D
    (15 canonical + gripper residual). When return_seq_id=True, also returns a
    per-window int array carrying the source path index (used to group windows
    into MIL bags); the 3-tuple return is preserved when False.
    """
    kin_parts, cnn_parts, y_parts, sid_parts = [], [], [], []
    for pi, p in enumerate(paths):
        with np.load(p) as z:
            kin = z["kin"].astype(np.float32)
            cnn = z["cnn"]  # fp16
            label = z["label"].astype(np.float32)
        # Reconcile kin width to KIN_DIM (zero-pad narrower, clip wider).
        if kin.shape[1] < KIN_DIM:
            pad = np.zeros((kin.shape[0], KIN_DIM - kin.shape[1]), dtype=np.float32)
            kin = np.concatenate([kin, pad], axis=1)
        elif kin.shape[1] > KIN_DIM:
            kin = kin[:, :KIN_DIM]
        # Drop NaN/inf rows in kin (sync edges sometimes leak through).
        finite_mask = np.isfinite(kin).all(axis=1)
        if not finite_mask.all():
            kin = kin[finite_mask]
            cnn = cnn[finite_mask]
            label = label[finite_mask]
        Xk, Xc, yy = windows_from_seq(kin, cnn, label)
        if Xk.shape[0] == 0:
            continue
        kin_parts.append(Xk)
        cnn_parts.append(Xc)
        y_parts.append(yy)
        if return_seq_id:
            sid_parts.append(np.full((Xk.shape[0],), pi, dtype=np.int64))
    if not kin_parts:
        empty = (np.empty((0, SEQ_LEN, KIN_DIM), dtype=np.float32),
                 np.empty((0, SEQ_LEN, CNN_DIM), dtype=np.float16),
                 np.empty((0,), dtype=np.float32))
        return empty + ((np.empty((0,), dtype=np.int64),) if return_seq_id else ())
    out = (np.concatenate(kin_parts, axis=0),
           np.concatenate(cnn_parts, axis=0),
           np.concatenate(y_parts, axis=0))
    if return_seq_id:
        out = out + (np.concatenate(sid_parts, axis=0),)
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def _seq_label_max(path: str) -> int:
    """Read only the `label` array from a cache .npz and return its sequence-
    level binary label (1 if any frame in the sequence is a failure)."""
    with np.load(path) as z:
        return int(np.asarray(z["label"]).max() > 0.5)


def split_indistribution(
    cache: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Per-dataset 70/15/15 split STRATIFIED on sequence-level label
    (label.max() > 0.5). Deterministic via sklearn random_state=SEED.

    Stratification matters because for DROID and Fractal every frame in
    a sequence has the same label, so a non-stratified split could leave
    train/val/test with very different positive ratios. For Reassemble
    (n=48) only 3 sequences have label.max()==0 (pure success), so a
    non-stratified split could put zero successes in val or test.
    Stratify ensures each fold sees both classes proportionally.

    Returns (train_paths, val_paths, test_paths, test_per_ds, val_per_ds).
    The per-dataset dicts enable per-domain threshold tuning + breakdown.
    """
    from sklearn.model_selection import train_test_split  # local: optional dep

    train_p, val_p, test_p = [], [], []
    test_per_ds: Dict[str, List[str]] = {}
    val_per_ds: Dict[str, List[str]] = {}
    for dsid, paths in cache.items():
        nd = len(paths)
        if nd == 0:
            continue
        if nd < 3:
            log.warning("[%s] only %d seq, all to train", dsid, nd)
            train_p.extend(paths)
            test_per_ds[dsid] = []
            val_per_ds[dsid] = []
            continue

        labels = np.array([_seq_label_max(p) for p in paths], dtype=np.int64)
        n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())

        # If a class has < 2 samples, fall back to positional split (sklearn
        # stratify cannot split a class with < 2 samples).
        if min(n_pos, n_neg) < 2:
            log.warning(
                "[%s] %d pos / %d neg seq -> stratified split impossible, "
                "falling back to positional", dsid, n_pos, n_neg,
            )
            n_tr = max(1, int(nd * TRAIN_RATIO_INDIST))
            n_va = max(1, int(nd * VAL_RATIO_INDIST))
            n_tr = min(n_tr, nd - 2)
            n_va = min(n_va, nd - n_tr - 1)
            ds_train = paths[:n_tr]
            ds_val = paths[n_tr:n_tr + n_va]
            ds_test = paths[n_tr + n_va:]
        else:
            test_size = 1.0 - TRAIN_RATIO_INDIST
            ds_train, rest, _, rest_y = train_test_split(
                paths, labels,
                test_size=test_size,
                stratify=labels,
                random_state=SEED,
            )
            # Second split: val vs test. If `rest` has too few samples of one
            # class to stratify (e.g. Reassemble has only 3 negatives total ->
            # rest may have just 1), fall back to a shuffled non-stratified
            # split. Determinism preserved via the same SEED.
            val_frac_of_rest = VAL_RATIO_INDIST / test_size
            rest_y_arr = np.asarray(rest_y)
            min_rest_class = int(min((rest_y_arr == 0).sum(), (rest_y_arr == 1).sum()))
            if min_rest_class < 2:
                log.warning(
                    "[%s] rest set too small to stratify val/test (min class=%d) "
                    "-> shuffled split", dsid, min_rest_class,
                )
                ds_val, ds_test = train_test_split(
                    rest,
                    train_size=val_frac_of_rest,
                    random_state=SEED,
                )
            else:
                ds_val, ds_test = train_test_split(
                    rest,
                    train_size=val_frac_of_rest,
                    stratify=rest_y,
                    random_state=SEED,
                )

        train_p.extend(ds_train)
        val_p.extend(ds_val)
        test_p.extend(ds_test)
        val_per_ds[dsid] = ds_val
        test_per_ds[dsid] = ds_test

        def _pos_ratio(ps: List[str]) -> float:
            if not ps:
                return 0.0
            ys = [_seq_label_max(p) for p in ps]
            return sum(ys) / len(ys)

        log.info(
            "  split[%s] train=%d (pos%%=%.1f) val=%d (pos%%=%.1f) "
            "test=%d (pos%%=%.1f) of %d (pos%%=%.1f)",
            dsid, len(ds_train), 100 * _pos_ratio(ds_train),
            len(ds_val), 100 * _pos_ratio(ds_val),
            len(ds_test), 100 * _pos_ratio(ds_test),
            nd, 100 * (n_pos / nd),
        )
    return train_p, val_p, test_p, test_per_ds, val_per_ds


def split_zeroshot(
    train_cache: Dict[str, List[str]],
    test_cache: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    """85/15 train/val split (per-dataset stratified) from train_cache;
    test = full union of test_cache."""
    train_p, val_p, test_p = [], [], []
    for dsid, paths in train_cache.items():
        nd = len(paths)
        if nd == 0:
            continue
        if nd < 2:
            train_p.extend(paths)
            continue
        n_va = max(1, int(nd * VAL_RATIO_ZEROSHOT))
        n_tr = nd - n_va
        train_p.extend(paths[:n_tr])
        val_p.extend(paths[n_tr:])
        log.info("  split[%s] train=%d val=%d (of %d, zero-shot)", dsid, n_tr, n_va, nd)
    for dsid, paths in test_cache.items():
        test_p.extend(paths)
        log.info("  split[%s] test=%d (zero-shot held-out)", dsid, len(paths))
    return train_p, val_p, test_p


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def fit_kin_scaler(X_kin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = X_kin.reshape(-1, KIN_DIM)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_kin_scaler(X_kin: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if X_kin.shape[0] == 0:
        return X_kin
    out = (X_kin - mean) / std
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def fit_cnn_scaler(X_cnn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit z-score on CNN features (per-channel mean/std) chunk-by-chunk to
    avoid 10+ GB peak allocations on large window sets.
    Uses sum and sum-of-squares accumulators, then derives mean/std at the end.
    """
    cnn_dim = X_cnn.shape[-1]
    n_total = X_cnn.shape[0] * X_cnn.shape[1]  # total timesteps across all windows

    # Per-channel accumulators in float64 for numerical stability.
    sum_acc = np.zeros(cnn_dim, dtype=np.float64)
    sumsq_acc = np.zeros(cnn_dim, dtype=np.float64)

    chunk = 2000  # windows per chunk; ~230 MB per chunk for CNN_DIM=576, seq=50
    for i in range(0, X_cnn.shape[0], chunk):
        block = X_cnn[i:i + chunk].astype(np.float32).reshape(-1, cnn_dim)
        sum_acc += block.sum(axis=0)
        sumsq_acc += (block.astype(np.float64) ** 2).sum(axis=0)

    mean = (sum_acc / n_total).astype(np.float32)
    var = (sumsq_acc / n_total) - (mean.astype(np.float64) ** 2)
    var = np.clip(var, 0.0, None)  # numerical safety
    std = np.sqrt(var).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_cnn_scaler(X_cnn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score chunk-by-chunk to avoid multi-GB peak allocations.
    Full-set numpy ops on (75k, 50, 576) float32 = 8GB peak + bool masks for
    nan_to_num pushing OOM on 16GB systems with other jobs running."""
    if X_cnn.shape[0] == 0:
        return X_cnn
    out = np.empty(X_cnn.shape, dtype=np.float16)
    chunk = 2000
    for i in range(0, X_cnn.shape[0], chunk):
        block = X_cnn[i:i + chunk].astype(np.float32)
        block = (block - mean) / std
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out[i:i + chunk] = block.astype(np.float16)
    return out


def fit_ipca(X_cnn: np.ndarray, n_components: int, batch_size: int = 2048):
    """Fit Incremental PCA on CNN features chunk-by-chunk to dimension-reduce
    the visual stream (e.g. 360 spatial features -> 20 components). Returns the
    fitted IncrementalPCA object. Operates on flattened (N*T, cnn_dim) data."""
    from sklearn.decomposition import IncrementalPCA
    cnn_dim = X_cnn.shape[-1]
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    chunk = 2000  # windows per chunk
    for i in range(0, X_cnn.shape[0], chunk):
        block = X_cnn[i:i + chunk].astype(np.float32).reshape(-1, cnn_dim)
        ipca.partial_fit(block)
    log.info("IPCA fitted: %d components, explained variance ratio sum=%.3f",
             n_components, float(ipca.explained_variance_ratio_.sum()))
    return ipca


def apply_ipca(X_cnn: np.ndarray, ipca, n_components: int) -> np.ndarray:
    """Apply IPCA chunk-by-chunk. Input (N, T, cnn_dim) -> output (N, T, n_components) fp16."""
    if X_cnn.shape[0] == 0:
        return np.empty((0, X_cnn.shape[1] if X_cnn.ndim > 1 else 50, n_components), dtype=np.float16)
    N, T, _ = X_cnn.shape
    out = np.empty((N, T, n_components), dtype=np.float16)
    chunk = 2000
    for i in range(0, N, chunk):
        block = X_cnn[i:i + chunk].astype(np.float32).reshape(-1, X_cnn.shape[-1])
        reduced = ipca.transform(block).astype(np.float16)
        out[i:i + chunk] = reduced.reshape(-1, T, n_components)
    return out


# ----------------------------------------------------------------------------
# Training loop (custom; Trainer does not support custom samplers).
# ----------------------------------------------------------------------------

def make_loader(ds: CachedFeatureDataset, batch_size: int, shuffle: bool,
                weighted: bool, num_workers: int) -> DataLoader:
    if weighted and shuffle:
        labels = ds.labels
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        # weight per sample = 1 / class_freq, balanced sampling
        pos_w = 0.5 / max(n_pos, 1)
        neg_w = 0.5 / max(n_neg, 1)
        weights = np.where(labels > 0.5, pos_w, neg_w).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle)


class FocalLossSmoothed(nn.Module):
    """Focal loss with optional binary label smoothing.
    label_smoothing s in [0, 0.5): targets become y*(1-s) + (1-y)*s.
    s=0 reduces to plain focal."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        if self.label_smoothing > 0.0:
            s = self.label_smoothing
            targets = targets * (1.0 - s) + (1.0 - targets) * s
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def train_model(
    model: nn.Module,
    train_ds: CachedFeatureDataset,
    val_ds: CachedFeatureDataset,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_type: str,
    pos_weight: float,
    use_scheduler: bool,
    weighted_sampler: bool,
    num_workers: int,
    checkpoint_dir: str,
    label_smoothing: float = 0.0,
    swa_start_epoch: int = 0,  # 0 = SWA disabled; >0 = epoch from which to start averaging
    mixup_alpha: float = 0.0,  # 0 = no mixup; ~0.3 = strong mix occasionally
    mil_mode: bool = False,
    use_pcgrad: bool = False,
) -> Dict[str, list]:
    device = get_device()
    log.info("Device: %s", device)
    model = model.to(device)

    if mil_mode:
        from torch.utils.data import WeightedRandomSampler
        log.info("MIL mode ON: bag-level training (bag = sequence, loss bag-level)")
        if mixup_alpha > 0.0:
            log.warning("MIL mode: mixup_alpha forced to 0 (variable bag size)")
            mixup_alpha = 0.0

        def _mil_loader(ds, shuffle, weighted):
            sampler = None
            if weighted and shuffle:
                bl = ds.bag_labels
                n_pos = float(bl.sum()); n_neg = float(len(bl) - n_pos)
                w_pos = 0.5 / max(n_pos, 1.0); w_neg = 0.5 / max(n_neg, 1.0)
                weights = np.where(bl > 0.5, w_pos, w_neg).astype(np.float64)
                sampler = WeightedRandomSampler(weights, num_samples=len(bl), replacement=True)
                shuffle = False
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, collate_fn=collate_bags)

        train_loader = _mil_loader(train_ds, shuffle=True, weighted=weighted_sampler)
        val_loader = _mil_loader(val_ds, shuffle=False, weighted=False)
    else:
        train_loader = make_loader(train_ds, batch_size, shuffle=True,
                                   weighted=weighted_sampler, num_workers=num_workers)
        val_loader = make_loader(val_ds, batch_size, shuffle=False,
                                 weighted=False, num_workers=num_workers)

    if loss_type == "focal":
        criterion = FocalLossSmoothed(alpha=0.25, gamma=2.0, pos_weight=pos_weight,
                                      label_smoothing=label_smoothing)
        log.info("Loss: FocalLoss(alpha=0.25, gamma=2.0, pos_weight=%.3f, label_smoothing=%.3f)",
                 pos_weight, label_smoothing)
    else:
        pw = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        log.info("Loss: BCEWithLogitsLoss(pos_weight=%.3f)", pos_weight)

    base_optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_pcgrad:
        from models.pcgrad import PCGrad
        optimizer = PCGrad(base_optim, reduction="mean")
        log.info("Optimizer: PCGrad(Adam) - per-task gradient projection on conflict")
    else:
        optimizer = base_optim
    scheduler = None
    if use_scheduler:
        # Scheduler steps the underlying Adam, not the PCGrad wrapper.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            base_optim, mode="min", factor=0.5, patience=2,
        )
        log.info("Scheduler: ReduceLROnPlateau(factor=0.5, patience=2)")

    # SWA setup: maintain a running average of weights from swa_start_epoch onward.
    swa_model = None
    if swa_start_epoch > 0:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        log.info("SWA enabled: averaging weights from epoch %d onward", swa_start_epoch)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            if mil_mode:
                x_kin, x_cnn, mask, y, nl_emb, is_droid = batch
                x_kin = x_kin.to(device); x_cnn = x_cnn.to(device)
                mask = mask.to(device); y = y.to(device)
                nl_emb = nl_emb.to(device); is_droid = is_droid.to(device)
                optimizer.zero_grad()
                logits, _ = model(x_kin, x_cnn, mask, nl_emb=nl_emb, is_droid=is_droid)
                loss = criterion(logits, y)
                # v10l: PCGrad split-loss when both datasets are in the batch.
                if use_pcgrad:
                    m_d = is_droid.bool()
                    if bool(m_d.any()) and bool((~m_d).any()):
                        loss_d = criterion(logits[m_d], y[m_d])
                        loss_f = criterion(logits[~m_d], y[~m_d])
                        optimizer.pc_backward([loss_d, loss_f])
                    else:
                        loss.backward()
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running += loss.item()
                continue

            x_kin, x_cnn, y = batch
            x_kin = x_kin.to(device); x_cnn = x_cnn.to(device); y = y.to(device)

            # Temporal MixUp: linearly interpolate two random batches
            # (Beta(0.3, 0.3) is bimodal at 0/1 -> mostly clean samples with
            # occasional strong mix; softens decision boundary).
            if mixup_alpha > 0.0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(x_kin.shape[0], device=device)
                x_kin = lam * x_kin + (1.0 - lam) * x_kin[idx]
                x_cnn = lam * x_cnn + (1.0 - lam) * x_cnn[idx]
                y = lam * y + (1.0 - lam) * y[idx]

            optimizer.zero_grad()
            logits = model(x_kin, x_cnn)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        # SWA: update averaged weights from configured epoch onward.
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        model.eval()
        v_running = 0.0
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                if mil_mode:
                    x_kin, x_cnn, mask, y, nl_emb, is_droid = batch
                    x_kin = x_kin.to(device); x_cnn = x_cnn.to(device)
                    mask = mask.to(device); y = y.to(device)
                    nl_emb = nl_emb.to(device); is_droid = is_droid.to(device)
                    logits, _ = model(x_kin, x_cnn, mask, nl_emb=nl_emb, is_droid=is_droid)
                else:
                    x_kin, x_cnn, y = batch
                    x_kin = x_kin.to(device); x_cnn = x_cnn.to(device); y = y.to(device)
                    logits = model(x_kin, x_cnn)
                v_running += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.shape[0]
        val_loss = v_running / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        log.info("Epoch %3d/%d | train=%.4f val=%.4f val_acc=%.3f",
                 epoch, epochs, train_loss, val_loss, val_acc)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            log.info("  -> saved best (val=%.4f)", best_val)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stop at epoch %d", epoch)
                break

    # SWA: if enabled, replace best weights with the running average and
    # recompute BatchNorm statistics. update_bn is a no-op for architectures
    # without BN layers; the call is wrapped to be safe in that case.
    # Use the SWA model as final unless its val loss is materially worse.
    if swa_model is not None and epoch >= swa_start_epoch:
        # No BatchNorm in these models, so update_bn is a no-op; skip it in MIL
        # mode where its positional-arg call convention does not match forward().
        if not mil_mode:
            from torch.optim.swa_utils import update_bn
            try:
                update_bn(train_loader, swa_model, device=device)
            except Exception:
                pass
        # Evaluate SWA on val set; keep it only if it's at least as good.
        swa_model.eval()
        v_running_swa = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if mil_mode:
                    x_kin, x_cnn, mask, y, nl_emb, is_droid = batch
                    x_kin = x_kin.to(device); x_cnn = x_cnn.to(device)
                    mask = mask.to(device); y = y.to(device)
                    nl_emb = nl_emb.to(device); is_droid = is_droid.to(device)
                    logits, _ = swa_model(x_kin, x_cnn, mask, nl_emb=nl_emb, is_droid=is_droid)
                else:
                    x_kin, x_cnn, y = batch
                    x_kin = x_kin.to(device); x_cnn = x_cnn.to(device); y = y.to(device)
                    logits = swa_model(x_kin, x_cnn)
                v_running_swa += criterion(logits, y).item()
        swa_val = v_running_swa / max(len(val_loader), 1)
        log.info("SWA val_loss=%.4f vs best_val=%.4f", swa_val, best_val)
        if swa_val <= best_val * 1.05:  # accept SWA if within 5% of best single-model val
            log.info("  -> using SWA-averaged weights as final model")
            torch.save(swa_model.module.state_dict(), best_path)
            model.load_state_dict(swa_model.module.state_dict())
            return history

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return history


# --------------------------------------------------------------------------
# Evaluation + threshold tuning
# --------------------------------------------------------------------------

@torch.no_grad()
def collect_probs(model: nn.Module, ds: CachedFeatureDataset, batch_size: int):
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    model.eval()
    for x_kin, x_cnn, y in loader:
        x_kin = x_kin.to(device); x_cnn = x_cnn.to(device)
        p = model.predict_proba(x_kin, x_cnn)
        probs.extend(p.cpu().numpy())
        labels.extend(y.numpy())
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)


@torch.no_grad()
def mil_collect_probs(model: nn.Module, bag_ds, batch_size: int):
    """Bag-level inference for the MIL model. Returns (probs, labels) per bag."""
    device = next(model.parameters()).device
    loader = DataLoader(bag_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bags)
    probs, labels = [], []
    model.eval()
    for x_kin, x_cnn, mask, y, nl_emb, is_droid in loader:
        x_kin = x_kin.to(device); x_cnn = x_cnn.to(device); mask = mask.to(device)
        nl_emb = nl_emb.to(device); is_droid = is_droid.to(device)
        p, _ = model.predict_proba(x_kin, x_cnn, mask, nl_emb=nl_emb, is_droid=is_droid)
        probs.extend(p.cpu().numpy())
        labels.extend(y.numpy())
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _path_to_dataset(cache: Dict[str, List[str]]) -> Dict[str, str]:
    """abspath(npz) -> dataset id, across every dataset in the cache index."""
    return {os.path.abspath(p): d for d, ps in cache.items() for p in ps}


def _bag_unique_seqs(sid_arr: Optional[np.ndarray]) -> List[int]:
    """Unique seq ids in first-seen order, matching CachedSequenceBagDataset."""
    if sid_arr is None or len(sid_arr) == 0:
        return []
    return list(dict.fromkeys(sid_arr.tolist()))


def bag_dataset_ids_for(paths: List[str], sid_arr, path_to_ds: Dict[str, str]):
    """Per-bag dataset id, ordered to match the bag dataset's unique-seq order."""
    uniq = _bag_unique_seqs(sid_arr)
    if not uniq:
        return None
    return [path_to_ds.get(os.path.abspath(paths[s]), "unknown") for s in uniq]


def bag_nl_emb_for(paths: List[str], sid_arr, path_to_ds: Dict[str, str]):
    """Per-bag 512-D NL embedding loaded from the Fractal .npz; zeros for DROID."""
    uniq = _bag_unique_seqs(sid_arr)
    if not uniq:
        return None
    out = np.zeros((len(uniq), 512), dtype=np.float32)
    for i, s in enumerate(uniq):
        p = paths[s]
        if path_to_ds.get(os.path.abspath(p), "unknown") == "fractal_rt1":
            try:
                with np.load(p) as z:
                    if "nl_emb" in z.files:
                        out[i] = z["nl_emb"].astype(np.float32)
            except Exception as e:
                log.warning("Could not load nl_emb from %s: %s", p, e)
    return out


def bag_is_droid_for(paths: List[str], sid_arr, path_to_ds: Dict[str, str]):
    """Per-bag bool mask: True for DROID bags (routes the learnable NL null token)."""
    uniq = _bag_unique_seqs(sid_arr)
    if not uniq:
        return None
    return np.array(
        [path_to_ds.get(os.path.abspath(paths[s]), "unknown") == "droid" for s in uniq],
        dtype=bool,
    )


def best_threshold_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve, f1_score
    if len(np.unique(labels)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(labels, probs)
    # f1 per threshold; t has len(p)-1
    f1 = (2 * p * r) / np.clip(p + r, 1e-9, None)
    f1 = f1[:-1]  # align to thresholds
    if len(f1) == 0:
        return 0.5
    best = int(np.argmax(f1))
    return float(t[best])


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve,
    )
    preds = (probs >= threshold).astype(np.float64)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(labels, preds, zero_division=0))
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc = (fpr.tolist(), tpr.tolist())
    except ValueError:
        roc = ([], [])
    rep = classification_report(labels, preds, labels=[0, 1],
                                target_names=["success", "failure"], zero_division=0)
    return {"accuracy": acc, "auc": auc, "f1": f1,
            "confusion_matrix": cm, "roc": roc,
            "classification_report": rep, "threshold": threshold}


def save_plots(history: Dict, metrics: Dict, out_dir: str, header: str) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    epochs_x = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_x, history["train_loss"], label="train", linewidth=2)
    ax.plot(epochs_x, history["val_loss"], label="val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Loss: {header}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150); plt.close(fig)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Success", "Failure"], yticklabels=["Success", "Failure"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"CM (acc={metrics['accuracy']:.3f}, auc={metrics['auc']:.3f}, "
                 f"f1={metrics['f1']:.3f}, thr={metrics['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    fpr, tpr = metrics["roc"]
    if fpr and not np.isnan(metrics["auc"]):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__),
                                                       "results", "cnn_cache"))
    p.add_argument("--train-datasets", default="reassemble,droid,fractal_rt1")
    p.add_argument("--test-datasets", default="reassemble,droid,fractal_rt1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--loss", choices=["focal", "bce"], default="focal")
    p.add_argument("--use-scheduler", action="store_true",
                   help="ReduceLROnPlateau (factor=0.5, patience=2)")
    p.add_argument("--weighted-sampler", action="store_true",
                   help="WeightedRandomSampler on train (balances classes)")
    p.add_argument("--kin-noise-std", type=float, default=0.0,
                   help="Gaussian noise std on X_kin in train __getitem__")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__),
                                                        "results", "multimodal_indist"))
    p.add_argument("--smoke", action="store_true",
                   help="3 seq/dataset, 2 epochs (subsamples cache index)")
    p.add_argument("--stride", type=int, default=10,
                   help="sliding window stride (default 10; raise to 25 to reduce overfit)")
    p.add_argument("--cnn-dim", type=int, default=576,
                   help="CNN feature dim per frame (576 MobileNetV3, 384 DINOv2-S, 512 ResNet18)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="binary label smoothing factor (e.g. 0.05 maps 0->0.05, 1->0.95)")
    p.add_argument("--swa-start-epoch", type=int, default=0,
                   help="epoch from which to start Stochastic Weight Averaging (0 = disabled)")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="MixUp Beta(alpha, alpha) parameter (0 = disabled, 0.3 = recommended)")
    p.add_argument("--ipca-components", type=int, default=0,
                   help="reduce CNN features to N components via IncrementalPCA (0 = disabled)")
    p.add_argument("--late-fusion", action="store_true",
                   help="use LateFusionLSTM (separate kin/visual streams) instead of early fusion")
    p.add_argument("--attn-pool", action="store_true",
                   help="(LateFusionLSTM only) use learnable attention pool over time "
                        "instead of last-timestep readout. Better for episode-level "
                        "labels (DROID/Fractal) where any frame in the window is informative.")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"random seed (default: {SEED}). Override to obtain "
                        "independent runs for ensemble diversity.")
    p.add_argument("--no-pos-weight", action="store_true",
                   help="disable auto pos_weight in FocalLoss (rely on alpha+gamma only)")
    # -- MIL (v10l) ----------------------------------------------------------
    p.add_argument("--mil-attention", action="store_true",
                   help="Multiple Instance Learning: bag = sequence, instances = windows. "
                        "Loss is bag-level against the sequence label.")
    p.add_argument("--mil-pool-mode", type=str, default="gated",
                   choices=["gated", "max", "mean"],
                   help="Bag aggregator when --mil-attention is on (default gated, Ilse 2018).")
    p.add_argument("--lstm-direction", type=str, default="bi", choices=["bi", "uni"],
                   help="WindowEncoder LSTM direction. 'uni' (v10h+) prevents the backward "
                        "pass from leaking terminal-failure state to early frames.")
    p.add_argument("--droid-tail-fraction", type=float, default=1.0,
                   help="Restrict each DROID FAILURE bag to its last frac*len windows (v10h+). "
                        "1.0 = no filtering.")
    p.add_argument("--fractal-tail-fraction", type=float, default=1.0,
                   help="Same as --droid-tail-fraction for Fractal failure bags.")
    p.add_argument("--fractal-cache-dir", type=str, default=None,
                   help="Override --cache-dir for Fractal only (v10i+: 16-D kin + nl_emb).")
    p.add_argument("--use-nl-concat", action="store_true",
                   help="v10l: project the 512-D NL embedding (Xavier) and concat to each "
                        "instance before the bag pool; DROID uses a learnable null token.")
    p.add_argument("--nl-concat-dim", type=int, default=64,
                   help="Projection dim for --use-nl-concat (default 64).")
    p.add_argument("--use-pcgrad", action="store_true",
                   help="v10l: PCGrad gradient surgery (split loss per dataset, project "
                        "conflicting gradients) to stop DROID from cancelling Fractal.")
    p.add_argument("--balanced-train", action="store_true",
                   help="Use the Mosaico pools split: 70/15/15 per dataset, train balanced to "
                        "--success-ratio, val/test at natural prior. Requires --pools-json.")
    p.add_argument("--success-ratio", type=float, default=0.65,
                   help="Train success fraction per dataset when --balanced-train (default 0.65).")
    p.add_argument("--pools-json", type=str, default=None,
                   help="Path to pools_via_mosaico.json (success/failure names per dataset "
                        "from the Mosaico query layer).")
    return p.parse_args()


def _detect_kin_dim(cache: Dict[str, List[str]], default: int = 16) -> int:
    """Max kin width across one sample npz per dataset (DROID 15, Fractal 16)."""
    dims = []
    for _d, ps in cache.items():
        if ps:
            try:
                with np.load(ps[0]) as z:
                    dims.append(int(z["kin"].shape[1]))
            except Exception:
                pass
    return max(dims) if dims else default


def main_mil(args) -> int:
    """v10l MIL training path: bag-level LateFusionLSTMMIL on Mosaico pools."""
    global SEED, STRIDE, CNN_DIM, KIN_DIM
    SEED = args.seed
    set_seed(SEED)
    STRIDE = args.stride
    CNN_DIM = args.cnn_dim

    train_dsids = [d.strip() for d in args.train_datasets.split(",") if d.strip()]
    test_dsids = [d.strip() for d in args.test_datasets.split(",") if d.strip()]

    if args.smoke:
        args.epochs = 2
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    log.info("=== Cached training (MIL / v10l) ===")
    log.info("train=%s test=%s epochs=%d batch=%d loss=%s pool=%s lstm=%s",
             train_dsids, test_dsids, args.epochs, args.batch_size, args.loss,
             args.mil_pool_mode, args.lstm_direction)

    overrides = {}
    if args.fractal_cache_dir:
        overrides["fractal_rt1"] = os.path.join(args.fractal_cache_dir, "fractal_rt1")
        log.info("Fractal cache override: %s", overrides["fractal_rt1"])
    all_dsids = list(dict.fromkeys(train_dsids + test_dsids))
    cache = index_cache(args.cache_dir, all_dsids, overrides=overrides)
    path_to_ds = _path_to_dataset(cache)

    KIN_DIM = _detect_kin_dim(cache)
    log.info("MIL config: KIN_DIM=%d SEQ_LEN=%d STRIDE=%d CNN_DIM=%d",
             KIN_DIM, SEQ_LEN, STRIDE, CNN_DIM)

    if args.smoke:
        for d in cache:
            cache[d] = cache[d][:3]

    unified = {d: cache.get(d, []) for d in train_dsids}
    if args.pools_json and args.balanced_train:
        log.info("Split via Mosaico pools (train balanced %.2f, val/test natural prior)",
                 args.success_ratio)
        train_paths, val_paths, test_paths, test_per_ds, val_per_ds = split_by_pools_proportional(
            unified, args.pools_json, success_ratio_train=args.success_ratio, seed=SEED)
    else:
        train_paths, val_paths, test_paths, test_per_ds, val_per_ds = split_indistribution(unified)
    log.info("Sequence counts: train=%d val=%d test=%d",
             len(train_paths), len(val_paths), len(test_paths))

    log.info("Loading + windowing (train/val/test)...")
    Xk_tr, Xc_tr, y_tr, sid_tr = load_split_windows(train_paths, return_seq_id=True)
    Xk_va, Xc_va, y_va, sid_va = load_split_windows(val_paths, return_seq_id=True)
    Xk_te, Xc_te, y_te, sid_te = load_split_windows(test_paths, return_seq_id=True)
    log.info("  train windows: kin=%s cnn=%s pos=%.3f", Xk_tr.shape, Xc_tr.shape,
             float(y_tr.mean()) if len(y_tr) else 0.0)
    if Xk_tr.shape[0] == 0 or Xk_va.shape[0] == 0:
        log.error("Empty train or val tensors; aborting.")
        return 1

    mean, std = fit_kin_scaler(Xk_tr)
    Xk_tr = apply_kin_scaler(Xk_tr, mean, std)
    Xk_va = apply_kin_scaler(Xk_va, mean, std)
    Xk_te = apply_kin_scaler(Xk_te, mean, std)
    cnn_mean, cnn_std = fit_cnn_scaler(Xc_tr)
    log.info("CNN scaler: mean range=[%.3f, %.3f] std range=[%.3f, %.3f]",
             float(cnn_mean.min()), float(cnn_mean.max()),
             float(cnn_std.min()), float(cnn_std.max()))
    Xc_tr = apply_cnn_scaler(Xc_tr, cnn_mean, cnn_std)
    Xc_va = apply_cnn_scaler(Xc_va, cnn_mean, cnn_std)
    Xc_te = apply_cnn_scaler(Xc_te, cnn_mean, cnn_std)
    effective_cnn_dim = Xc_tr.shape[-1]

    np.savez(os.path.join(args.output_dir, "scaler.npz"),
             mean=mean, std=std, cnn_mean=cnn_mean, cnn_std=cnn_std)

    tail_for: Dict[str, float] = {}
    if args.droid_tail_fraction < 1.0:
        tail_for["droid"] = float(args.droid_tail_fraction)
    if args.fractal_tail_fraction < 1.0:
        tail_for["fractal_rt1"] = float(args.fractal_tail_fraction)

    def _bag_ds(Xk, Xc, y, sid, paths):
        return CachedSequenceBagDataset(
            Xk, Xc, y, sid,
            bag_dataset_ids=bag_dataset_ids_for(paths, sid, path_to_ds),
            tail_fraction_for=tail_for,
            bag_nl_emb=bag_nl_emb_for(paths, sid, path_to_ds),
            bag_is_droid=bag_is_droid_for(paths, sid, path_to_ds),
        )

    train_ds = _bag_ds(Xk_tr, Xc_tr, y_tr, sid_tr, train_paths)
    val_ds = _bag_ds(Xk_va, Xc_va, y_va, sid_va, val_paths)
    test_ds = _bag_ds(Xk_te, Xc_te, y_te, sid_te, test_paths) if Xk_te.shape[0] > 0 else None
    log.info("MIL datasets: train bags=%d val bags=%d test bags=%d",
             len(train_ds), len(val_ds), len(test_ds) if test_ds else 0)
    if train_ds.filter_stats.get("applied"):
        fs = train_ds.filter_stats
        log.info("Window filtering (train): %d bags, avg len %.1f -> %.1f, %s",
                 fs["n_bags_filtered"], fs["avg_len_before"], fs["avg_len_after"],
                 fs["tail_fraction_for"])

    bl = train_ds.bag_labels
    n_pos = float(bl.sum()); n_neg = float(len(bl) - n_pos)
    pos_weight = 1.0 if args.no_pos_weight else float(n_neg / max(n_pos, 1))
    log.info("Class balance (bags): n_pos=%d n_neg=%d pos_weight=%.3f (no_pos_weight=%s)",
             int(n_pos), int(n_neg), pos_weight, args.no_pos_weight)

    model = LateFusionLSTMMIL(
        kin_dim=KIN_DIM, cnn_dim=effective_cnn_dim,
        kin_hidden=args.hidden // 4, vis_hidden=args.hidden // 4,
        num_layers=1, dropout=args.dropout,
        bidirectional=(args.lstm_direction == "bi"),
        attn_pool=args.attn_pool,
        mil_attn_hidden=args.hidden // 2, head_hidden=args.hidden // 2,
        pool_mode=args.mil_pool_mode,
        use_nl_concat=args.use_nl_concat, nl_concat_dim=args.nl_concat_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: LateFusionLSTMMIL (%d params, nl_concat=%s pcgrad=%s)",
             n_params, args.use_nl_concat, args.use_pcgrad)

    t0 = time.time()
    history = train_model(
        model, train_ds, val_ds,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience,
        loss_type=args.loss, pos_weight=pos_weight,
        use_scheduler=args.use_scheduler, weighted_sampler=args.weighted_sampler,
        num_workers=args.num_workers, checkpoint_dir=checkpoint_dir,
        label_smoothing=args.label_smoothing, swa_start_epoch=args.swa_start_epoch,
        mixup_alpha=0.0, mil_mode=True, use_pcgrad=args.use_pcgrad,
    )
    train_time = time.time() - t0
    log.info("Training done in %.0fs (%d epochs)", train_time, len(history["train_loss"]))

    val_probs, val_labels = mil_collect_probs(model, val_ds, args.batch_size)
    thr = best_threshold_f1(val_probs, val_labels)
    log.info("Tuned threshold (max F1 on val): %.3f", thr)

    if test_ds is not None:
        test_probs, test_labels = mil_collect_probs(model, test_ds, args.batch_size)
        metrics = evaluate(test_probs, test_labels, threshold=thr)
        metrics_at_05 = evaluate(test_probs, test_labels, threshold=0.5)
        log.info("Test overall (thr=%.3f): acc=%.4f auc=%.4f f1=%.4f",
                 thr, metrics["accuracy"], metrics["auc"], metrics["f1"])
        log.info("\n%s", metrics["classification_report"])
    else:
        metrics = {"accuracy": 0, "auc": float("nan"), "f1": 0,
                   "confusion_matrix": [[0, 0], [0, 0]], "roc": ([], []),
                   "classification_report": "no test split", "threshold": thr}
        metrics_at_05 = metrics

    per_ds_metrics: Dict[str, dict] = {}
    if test_per_ds:
        log.info("=== Per-dataset test breakdown (cross-format table) ===")
        for dsid, ds_paths in test_per_ds.items():
            if not ds_paths:
                continue
            ds_thr = thr
            ds_val_paths = val_per_ds.get(dsid, [])
            if ds_val_paths:
                Xk_v, Xc_v, y_v, sid_v = load_split_windows(ds_val_paths, return_seq_id=True)
                if Xk_v.shape[0] > 0:
                    Xk_v = apply_kin_scaler(Xk_v, mean, std)
                    Xc_v = apply_cnn_scaler(Xc_v, cnn_mean, cnn_std)
                    vds = _bag_ds(Xk_v, Xc_v, y_v, sid_v, ds_val_paths)
                    vp, vl = mil_collect_probs(model, vds, args.batch_size)
                    ds_thr = best_threshold_f1(vp, vl)
            Xk_d, Xc_d, y_d, sid_d = load_split_windows(ds_paths, return_seq_id=True)
            if Xk_d.shape[0] == 0:
                continue
            Xk_d = apply_kin_scaler(Xk_d, mean, std)
            Xc_d = apply_cnn_scaler(Xc_d, cnn_mean, cnn_std)
            dds = _bag_ds(Xk_d, Xc_d, y_d, sid_d, ds_paths)
            dp, dl = mil_collect_probs(model, dds, args.batch_size)
            ds_m = evaluate(dp, dl, threshold=ds_thr)
            ds_m_global = evaluate(dp, dl, threshold=thr)
            mp = float(dp[dl > 0.5].mean()) if (dl > 0.5).any() else 0.0
            mn = float(dp[dl <= 0.5].mean()) if (dl <= 0.5).any() else 0.0
            per_ds_metrics[dsid] = {
                k: ds_m[k] for k in ["accuracy", "auc", "f1", "threshold", "confusion_matrix"]
            }
            per_ds_metrics[dsid]["f1_global_thr"] = ds_m_global["f1"]
            per_ds_metrics[dsid]["class_separation"] = mp - mn
            per_ds_metrics[dsid]["mean_prob_failure_on_pos"] = mp
            per_ds_metrics[dsid]["mean_prob_failure_on_neg"] = mn
            log.info("  [%s] bags=%d pos=%.3f | AUC=%.4f | thr=%.3f F1=%.4f | class_sep=%.4f",
                     dsid, len(dl), float((dl > 0.5).mean()) if len(dl) else 0.0,
                     ds_m["auc"], ds_thr, ds_m["f1"], mp - mn)

    save_plots(history, metrics, args.output_dir,
               header=f"train={'+'.join(train_dsids)} test={'+'.join(test_dsids)} (MIL/v10l)")

    results = {
        "config": {
            "mode": "in-distribution", "train_datasets": train_dsids,
            "test_datasets": test_dsids, "cache_dir": args.cache_dir,
            "fractal_cache_dir": args.fractal_cache_dir,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "dropout": args.dropout, "hidden": args.hidden, "loss": args.loss,
            "pos_weight": pos_weight, "no_pos_weight": args.no_pos_weight,
            "use_scheduler": args.use_scheduler, "weighted_sampler": args.weighted_sampler,
            "kin_noise_std": args.kin_noise_std, "swa_start_epoch": args.swa_start_epoch,
            "ipca_components": 0, "attn_pool": args.attn_pool,
            "mil_attention": True, "mil_pool_mode": args.mil_pool_mode,
            "lstm_direction": args.lstm_direction,
            "droid_tail_fraction": args.droid_tail_fraction,
            "fractal_tail_fraction": args.fractal_tail_fraction,
            "use_nl_concat": args.use_nl_concat, "nl_concat_dim": args.nl_concat_dim,
            "use_pcgrad": args.use_pcgrad, "balanced_train": args.balanced_train,
            "success_ratio": args.success_ratio, "kin_dim_runtime": KIN_DIM,
            "cnn_dim": args.cnn_dim, "seed": SEED, "seq_len": SEQ_LEN, "stride": STRIDE,
        },
        "sequence_counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "window_counts": {"train": int(Xk_tr.shape[0]), "val": int(Xk_va.shape[0]),
                          "test": int(Xk_te.shape[0])},
        "train_time_sec": train_time,
        "test_metrics_tuned": {k: metrics[k] for k in
                               ["accuracy", "auc", "f1", "threshold",
                                "confusion_matrix", "classification_report"]},
        "test_metrics_at_0.5": {k: metrics_at_05[k] for k in
                                ["accuracy", "auc", "f1",
                                 "confusion_matrix", "classification_report"]},
        "per_dataset_test_metrics": per_ds_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output_dir)
    return 0


def main() -> int:
    args = parse_args()
    if args.mil_attention:
        return main_mil(args)
    # split_indistribution references the module-level SEED, propagate any CLI override.
    global SEED, STRIDE, CNN_DIM
    SEED = args.seed
    set_seed(SEED)

    # Apply CLI overrides to module-level constants used by window builders.
    STRIDE = args.stride
    CNN_DIM = args.cnn_dim
    log.info("Window config: SEQ_LEN=%d STRIDE=%d CNN_DIM=%d", SEQ_LEN, STRIDE, CNN_DIM)

    train_dsids = [d.strip() for d in args.train_datasets.split(",") if d.strip()]
    test_dsids = [d.strip() for d in args.test_datasets.split(",") if d.strip()]
    is_zeroshot = set(train_dsids).isdisjoint(set(test_dsids))

    if args.smoke:
        args.epochs = 2

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    log.info("=== Cached training ===")
    log.info("mode=%s train=%s test=%s epochs=%d batch=%d loss=%s",
             "zero-shot" if is_zeroshot else "in-distribution",
             train_dsids, test_dsids, args.epochs, args.batch_size, args.loss)

    train_cache = index_cache(args.cache_dir, train_dsids)
    test_cache = index_cache(args.cache_dir, test_dsids)

    if args.smoke:
        for d in train_cache:
            train_cache[d] = train_cache[d][:3]
        for d in test_cache:
            test_cache[d] = test_cache[d][:3]

    if is_zeroshot:
        train_paths, val_paths, test_paths = split_zeroshot(train_cache, test_cache)
        test_per_ds: Dict[str, List[str]] = {}  # no per-dataset breakdown in LOO mode
        val_per_ds: Dict[str, List[str]] = {}
    else:
        # in-distribution: always train on all 3 datasets together; report per-dataset test breakdown
        unified = {d: train_cache.get(d, []) for d in train_dsids}
        train_paths, val_paths, test_paths, test_per_ds, val_per_ds = split_indistribution(unified)

    log.info("Sequence counts: train=%d val=%d test=%d",
             len(train_paths), len(val_paths), len(test_paths))

    log.info("Loading + windowing train cache...")
    t0 = time.time()
    X_kin_tr, X_cnn_tr, y_tr = load_split_windows(train_paths)
    log.info("  train windows: kin=%s cnn=%s y=%s pos=%.3f (%.1fs)",
             X_kin_tr.shape, X_cnn_tr.shape, y_tr.shape,
             float(y_tr.mean()) if len(y_tr) else 0.0, time.time() - t0)

    log.info("Loading + windowing val cache...")
    t0 = time.time()
    X_kin_va, X_cnn_va, y_va = load_split_windows(val_paths)
    log.info("  val   windows: kin=%s pos=%.3f (%.1fs)",
             X_kin_va.shape, float(y_va.mean()) if len(y_va) else 0.0, time.time() - t0)

    log.info("Loading + windowing test cache...")
    t0 = time.time()
    X_kin_te, X_cnn_te, y_te = load_split_windows(test_paths)
    log.info("  test  windows: kin=%s pos=%.3f (%.1fs)",
             X_kin_te.shape, float(y_te.mean()) if len(y_te) else 0.0, time.time() - t0)

    if X_kin_tr.shape[0] == 0 or X_kin_va.shape[0] == 0:
        log.error("Empty train or val tensors; aborting.")
        return 1

    # Fit kin scaler on train, apply to all
    mean, std = fit_kin_scaler(X_kin_tr)
    X_kin_tr = apply_kin_scaler(X_kin_tr, mean, std)
    X_kin_va = apply_kin_scaler(X_kin_va, mean, std)
    X_kin_te = apply_kin_scaler(X_kin_te, mean, std)

    # Fit CNN scaler on train, apply to all (per-channel z-score)
    cnn_mean, cnn_std = fit_cnn_scaler(X_cnn_tr)
    log.info("CNN scaler: mean range=[%.3f, %.3f] std range=[%.3f, %.3f]",
             float(cnn_mean.min()), float(cnn_mean.max()),
             float(cnn_std.min()), float(cnn_std.max()))
    X_cnn_tr = apply_cnn_scaler(X_cnn_tr, cnn_mean, cnn_std)
    X_cnn_va = apply_cnn_scaler(X_cnn_va, cnn_mean, cnn_std)
    X_cnn_te = apply_cnn_scaler(X_cnn_te, cnn_mean, cnn_std)

    # IPCA dimensionality reduction (e.g. 360 spatial features -> 20 components).
    # Balances visual vs kin (15) dimensionality; reduces capacity for the LSTM
    # to overfit on a high-dim visual stream. Fitted on TRAIN only (no leakage).
    ipca = None
    effective_cnn_dim = X_cnn_tr.shape[-1]
    if args.ipca_components > 0:
        log.info("Fitting IncrementalPCA: %d components on train CNN features...",
                 args.ipca_components)
        t0 = time.time()
        ipca = fit_ipca(X_cnn_tr, args.ipca_components)
        log.info("  IPCA fit done in %.0fs", time.time() - t0)
        X_cnn_tr = apply_ipca(X_cnn_tr, ipca, args.ipca_components)
        X_cnn_va = apply_ipca(X_cnn_va, ipca, args.ipca_components)
        X_cnn_te = apply_ipca(X_cnn_te, ipca, args.ipca_components)
        effective_cnn_dim = args.ipca_components
        log.info("  Post-IPCA shapes: train=%s val=%s test=%s",
                 X_cnn_tr.shape, X_cnn_va.shape, X_cnn_te.shape)

    # Save scaler + IPCA components for inference reproducibility.
    save_dict = {"mean": mean, "std": std, "cnn_mean": cnn_mean, "cnn_std": cnn_std}
    if ipca is not None:
        save_dict["ipca_components"] = ipca.components_.astype(np.float32)
        save_dict["ipca_mean"] = ipca.mean_.astype(np.float32)
        save_dict["ipca_n_components"] = np.array(args.ipca_components)
    np.savez(os.path.join(args.output_dir, "scaler.npz"), **save_dict)

    # Datasets
    train_ds = CachedFeatureDataset(X_kin_tr, X_cnn_tr, y_tr,
                                    kin_noise_std=args.kin_noise_std)
    val_ds = CachedFeatureDataset(X_kin_va, X_cnn_va, y_va)
    test_ds = (CachedFeatureDataset(X_kin_te, X_cnn_te, y_te)
               if X_kin_te.shape[0] > 0 else None)

    # Auto pos_weight from train labels (can be disabled via --no-pos-weight
    # when stacking with focal loss alpha+gamma to avoid over-weighting failure
    # class, which caused v2's collapse).
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = 1.0 if args.no_pos_weight else float(n_neg / max(n_pos, 1))
    log.info("Class balance: n_pos=%d n_neg=%d pos_weight=%.3f (no_pos_weight=%s)",
             int(n_pos), int(n_neg), pos_weight, args.no_pos_weight)

    # Model: Late Fusion (separate kin/visual streams) or Early Fusion (concat at input)
    if args.late_fusion:
        model = LateFusionLSTM(
            kin_dim=KIN_DIM, cnn_dim=effective_cnn_dim,
            kin_hidden=args.hidden // 4, vis_hidden=args.hidden // 4,
            num_layers=1, dropout=args.dropout, bidirectional=True,
            fc_dropout=args.dropout,
            attn_pool=args.attn_pool,
        )
        log.info("Model: LateFusionLSTM (kin_hidden=%d, vis_hidden=%d, cnn_dim=%d, attn_pool=%s)",
                 args.hidden // 4, args.hidden // 4, effective_cnn_dim, args.attn_pool)
    else:
        model = CachedFeatureLSTM(
            kin_dim=KIN_DIM, cnn_dim=effective_cnn_dim,
            hidden_size=args.hidden, num_layers=2,
            dropout=args.dropout, bidirectional=True,
            fc_dropout=max(args.dropout - 0.1, 0.2),
        )
        log.info("Model: CachedFeatureLSTM (early fusion, cnn_dim=%d)", effective_cnn_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: CachedFeatureLSTM (%d trainable params)", n_params)

    t0 = time.time()
    history = train_model(
        model, train_ds, val_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_type=args.loss,
        pos_weight=pos_weight,
        use_scheduler=args.use_scheduler,
        weighted_sampler=args.weighted_sampler,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        label_smoothing=args.label_smoothing,
        swa_start_epoch=args.swa_start_epoch,
        mixup_alpha=args.mixup_alpha,
    )
    train_time = time.time() - t0
    log.info("Training done in %.0fs (%d epochs)", train_time, len(history["train_loss"]))

    # Threshold tuning on val, then apply to test
    val_probs, val_labels = collect_probs(model, val_ds, args.batch_size)
    thr = best_threshold_f1(val_probs, val_labels)
    log.info("Tuned threshold (max F1 on val): %.3f", thr)

    if test_ds is not None:
        test_probs, test_labels = collect_probs(model, test_ds, args.batch_size)
        metrics = evaluate(test_probs, test_labels, threshold=thr)
        # Also compute @0.5 for reference
        metrics_at_05 = evaluate(test_probs, test_labels, threshold=0.5)
        log.info("Test overall (thr=%.3f tuned): acc=%.4f auc=%.4f f1=%.4f",
                 thr, metrics["accuracy"], metrics["auc"], metrics["f1"])
        log.info("Test overall (thr=0.5):         acc=%.4f auc=%.4f f1=%.4f",
                 metrics_at_05["accuracy"], metrics_at_05["auc"], metrics_at_05["f1"])
        log.info("\n%s", metrics["classification_report"])
    else:
        metrics = {"accuracy": 0, "auc": float("nan"), "f1": 0,
                   "confusion_matrix": [[0, 0], [0, 0]], "roc": ([], []),
                   "classification_report": "no test split", "threshold": thr}
        metrics_at_05 = metrics

    # Per-dataset breakdown (in-distribution mode only).
    # Cross-format evidence: same model evaluated separately on each dataset's
    # test split, demonstrating that Mosaico canonicalization works per format.
    # Per-dataset threshold tuning: threshold computed on the val split of each
    # dataset and applied to the corresponding test split. Standard multi-domain
    # evaluation practice; val and test stay disjoint, no leakage.
    per_ds_metrics: Dict[str, dict] = {}
    if test_per_ds:
        log.info("=== Per-dataset test breakdown (cross-format table) ===")
        for dsid, ds_paths in test_per_ds.items():
            if not ds_paths:
                log.info("  [%s] no test sequences", dsid)
                continue

            # 1) Compute per-dataset threshold from this dataset's val split
            ds_val_paths = val_per_ds.get(dsid, []) if not is_zeroshot else []
            ds_thr = thr  # fallback to global threshold
            if ds_val_paths:
                Xk_v, Xc_v, yy_v = load_split_windows(ds_val_paths)
                if Xk_v.shape[0] > 0:
                    Xk_v = apply_kin_scaler(Xk_v, mean, std)
                    Xc_v = apply_cnn_scaler(Xc_v, cnn_mean, cnn_std)
                    if ipca is not None:
                        Xc_v = apply_ipca(Xc_v, ipca, args.ipca_components)
                    val_ds_ds = CachedFeatureDataset(Xk_v, Xc_v, yy_v)
                    v_probs, v_labels = collect_probs(model, val_ds_ds, args.batch_size)
                    ds_thr = best_threshold_f1(v_probs, v_labels)

            # 2) Load test windows for this dataset, apply same scalers
            Xk_ds, Xc_ds, yy_ds = load_split_windows(ds_paths)
            if Xk_ds.shape[0] == 0:
                log.info("  [%s] no windows (sequences too short)", dsid)
                continue
            Xk_ds = apply_kin_scaler(Xk_ds, mean, std)
            Xc_ds = apply_cnn_scaler(Xc_ds, cnn_mean, cnn_std)
            if ipca is not None:
                Xc_ds = apply_ipca(Xc_ds, ipca, args.ipca_components)
            ds_eval = CachedFeatureDataset(Xk_ds, Xc_ds, yy_ds)
            ds_probs, ds_labels = collect_probs(model, ds_eval, args.batch_size)

            # 3) Evaluate with per-dataset threshold AND with global threshold for comparison
            ds_m = evaluate(ds_probs, ds_labels, threshold=ds_thr)
            ds_m_global = evaluate(ds_probs, ds_labels, threshold=thr)
            per_ds_metrics[dsid] = {
                k: ds_m[k] for k in ["accuracy", "auc", "f1", "threshold", "confusion_matrix"]
            }
            per_ds_metrics[dsid]["f1_global_thr"] = ds_m_global["f1"]
            per_ds_metrics[dsid]["accuracy_global_thr"] = ds_m_global["accuracy"]
            log.info("  [%s] windows=%d pos=%.3f | AUC=%.4f | per-ds thr=%.3f F1=%.4f | global thr=%.3f F1=%.4f",
                     dsid, len(yy_ds), float(yy_ds.mean()) if len(yy_ds) else 0.0,
                     ds_m["auc"], ds_thr, ds_m["f1"], thr, ds_m_global["f1"])

    header = f"train={'+'.join(train_dsids)} test={'+'.join(test_dsids)}"
    save_plots(history, metrics, args.output_dir, header=header)

    results = {
        "config": {
            "mode": "zero-shot" if is_zeroshot else "in-distribution",
            "train_datasets": train_dsids,
            "test_datasets": test_dsids,
            "cache_dir": args.cache_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "dropout": args.dropout,
            "hidden": args.hidden,
            "loss": args.loss,
            "pos_weight": pos_weight,
            "no_pos_weight": args.no_pos_weight,
            "use_scheduler": args.use_scheduler,
            "weighted_sampler": args.weighted_sampler,
            "kin_noise_std": args.kin_noise_std,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "swa_start_epoch": args.swa_start_epoch,
            "ipca_components": args.ipca_components,
            "late_fusion": args.late_fusion,
            "attn_pool": args.attn_pool,
            "cnn_dim": args.cnn_dim,
            "seed": SEED,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
        },
        "sequence_counts": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "window_counts": {"train": int(X_kin_tr.shape[0]),
                          "val": int(X_kin_va.shape[0]),
                          "test": int(X_kin_te.shape[0])},
        "train_time_sec": train_time,
        "test_metrics_tuned": {k: metrics[k] for k in
                               ["accuracy", "auc", "f1", "threshold",
                                "confusion_matrix", "classification_report"]},
        "test_metrics_at_0.5": {k: metrics_at_05[k] for k in
                                ["accuracy", "auc", "f1",
                                 "confusion_matrix", "classification_report"]},
        # Cross-format breakdown: same model, evaluated per dataset separately.
        # Demonstrates that Mosaico canonicalization is format-agnostic
        # (HDF5 / h5 ROS bag / TFRecord).
        "per_dataset_test_metrics": per_ds_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output_dir)
    return 0


def split_by_pools_proportional(
    cache: Dict[str, List[str]],
    pools_json: str,
    success_ratio_train: float = 0.65,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build train/val/test from Mosaico-derived pools, sized as a fraction of
    the largest possible total per dataset.

    The train partition is balanced (ratio set by success_ratio_train); val and
    test stay at the natural per-dataset prior so test metrics reflect what the
    model will see in production. Sampling is deterministic via `seed`.

    Returns (train_p, val_p, test_p, test_per_ds, val_per_ds); index 3 is
    test_per_ds[dataset_id] -> sorted list of npz paths.
    """
    with open(pools_json) as f:
        pools = json.load(f)
    rng = np.random.default_rng(seed)

    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("train/val/test fractions must be in (0,1)")
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train+val+test must sum to 1, got {train_frac + val_frac + test_frac}",
        )

    train_p, val_p, test_p = [], [], []
    test_per_ds: Dict[str, List[str]] = {}
    val_per_ds: Dict[str, List[str]] = {}

    nv_frac = val_frac + test_frac
    fail_ratio_train = 1.0 - success_ratio_train

    for dsid, cache_paths in cache.items():
        if dsid not in pools:
            log.warning("  [proportional %s] no Mosaico pool entry; dataset dropped", dsid)
            continue
        pool_succ = set(pools[dsid].get("success", []))
        pool_fail = set(pools[dsid].get("failure", []))
        name_to_path = {os.path.basename(p).rsplit(".npz", 1)[0]: p for p in cache_paths}
        succ_paths = sorted([name_to_path[n] for n in name_to_path if n in pool_succ])
        fail_paths = sorted([name_to_path[n] for n in name_to_path if n in pool_fail])
        n_s, n_f = len(succ_paths), len(fail_paths)
        if n_s == 0 or n_f == 0:
            log.warning(
                "  [proportional %s] intersection empty (succ=%d fail=%d) -> dataset dropped",
                dsid, n_s, n_f,
            )
            continue

        prior_s = n_s / (n_s + n_f)
        succ_per_T = train_frac * success_ratio_train + nv_frac * prior_s
        fail_per_T = train_frac * fail_ratio_train + nv_frac * (1.0 - prior_s)
        T_max = int(min(n_s / max(succ_per_T, 1e-9), n_f / max(fail_per_T, 1e-9)))

        # Per-class sample counts
        n_tr_s = int(round(train_frac * success_ratio_train * T_max))
        n_tr_f = int(round(train_frac * fail_ratio_train  * T_max))
        n_va_s = int(round(val_frac   * prior_s            * T_max))
        n_va_f = int(round(val_frac   * (1.0 - prior_s)    * T_max))
        n_te_s = int(round(test_frac  * prior_s            * T_max))
        n_te_f = int(round(test_frac  * (1.0 - prior_s)    * T_max))

        # Clip to availability with a final safety cap (rounding edges).
        if n_tr_s + n_va_s + n_te_s > n_s:
            n_tr_s = max(0, n_s - n_va_s - n_te_s)
        if n_tr_f + n_va_f + n_te_f > n_f:
            n_tr_f = max(0, n_f - n_va_f - n_te_f)

        # Deterministic shuffle, then partition by cursor.
        s_idx = rng.permutation(n_s)
        f_idx = rng.permutation(n_f)
        # Layout: [val | test | train], to keep val/test small even if train is large.
        s_va = [succ_paths[s_idx[i]] for i in range(n_va_s)]
        s_te = [succ_paths[s_idx[n_va_s + i]] for i in range(n_te_s)]
        s_tr = [succ_paths[s_idx[n_va_s + n_te_s + i]] for i in range(n_tr_s)]
        f_va = [fail_paths[f_idx[i]] for i in range(n_va_f)]
        f_te = [fail_paths[f_idx[n_va_f + i]] for i in range(n_te_f)]
        f_tr = [fail_paths[f_idx[n_va_f + n_te_f + i]] for i in range(n_tr_f)]

        ds_train = sorted(s_tr + f_tr)
        ds_val   = sorted(s_va + f_va)
        ds_test  = sorted(s_te + f_te)

        train_p.extend(ds_train)
        val_p.extend(ds_val)
        test_p.extend(ds_test)
        val_per_ds[dsid] = ds_val
        test_per_ds[dsid] = ds_test

        log.info(
            "  [proportional %s] pool(s,f)=(%d,%d) prior_s=%.3f T_max=%d | "
            "train=%d (s=%d, f=%d) | val=%d (s=%d, f=%d) | test=%d (s=%d, f=%d)",
            dsid, n_s, n_f, prior_s, T_max,
            len(ds_train), n_tr_s, n_tr_f,
            len(ds_val),   n_va_s, n_va_f,
            len(ds_test),  n_te_s, n_te_f,
        )
    return train_p, val_p, test_p, test_per_ds, val_per_ds


if __name__ == "__main__":
    sys.exit(main())
