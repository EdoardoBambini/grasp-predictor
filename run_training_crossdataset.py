"""
Cross-dataset training pipeline. Pulls all 4 datasets (reassemble, droid,
mml, fractal_rt1) through Mosaico and trains the Grasp Integrity LSTM on
the canonical 15-feature tensors.

Pipeline: MosaicoClient -> QuerySequence -> DataFrameExtractor ->
SyncTransformer(50Hz, SyncHold) -> feature_mapper.project -> label_adapters
-> TensorBuilder(seq_len=50, stride=10) -> stratified split -> Trainer.

Usage:
    py run_training_crossdataset.py                       # all 4 datasets
    py run_training_crossdataset.py --cap 5 --epochs 3    # smoke test
    py run_training_crossdataset.py --datasets reassemble --cap 50 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data import feature_mapper, label_adapters
from data.cross_dataset_ingestor import CrossDatasetIngestor, summarize_sequence
from data.tensor_builder import TensorBuilder
from models.dataset import GraspDataset
from models.lstm import GraspIntegrityLSTM
from models.trainer import Trainer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("training.crossdataset")

# ---- Fixed config (CLI-overridable where useful) --------------------
SEED = 42
MOSAICOD_HOST = "127.0.0.1"
MOSAICOD_PORT = 6726
SYNC_FPS = 50.0         # SyncTransformer target frequency (Hz), brief requirement
WINDOW_SEC = 5.0        # chunk size from DataFrameExtractor.to_pandas_chunks
SEQ_LEN = 50            # 1s at 50Hz
STRIDE = 10             # 80% overlap

HIDDEN = 256
LAYERS = 2
DROPOUT = 0.3
FC_DROPOUT = 0.2
BIDIR = True

BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8            # early-stop patience on val_loss

# 70/15/15 stratified per dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+",
                   default=list(feature_mapper.TOPICS_PER_DATASET.keys()),
                   help="Subset of datasets to include.")
    p.add_argument("--cap", type=int, default=0,
                   help="Max sequences per dataset (0 = use all, the default for full cross-dataset runs).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--output", default=os.path.join(os.path.dirname(__file__),
                                                     "results", "crossdataset_run"))
    p.add_argument("--host", default=MOSAICOD_HOST)
    p.add_argument("--port", type=int, default=MOSAICOD_PORT)
    return p.parse_args()


def ingest_all(args: argparse.Namespace) -> List[Tuple[str, str, pd.DataFrame]]:
    """Pull every requested sequence, return list of (dsid, name, df)."""
    cap = args.cap if args.cap > 0 else None
    rows: List[Tuple[str, str, pd.DataFrame]] = []
    summaries: List[Dict[str, float]] = []

    t0 = time.time()
    with CrossDatasetIngestor(
        host=args.host, port=args.port,
        sync_target_fps=SYNC_FPS, window_sec=WINDOW_SEC,
        datasets=args.datasets, cap_per_dataset=cap,
    ) as ing:
        for dsid, name, df in ing.iter_sequences():
            rows.append((dsid, name, df))
            summaries.append(summarize_sequence(dsid, name, df))

    log.info("Ingestion complete: %d sequences in %.0fs", len(rows), time.time() - t0)

    if rows:
        counts: Dict[str, int] = {}
        pos_rows: Dict[str, float] = {}
        total_rows: Dict[str, int] = {}
        for s in summaries:
            ds = s["dsid"]
            counts[ds] = counts.get(ds, 0) + 1
            total_rows[ds] = total_rows.get(ds, 0) + int(s["rows"])
            pos_rows[ds] = pos_rows.get(ds, 0.0) + float(s["pos_ratio"]) * int(s["rows"])
        log.info("Per-dataset summary:")
        for ds, n in counts.items():
            tr = total_rows[ds]
            avg_pos = pos_rows[ds] / max(tr, 1)
            log.info("  %-12s sequences=%3d  rows=%8d  pos_ratio=%.3f",
                     ds, n, tr, avg_pos)
    return rows


def build_split_tensors(
    sequences: List[Tuple[str, str, pd.DataFrame]],
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    TensorBuilder,
    Tuple[int, int, int],
]:
    """Stratified per-dataset split.

    iter_sequences() emits sequences grouped by dsid. A global temporal
    slice would then put all Reassemble in train and all Fractal in test,
    leaving the test set with zero failures -> AUC undefined. Each dataset
    is sliced 70/15/15 independently and the splits are concatenated.
    """
    n = len(sequences)
    if n < 3:
        raise RuntimeError(f"Need >=3 sequences for train/val/test split, got {n}")

    # Group sequences by dataset_id.
    per_ds: Dict[str, List[Tuple[str, str, pd.DataFrame]]] = {}
    for item in sequences:
        per_ds.setdefault(item[0], []).append(item)

    train_seqs: List[Tuple[str, str, pd.DataFrame]] = []
    val_seqs: List[Tuple[str, str, pd.DataFrame]] = []
    test_seqs: List[Tuple[str, str, pd.DataFrame]] = []
    for dsid, items in per_ds.items():
        nd = len(items)
        if nd < 3:
            # Can't meaningfully split 2 or fewer sequences.
            log.warning("Dataset %s has only %d seq(s); all go to train", dsid, nd)
            train_seqs.extend(items)
            continue
        nd_train = max(1, int(nd * TRAIN_RATIO))
        nd_val = max(1, int(nd * VAL_RATIO))
        # Clamp so val and test each get >=1 sequence.
        nd_train = min(nd_train, nd - 2)
        nd_val = min(nd_val, nd - nd_train - 1)
        train_seqs.extend(items[:nd_train])
        val_seqs.extend(items[nd_train : nd_train + nd_val])
        test_seqs.extend(items[nd_train + nd_val :])
        log.info("  split[%s] train=%d val=%d test=%d (of %d)",
                 dsid, nd_train, nd_val, nd - nd_train - nd_val, nd)

    log.info("Split: train=%d val=%d test=%d sequences (stratified by dataset)",
             len(train_seqs), len(val_seqs), len(test_seqs))

    feat_cols = list(feature_mapper.EXTENDED_FEATURES)
    builder = TensorBuilder(
        sequence_length=SEQ_LEN,
        feature_columns=feat_cols,
        label_column=label_adapters.LABEL_COL,
        stride=STRIDE,
        normalize=True,
    )

    # Scaler fitted ONLY on train; val/test reuse builder._mean, builder._std
    # via _transform_chunks (no refit) — refitting would leak test statistics.
    train_dfs = [df for _, _, df in train_seqs]
    X_train, y_train = builder.build_from_chunks(train_dfs)

    X_val, y_val = _transform_chunks(builder, [df for _, _, df in val_seqs])
    X_test, y_test = _transform_chunks(builder, [df for _, _, df in test_seqs])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), builder, (
        len(train_seqs), len(val_seqs), len(test_seqs),
    )


def _transform_chunks(
    builder: TensorBuilder, chunks: List[pd.DataFrame],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build windows for val/test reusing the scaler fitted on train."""
    all_X, all_y = [], []
    for chunk in chunks:
        X, y = builder.build(chunk)
        if X.numel() == 0:
            continue
        if builder._mean is not None:
            N, T, F = X.shape
            flat = X.reshape(-1, F).numpy()
            normed = ((flat - builder._mean) / builder._std).reshape(N, T, F).astype(np.float32)
            X = torch.from_numpy(normed)
        all_X.append(X)
        all_y.append(y)
    if not all_X:
        F = len(builder.feature_columns or feature_mapper.EXTENDED_FEATURES)
        return torch.empty(0, SEQ_LEN, F), torch.empty(0)
    return torch.cat(all_X), torch.cat(all_y)


def evaluate(model: GraspIntegrityLSTM, ds: GraspDataset) -> Dict[str, object]:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve,
    )

    loader = DataLoader(ds, batch_size=BATCH)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        device = next(model.parameters()).device
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            probs = model.predict_proba(X_b)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    probs = np.asarray(all_probs, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.float64)

    acc = float((preds == labels).mean()) if len(labels) else 0.0
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        # roc_auc_score raises when all labels are the same class.
        auc = float("nan")
    f1 = float(f1_score(labels, preds, zero_division=0))
    # [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc = (fpr.tolist(), tpr.tolist())
    except ValueError:
        roc = ([], [])
    report = classification_report(labels, preds, labels=[0, 1],
                                   target_names=["success", "failure"],
                                   zero_division=0)

    return {
        "accuracy": acc, "auc": auc, "f1": f1,
        "confusion_matrix": cm,
        "roc": roc,
        "classification_report": report,
        "probs": probs.tolist(),
        "labels": labels.tolist(),
    }


def save_plots(history: Dict, metrics: Dict, out_dir: str, header: str) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    os.makedirs(out_dir, exist_ok=True)
    epochs_x = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_x, history["train_loss"], label="train", linewidth=2)
    ax.plot(epochs_x, history["val_loss"], label="val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Loss: {header}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_x, history["val_accuracy"], color="green", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Val accuracy: {header}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "val_accuracy.png"), dpi=150)
    plt.close(fig)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Success", "Failure"],
                yticklabels=["Success", "Failure"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (acc={metrics['accuracy']:.3f}, "
                 f"auc={metrics['auc']:.3f}, f1={metrics['f1']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fpr, tpr = metrics["roc"]
    if fpr and not np.isnan(metrics["auc"]):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_seed(SEED)

    log.info("=== Cross-dataset training | Brief #002 ===")
    log.info("Datasets: %s | cap=%s | epochs=%d", args.datasets, args.cap, args.epochs)

    # ---- Module A: ingest via Mosaico -------------------------------
    sequences = ingest_all(args)
    if not sequences:
        log.error("No sequences ingested, aborting.")
        return 1

    # ---- Tensor prep & split ----------------------------------------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), builder, (
        n_train_seq, n_val_seq, n_test_seq,
    ) = build_split_tensors(sequences)

    if X_train.numel() == 0 or X_val.numel() == 0 or X_test.numel() == 0:
        log.error("Empty tensor in one of the splits, increase --cap.")
        return 1

    n_features = builder.n_features or len(feature_mapper.EXTENDED_FEATURES)
    log.info("Features=%d | train=%s pos=%.3f | val=%s pos=%.3f | test=%s pos=%.3f",
             n_features,
             tuple(X_train.shape), float(y_train.mean()),
             tuple(X_val.shape), float(y_val.mean()),
             tuple(X_test.shape), float(y_test.mean()))

    train_ds = GraspDataset(X_train, y_train)
    val_ds = GraspDataset(X_val, y_val)
    test_ds = GraspDataset(X_test, y_test)

    # ---- Module B: training -----------------------------------------
    model = GraspIntegrityLSTM(
        input_size=n_features,
        hidden_size=HIDDEN, num_layers=LAYERS,
        dropout=DROPOUT, bidirectional=BIDIR, fc_dropout=FC_DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%d params)", model.__class__.__name__, total_params)

    # pos_weight=None -> trainer auto-computes n_neg/n_pos from the train set.
    trainer = Trainer(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        batch_size=BATCH, num_epochs=args.epochs,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        optimizer_name="adam", pos_weight=None,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=PATIENCE, seed=SEED,
        loss_type="bce",       # "focal" available as alternative
        use_lr_scheduler=True,
    )

    t0 = time.time()
    history = trainer.train()
    train_time = time.time() - t0
    log.info("Training done in %.0fs (%d epochs ran)", train_time, len(history["train_loss"]))

    trainer.load_best_checkpoint()

    # ---- Evaluation -------------------------------------------------
    metrics = evaluate(model, test_ds)
    log.info("Test: acc=%.4f auc=%.4f f1=%.4f",
             metrics["accuracy"], metrics["auc"], metrics["f1"])
    log.info("\n%s", metrics["classification_report"])

    # ---- Persist artifacts ------------------------------------------
    header = f"datasets={'+'.join(args.datasets)} cap={args.cap}"
    save_plots(history, metrics, args.output, header=header)

    results = {
        "config": {
            "datasets": args.datasets,
            "cap_per_dataset": args.cap,
            "epochs": args.epochs,
            "sync_fps": SYNC_FPS,
            "sequence_length": SEQ_LEN,
            "stride": STRIDE,
            "hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT,
            "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "seed": SEED,
            "features": list(feature_mapper.EXTENDED_FEATURES),
        },
        "sequence_counts": {
            "train": n_train_seq, "val": n_val_seq, "test": n_test_seq,
        },
        "window_counts": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "train_time_sec": train_time,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["classification_report"],
        },
        "history": history,
    }
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
