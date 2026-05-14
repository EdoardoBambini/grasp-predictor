"""Generate per-dataset diagnostic figures from results.json.

Reads `results/multimodal_indist_v9_sharp/results.json` and writes:

- `results/multimodal_indist_v9_sharp/per_dataset_auc.png`
- `results/multimodal_indist_v9_sharp/per_dataset_confusion.png`

Both are derived from `per_dataset_test_metrics` (uses per-dataset tuned threshold
for the confusion matrices, and the AUC value for the bar chart). No inference
needed; only the persisted metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "multimodal_indist_v9_sharp"
RESULTS_JSON = RESULTS_DIR / "results.json"

DATASET_ORDER = ["reassemble", "droid", "fractal_rt1"]
DATASET_LABELS = {"reassemble": "Reassemble", "droid": "DROID", "fractal_rt1": "Fractal RT-1"}


def _load() -> dict:
    with RESULTS_JSON.open() as f:
        return json.load(f)


def plot_per_dataset_auc(metrics: dict, overall_auc: float, out: Path) -> None:
    labels = ["Overall"] + [DATASET_LABELS[d] for d in DATASET_ORDER]
    aucs = [overall_auc] + [metrics[d]["auc"] for d in DATASET_ORDER]
    colors = ["#444"] + ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, aucs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="chance (0.5)")
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("AUC")
    ax.set_title("Per-dataset AUC on held-out test split")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_dataset_confusion(metrics: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, dataset in zip(axes, DATASET_ORDER):
        cm = np.array(metrics[dataset]["confusion_matrix"])
        thr = metrics[dataset]["threshold"]
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks([0, 1], ["pred succ", "pred fail"])
        ax.set_yticks([0, 1], ["true succ", "true fail"])
        ax.set_title(f"{DATASET_LABELS[dataset]}\nthr={thr:.3f}, AUC={metrics[dataset]['auc']:.3f}", fontsize=10)
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=text_color, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-dataset confusion matrices (test split, per-dataset tuned threshold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    data = _load()
    per_dataset = data["per_dataset_test_metrics"]
    overall_auc = data["test_metrics_tuned"]["auc"]

    auc_out = RESULTS_DIR / "per_dataset_auc.png"
    cm_out = RESULTS_DIR / "per_dataset_confusion.png"

    plot_per_dataset_auc(per_dataset, overall_auc, auc_out)
    plot_per_dataset_confusion(per_dataset, cm_out)

    print(f"Wrote {auc_out}")
    print(f"Wrote {cm_out}")


if __name__ == "__main__":
    main()
