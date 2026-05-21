"""Regenerate the figures embedded in README.md and TECHNICAL_WRITEUP.md.

Reads the released run's results.json and writes four PNGs into docs/figures/:
  - loss_curves.png       train/validation loss per epoch (from history)
  - class_separation.png  per-dataset mean P(failure) on success vs failure bags
                          plus per-dataset AUC (the finding figure)
  - architecture.png      LateFusionLSTMMIL schematic
  - data_pipeline.png     Mosaico query -> extract -> sync -> cache -> train chain

Static diagrams (architecture, data_pipeline) do not depend on the run; the two
data-driven figures are read from the run directory's results.json.

Usage:
  python scripts/make_figures.py
  python scripts/make_figures.py --run-dir results/multimodal_indist_v10l_seed7
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _loss_curves(results: dict, out_path: str) -> None:
    hist = results["history"]
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, hist["train_loss"], marker="o", linewidth=2, label="train")
    ax.plot(epochs, hist["val_loss"], marker="o", linewidth=2, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("v10l training curves (released seed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _class_separation(results: dict, out_path: str) -> None:
    per_ds = results["per_dataset_test_metrics"]
    order = [d for d in ("droid", "fractal_rt1") if d in per_ds]
    labels = {"droid": "DROID", "fractal_rt1": "Fractal RT-1"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.36
    xs = range(len(order))
    succ = [per_ds[d]["mean_prob_failure_on_neg"] for d in order]  # P(fail) on success bags
    fail = [per_ds[d]["mean_prob_failure_on_pos"] for d in order]  # P(fail) on failure bags
    b1 = ax.bar([x - width / 2 for x in xs], succ, width,
                label="success bags", color="#2c7fb8")
    b2 = ax.bar([x + width / 2 for x in xs], fail, width,
                label="failure bags", color="#d95f0e")
    ax.bar_label(b1, fmt="%.2f", padding=2)
    ax.bar_label(b2, fmt="%.2f", padding=2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        ["%s\nAUC %.2f  |  class sep %.2f"
         % (labels[d], per_ds[d]["auc"], per_ds[d]["class_separation"]) for d in order]
    )
    ax.set_ylabel("Mean predicted P(failure)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-dataset class separation: predicted failure probability by true label")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _box(ax, x, y, w, h, text, color="#eef4fb", edge="#1f77b4"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="#555"))


def _architecture(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("LateFusionLSTMMIL (released model, 237,635 params)", fontsize=12)

    _box(ax, 0.3, 5.2, 2.4, 1.0, "Kinematics\n(B, M, T=50, 16)", "#eaf6ea", "#2ca02c")
    _box(ax, 0.3, 3.4, 2.4, 1.0, "CNN features\n(B, M, T=50, 360)", "#eaf6ea", "#2ca02c")
    _box(ax, 0.3, 1.4, 2.4, 1.0, "NL embedding (USE)\n512-D per episode", "#fdeaea", "#d62728")

    _box(ax, 3.4, 5.2, 2.4, 1.0, "kin LSTM (uni, 64)\n+ temporal attn", "#eef4fb")
    _box(ax, 3.4, 3.4, 2.4, 1.0, "vis LSTM (uni, 64)\n+ temporal attn", "#eef4fb")
    _box(ax, 3.4, 1.4, 2.4, 1.0, "Linear 512->64\n(Xavier; null for DROID)", "#fdeaea", "#d62728")

    _box(ax, 6.5, 3.7, 2.4, 1.6, "concat ->\ninstance 128 + 64\n= 192-D", "#fff7e6", "#e6a700")
    _box(ax, 9.2, 3.7, 2.4, 1.6, "Gated Attention\npool (Ilse 2018)\n-> bag 192-D", "#eef4fb")
    _box(ax, 9.2, 1.3, 2.4, 1.2, "head: Linear-ReLU\nDropout-Linear\n-> P(failure)", "#eef4fb")

    _arrow(ax, 2.7, 5.7, 3.4, 5.7)
    _arrow(ax, 2.7, 3.9, 3.4, 3.9)
    _arrow(ax, 2.7, 1.9, 3.4, 1.9)
    _arrow(ax, 5.8, 5.7, 6.7, 5.0)
    _arrow(ax, 5.8, 3.9, 6.7, 4.4)
    _arrow(ax, 5.8, 1.9, 6.7, 4.0)
    _arrow(ax, 8.9, 4.5, 9.2, 4.5)
    _arrow(ax, 10.4, 3.7, 10.4, 2.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _data_pipeline(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title("Cross-format data pipeline on Mosaico (same chain for both datasets)",
                 fontsize=12)
    steps = [
        "client.query(...)\nselect server side",
        "sequence_handler\n(name)",
        "DataFrameExtractor\nto_pandas_chunks",
        "SyncTransformer\n50 Hz, SyncHold",
        "feature_mapper\nproject + derived",
        "label_adapters\nepisode label",
        "cache .npz\nkin/cnn/label/nl_emb",
    ]
    w = 1.65; gap = 0.18; x = 0.15; y = 1.0; h = 1.1
    centers = []
    for i, s in enumerate(steps):
        color = "#fff7e6" if i == 0 else "#eef4fb"
        _box(ax, x, y, w, h, s, color)
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(steps) - 1):
        _arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/multimodal_indist_v10l_seed7",
                    help="Run directory with results.json (relative to project root).")
    ap.add_argument("--out-dir", default="docs/figures",
                    help="Output directory for the PNGs (relative to project root).")
    args = ap.parse_args()

    run_dir = os.path.join(PROJ_ROOT, args.run_dir)
    out_dir = os.path.join(PROJ_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results.json")) as f:
        results = json.load(f)

    _loss_curves(results, os.path.join(out_dir, "loss_curves.png"))
    _class_separation(results, os.path.join(out_dir, "class_separation.png"))
    _architecture(os.path.join(out_dir, "architecture.png"))
    _data_pipeline(os.path.join(out_dir, "data_pipeline.png"))
    print("Wrote loss_curves.png, class_separation.png, architecture.png, "
          "data_pipeline.png to %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
