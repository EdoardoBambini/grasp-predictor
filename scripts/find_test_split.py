"""Reproduce the 70/15/15 stratified split (seed 42) used by
run_training_cached.py to identify the held-out test sequences."""
import glob
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_RATIO_INDIST = 0.70
VAL_RATIO_INDIST = 0.15

paths = sorted(glob.glob(os.path.join(
    ROOT, "results/cnn_cache_spatial/reassemble", "*.npz")))

labels = []
for p in paths:
    with np.load(p) as z:
        labels.append(int(np.asarray(z["label"]).max() > 0.5))
labels = np.array(labels, dtype=np.int64)

n_pos = int(labels.sum())
n_neg = int(len(labels) - n_pos)
print(f"Reassemble: {len(paths)} seq totali, {n_pos} pos, {n_neg} neg")

test_size = 1.0 - TRAIN_RATIO_INDIST  # 0.30
ds_train, rest, _, rest_y = train_test_split(
    paths, labels,
    test_size=test_size,
    stratify=labels,
    random_state=SEED,
)

val_frac_of_rest = VAL_RATIO_INDIST / test_size  # 0.5
rest_y_arr = np.asarray(rest_y)
min_rest_class = int(min((rest_y_arr == 0).sum(), (rest_y_arr == 1).sum()))
print(f"rest set: {len(rest)} seq, min class size={min_rest_class}")

if min_rest_class < 2:
    print("-> shuffled split (no stratify)")
    ds_val, ds_test = train_test_split(
        rest, train_size=val_frac_of_rest, random_state=SEED)
else:
    print("-> stratified split")
    ds_val, ds_test = train_test_split(
        rest, train_size=val_frac_of_rest,
        stratify=rest_y, random_state=SEED)

print(f"\n=== TRAIN ({len(ds_train)} seq) ===")
for p in ds_train:
    name = os.path.basename(p)
    lbl = labels[paths.index(p)]
    print(f"  [{lbl}] {name}")

print(f"\n=== VAL ({len(ds_val)} seq) ===")
for p in ds_val:
    name = os.path.basename(p)
    lbl = labels[paths.index(p)]
    print(f"  [{lbl}] {name}")

print(f"\n=== TEST ({len(ds_test)} seq) - HELD OUT ===")
for p in ds_test:
    name = os.path.basename(p)
    lbl = labels[paths.index(p)]
    print(f"  [{lbl}] {name}")
