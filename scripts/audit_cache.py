"""Per-dataset gripper distribution + label granularity audit over the .npz cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "results" / "cnn_cache_spatial"
OUT_PATH = REPO_ROOT / "results" / "audit_cache.json"

DATASETS = ("reassemble", "droid", "fractal_rt1")
GRIPPER_STATE_IDX = 7
GRIPPER_VEL_IDX = 14


def _scalar_stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    a = arr.astype(np.float64, copy=False)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"n": int(a.size), "all_nonfinite": True}
    qs = np.quantile(finite, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    bins = np.histogram(finite, bins=20)[0].tolist()
    return {
        "n": int(a.size),
        "n_finite": int(finite.size),
        "min": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "median": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "histogram_20bin": bins,
    }


def audit_dataset(dsid: str) -> dict:
    sub = CACHE_DIR / dsid
    files = sorted(sub.glob("*.npz"))
    print(f"[{dsid}] {len(files)} .npz files", flush=True)

    n_seq = len(files)
    n_seq_constant = 0
    n_seq_failure = 0
    total_frames = 0
    total_failure_frames = 0

    grip_chunks = []
    grip_vel_chunks = []
    cnn_sum = None
    cnn_sumsq = None
    cnn_count = 0

    seq_lengths = []

    for i, f in enumerate(files):
        try:
            d = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"  [{i}] {f.name}: load error {type(e).__name__}: {e}", flush=True)
            continue

        kin = d["kin"]            # (T, 15) float32
        cnn = d["cnn"]            # (T, 360) float16
        label = d["label"]        # (T,) float32

        T = kin.shape[0]
        if T == 0:
            continue

        seq_lengths.append(T)
        total_frames += T
        total_failure_frames += int((label > 0.5).sum())

        u = np.unique(label)
        if u.size <= 1:
            n_seq_constant += 1
        if float(label.max()) > 0.5:
            n_seq_failure += 1

        grip_chunks.append(kin[:, GRIPPER_STATE_IDX].astype(np.float64))
        grip_vel_chunks.append(kin[:, GRIPPER_VEL_IDX].astype(np.float64))

        cnn_f32 = cnn.astype(np.float64, copy=False)
        finite_mask = np.isfinite(cnn_f32).all(axis=1)
        cnn_finite = cnn_f32[finite_mask]
        if cnn_finite.shape[0] > 0:
            if cnn_sum is None:
                cnn_sum = cnn_finite.sum(axis=0)
                cnn_sumsq = (cnn_finite ** 2).sum(axis=0)
            else:
                cnn_sum += cnn_finite.sum(axis=0)
                cnn_sumsq += (cnn_finite ** 2).sum(axis=0)
            cnn_count += cnn_finite.shape[0]

        if (i + 1) % 200 == 0:
            print(f"  [{dsid}] processed {i + 1}/{n_seq}", flush=True)

    grip = np.concatenate(grip_chunks) if grip_chunks else np.empty(0)
    grip_vel = np.concatenate(grip_vel_chunks) if grip_vel_chunks else np.empty(0)

    cnn_mean = (cnn_sum / cnn_count) if cnn_count > 0 else None
    cnn_var = (cnn_sumsq / cnn_count - cnn_mean ** 2) if cnn_count > 0 else None
    cnn_std = np.sqrt(np.clip(cnn_var, 0.0, None)) if cnn_var is not None else None

    cnn_summary = None
    if cnn_mean is not None:
        cnn_summary = {
            "n_frames": int(cnn_count),
            "channel_mean_min": float(cnn_mean.min()),
            "channel_mean_median": float(np.median(cnn_mean)),
            "channel_mean_max": float(cnn_mean.max()),
            "channel_std_min": float(cnn_std.min()),
            "channel_std_median": float(np.median(cnn_std)),
            "channel_std_max": float(cnn_std.max()),
            "n_dead_channels_std_lt_1e-4": int((cnn_std < 1e-4).sum()),
            "n_saturated_channels_std_gt_50": int((cnn_std > 50.0).sum()),
        }

    return {
        "n_sequences": n_seq,
        "n_sequences_with_constant_label": n_seq_constant,
        "n_sequences_with_failure": n_seq_failure,
        "frame_level_failure_rate": (
            total_failure_frames / total_frames if total_frames > 0 else 0.0
        ),
        "sequence_level_failure_rate": (
            n_seq_failure / n_seq if n_seq > 0 else 0.0
        ),
        "constant_label_fraction": (
            n_seq_constant / n_seq if n_seq > 0 else 0.0
        ),
        "seq_length_stats": _scalar_stats(np.asarray(seq_lengths, dtype=np.float64)),
        "gripper_state_kin7": _scalar_stats(grip),
        "gripper_vel_kin14": _scalar_stats(grip_vel),
        "cnn_features_summary": cnn_summary,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for dsid in DATASETS:
        report[dsid] = audit_dataset(dsid)

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}", flush=True)

    print("\n=== Summary table ===", flush=True)
    fmt = "{:<14}{:>8}{:>10}{:>10}{:>14}{:>16}"
    print(fmt.format("dataset", "n_seq", "frm_fail%", "seq_fail%", "const_lbl%", "grip_median"))
    for dsid in DATASETS:
        r = report[dsid]
        gm = r["gripper_state_kin7"].get("median", float("nan"))
        print(fmt.format(
            dsid, r["n_sequences"],
            f"{100.0 * r['frame_level_failure_rate']:.1f}",
            f"{100.0 * r['sequence_level_failure_rate']:.1f}",
            f"{100.0 * r['constant_label_fraction']:.1f}",
            f"{gm:.4f}",
        ))


if __name__ == "__main__":
    main()
