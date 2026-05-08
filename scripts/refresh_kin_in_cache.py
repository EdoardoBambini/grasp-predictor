"""Refresh the `kin` and `label` arrays of each cached .npz from Mosaico,
keeping the existing `cnn` array verbatim. Idempotent."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import feature_mapper, label_adapters
from data.cross_dataset_ingestor import CrossDatasetIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("refresh_kin")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "cnn_cache_spatial"


def is_already_binarized(kin: np.ndarray) -> bool:
    """Check if gripper_state column is already in {0, 1} (within float tolerance)."""
    g = kin[:, 7]
    g = g[np.isfinite(g)]
    if g.size == 0:
        return False
    uniq = np.unique(np.round(g, 4))
    return set(uniq.tolist()).issubset({0.0, 1.0})


def npz_path(cache_dir: Path, dsid: str, seq_name: str) -> Path:
    safe = seq_name.replace("/", "_").replace("\\", "_")
    return cache_dir / dsid / f"{safe}.npz"


def refresh_one(
    ing: CrossDatasetIngestor,
    dsid: str,
    seq_name: str,
    out_path: Path,
    force: bool = False,
) -> str:
    """Refresh a single .npz. Returns one of: 'ok', 'skip', 'fail', 'mismatch', 'empty'."""
    if not out_path.exists():
        return "empty"

    try:
        existing = np.load(out_path, allow_pickle=False)
        old_kin = existing["kin"]
        cnn = existing["cnn"]
        old_label = existing["label"]
    except Exception as e:
        log.warning("[%s] %s: load fail %s", dsid, seq_name, e)
        return "fail"

    if not force and is_already_binarized(old_kin):
        return "skip"

    try:
        df = ing.process_sequence(dsid, seq_name, include_images=False)
    except Exception as e:
        log.warning("[%s] %s: process_sequence fail %s", dsid, seq_name, e)
        return "fail"

    if df is None or df.empty:
        log.warning("[%s] %s: dataframe empty", dsid, seq_name)
        return "empty"

    new_kin = df[list(feature_mapper.EXTENDED_FEATURES)].to_numpy(dtype=np.float32)
    new_label = df[label_adapters.LABEL_COL].to_numpy(dtype=np.float32)

    diff = new_kin.shape[0] - cnn.shape[0]
    if diff != 0:
        # Off-by-1/2 boundary issues happen sporadically (~5% of DROID): the
        # SyncTransformer in the new run produces 1 fewer dense rows than the
        # cached cnn, likely from chunk-edge interpolation jitter. Tolerate
        # |diff| <= 2 by trimming both arrays to the common length; reject
        # bigger mismatches as real corruption.
        if abs(diff) <= 2:
            common = min(new_kin.shape[0], cnn.shape[0])
            new_kin = new_kin[:common]
            new_label = new_label[:common]
            cnn = cnn[:common]
            log.info(
                "[%s] %s: aligned diff=%+d to common=%d (kin/cnn boundary jitter)",
                dsid, seq_name, diff, common,
            )
        else:
            log.warning(
                "[%s] %s: kin/cnn length mismatch new_kin=%d cnn=%d (was old_kin=%d) "
                "- diff %+d > tolerance, skipping",
                dsid, seq_name, new_kin.shape[0], cnn.shape[0], old_kin.shape[0], diff,
            )
            return "mismatch"

    g = new_kin[:, 7]
    g_finite = g[np.isfinite(g)]
    n_one = int((g_finite > 0.5).sum())
    n_zero = int((g_finite <= 0.5).sum())

    np.savez_compressed(
        out_path,
        kin=new_kin,
        cnn=cnn,
        label=new_label,
    )

    log.debug(
        "[%s] %s: T=%d gripper {0:%d, 1:%d}", dsid, seq_name,
        new_kin.shape[0], n_zero, n_one,
    )
    return "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="reassemble,droid,fractal_rt1",
                   help="comma-separated dsids (default: all 3 datasets)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help=f"cache root (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6726)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds between sequences (helps MinIO under load; default 0)")
    p.add_argument("--force", action="store_true",
                   help="re-refresh even if gripper looks already binarized")
    p.add_argument("--cap-per-dataset", type=int, default=None,
                   help="optional cap (debugging)")
    p.add_argument("--summary-out", default=str(REPO_ROOT / "results" / "refresh_kin_summary.json"),
                   help="where to write the per-dataset summary JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datasets: List[str] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    cache_dir = Path(args.cache_dir)

    log.info("=== refresh_kin_in_cache ===")
    log.info("datasets=%s cache_dir=%s sleep=%.2f force=%s",
             datasets, cache_dir, args.sleep, args.force)

    grand_t0 = time.time()
    summary = {}

    with CrossDatasetIngestor(host=args.host, port=args.port,
                              datasets=datasets, cap_per_dataset=None) as ing:
        for dsid in datasets:
            sub = cache_dir / dsid
            files = sorted(sub.glob("*.npz"))
            if args.cap_per_dataset is not None:
                files = files[: args.cap_per_dataset]

            log.info("[%s] %d .npz to refresh", dsid, len(files))
            counts = {"ok": 0, "skip": 0, "fail": 0, "mismatch": 0, "empty": 0}
            ds_t0 = time.time()

            for i, f in enumerate(files, 1):
                seq_name = f.stem
                t0 = time.time()
                status = refresh_one(ing, dsid, seq_name, f, force=args.force)
                counts[status] += 1
                dt = time.time() - t0

                if i <= 3 or i % 50 == 0 or i == len(files):
                    log.info("[%s] %d/%d %s -> %s (%.1fs) | ok=%d skip=%d fail=%d mismatch=%d",
                             dsid, i, len(files), seq_name, status, dt,
                             counts["ok"], counts["skip"], counts["fail"], counts["mismatch"])

                if args.sleep > 0 and status in ("ok", "fail"):
                    time.sleep(args.sleep)

            elapsed = time.time() - ds_t0
            summary[dsid] = {**counts, "n_total": len(files),
                             "elapsed_sec": round(elapsed, 1)}
            log.info("[%s] done in %.0fs %s", dsid, elapsed, counts)

    grand = time.time() - grand_t0
    summary["_grand_total"] = {"elapsed_sec": round(grand, 1),
                               "elapsed_h": round(grand / 3600, 2)}
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("=== done in %.0fs (%.2fh). Summary -> %s ===", grand, grand / 3600, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
