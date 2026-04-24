"""
Dumps the StandardScaler used at training time to results/crossdataset_run/scaler.npz.

Re-runs the same ingestion + stratified split + fit_on_train as
run_training_crossdataset.py, then saves builder._mean and builder._std.
Needed at inference time (simulation replay) so the LSTM sees inputs in
the distribution it was trained on.

Usage: python scripts/dump_scaler.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import feature_mapper
from models.trainer import set_seed
import run_training_crossdataset as rtc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dump_scaler")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=rtc.MOSAICOD_HOST)
    parser.add_argument("--port", type=int, default=rtc.MOSAICOD_PORT)
    parser.add_argument("--cap", type=int, default=0,
                        help="cap_per_dataset (0 = all, matches the training default)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "crossdataset_run", "scaler.npz",
    ))
    args = parser.parse_args()

    # Same seed + same datasets + same ratios as training -> same split -> same stats.
    set_seed(rtc.SEED)
    ns = argparse.Namespace(
        host=args.host, port=args.port, cap=args.cap,
        datasets=list(feature_mapper.TOPICS_PER_DATASET.keys()),
    )

    sequences = rtc.ingest_all(ns)
    if not sequences:
        log.error("No sequences ingested; is mosaicod running?")
        return 1

    _, _, _, builder, _ = rtc.build_split_tensors(sequences)
    if builder._mean is None or builder._std is None:
        log.error("Builder did not fit a scaler (normalize=False?). Abort.")
        return 1

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        mean=builder._mean.astype(np.float32),
        std=builder._std.astype(np.float32),
        feature_columns=np.array(list(feature_mapper.EXTENDED_FEATURES)),
    )
    log.info("Wrote %s", out)
    log.info("  mean[:3]=%s ... std[:3]=%s",
             builder._mean[:3], builder._std[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
