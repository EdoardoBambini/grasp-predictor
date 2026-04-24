"""
Ingest all 149 Reassemble .h5 sequences into mosaicod.

Uses a ManipulationRunner fork with the ReassemblePlugin.
Only robot_state topics (no video/audio/events), fast ingestion.

Usage:
    python scripts/ingest_reassemble.py
    python scripts/ingest_reassemble.py --data-root "D:/datasets/reassemble/data"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest Reassemble into mosaicod")
    parser.add_argument(
        "--data-root",
        type=str,
        default="D:/datasets/reassemble/data",
        help="Path to the directory containing .h5 files",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6726)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    h5_files = sorted(data_root.glob("*.h5"))
    log.info("Found %d .h5 files in %s", len(h5_files), data_root)

    if not h5_files:
        log.error("No .h5 files found!")
        sys.exit(1)

    from mosaicolabs.comm import MosaicoClient
    from mosaicolabs.packs.manipulation.runner.runner import ManipulationRunner

    client = MosaicoClient.connect(args.host, args.port)
    log.info("Connected to mosaicod @ %s:%d", args.host, args.port)

    runner = ManipulationRunner(
        host=args.host,
        port=args.port,
    )

    t0 = time.time()
    report = runner.ingest_root(
        root=data_root,
        client=client,
        dataset_index=1,
        dataset_total=1,
    )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("INGESTION COMPLETE")
    log.info("  Discovered: %d", report.discovered)
    log.info("  Ingested:   %d", report.ingested)
    log.info("  Skipped:    %d", report.skipped)
    log.info("  Failed:     %d", report.failed)
    log.info("  Duration:   %.1f s (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
