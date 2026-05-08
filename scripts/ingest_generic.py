"""Generic ingest entry point: ManipulationRunner with auto-plugin selection."""
from __future__ import annotations

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6726)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        log.error("Path does not exist: %s", data_root)
        return 1

    from mosaicolabs.comm import MosaicoClient
    from mosaicolabs.packs.manipulation.runner.runner import ManipulationRunner

    client = MosaicoClient.connect(args.host, args.port)
    log.info("Connected to mosaicod @ %s:%d", args.host, args.port)

    runner = ManipulationRunner(host=args.host, port=args.port)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
