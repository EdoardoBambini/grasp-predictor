"""
Parallel re-ingestion of the Reassemble dataset via mosaicod.

Splits the .h5 sequences across N worker processes, each with its own
MosaicoClient connection. Every worker refreshes the existing-sequence
list before starting a new file, so concurrent workers never duplicate work.

Usage:
    py -3.13 run_ingestion_parallel.py
    py -3.13 run_ingestion_parallel.py --workers 4 \\
        --dataset-root "D:/datasets/reassemble/data"
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path


def worker(worker_id: int, file_paths: list[Path], host: str, port: int) -> dict:
    """Ingest a slice of sequences."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s W{worker_id} %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(f"worker-{worker_id}")

    # Per-process SDK import: gRPC connections can't be shared across
    # multiprocessing workers (socket-side state).
    from mosaicolabs import MosaicoClient
    from mosaicolabs.packs.manipulation.datasets.reassemble.plugin_training import (
        ReassembleTrainingPlugin,
    )
    from mosaicolabs.packs.manipulation.runner.executors.file_executor import (
        FileSequenceExecutor,
    )
    from rich.console import Console

    plugin = ReassembleTrainingPlugin()
    console = Console(stderr=True)
    executor = FileSequenceExecutor(console)
    client = MosaicoClient.connect(host, port)
    log.info("Connected, assigned %d files", len(file_paths))

    ingested = 0
    skipped = 0
    failed = 0

    for i, seq_path in enumerate(file_paths, 1):
        plan = plugin.create_ingestion_plan(seq_path)

        # Refresh every iteration: other workers may have added sequences
        # in the meantime, and names must be unique per catalog.
        existing = set(client.list_sequences())
        if plan.sequence_name in existing:
            log.info("[%d/%d] SKIP %s", i, len(file_paths), plan.sequence_name)
            skipped += 1
            continue

        log.info("[%d/%d] Ingesting %s ...", i, len(file_paths), plan.sequence_name)
        t1 = time.time()
        try:
            result = executor.ingest_sequence(
                seq_path, plan, client=client, existing_sequences=existing,
            )
            elapsed = time.time() - t1
            if result:
                ingested += 1
                log.info("  OK (%.1fs)", elapsed)
            else:
                skipped += 1
                log.info("  Skipped (%.1fs)", elapsed)
        except Exception:
            failed += 1
            log.exception("  FAILED %s", plan.sequence_name)

    client.close()
    return {"ingested": ingested, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset-root", default="D:/datasets/reassemble/data",
                        help="Reassemble .h5 root directory")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6726)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s MAIN %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    from mosaicolabs.packs.manipulation.datasets.reassemble.plugin_training import (
        ReassembleTrainingPlugin,
    )
    plugin = ReassembleTrainingPlugin()
    all_paths = plugin.discover_sequences(Path(args.dataset_root))
    log.info("Discovered %d h5 files, launching %d workers", len(all_paths), args.workers)

    chunks: list[list[Path]] = [[] for _ in range(args.workers)]
    for i, p in enumerate(all_paths):
        chunks[i % args.workers].append(p)

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(
            worker,
            [(wid, chunk, args.host, args.port) for wid, chunk in enumerate(chunks)],
        )

    total_time = time.time() - t0
    total_ingested = sum(r["ingested"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    log.info(
        "=== Done: ingested=%d skipped=%d failed=%d total_time=%.0fs (%.1f min) ===",
        total_ingested, total_skipped, total_failed, total_time, total_time / 60,
    )


if __name__ == "__main__":
    main()
