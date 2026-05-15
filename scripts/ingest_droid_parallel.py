"""Parallel ingestion of DROID into mosaicod via DROIDPlugin. Idempotent."""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path


def _setup_log(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {name} %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def worker(worker_id: int, sequence_paths: list[str], host: str, port: int,
           data_root: str) -> dict:
    """Ingest a slice of sequences. One MosaicoClient + Runner + Plugin per worker."""
    from pathlib import Path as P

    log = _setup_log(f"W{worker_id}")
    log.info("starting on %d sequences", len(sequence_paths))

    from mosaicolabs.comm import MosaicoClient
    from mosaicolabs.packs.manipulation.runner.runner import ManipulationRunner

    client = MosaicoClient.connect(host=host, port=port)
    runner = ManipulationRunner(host=host, port=port)

    # Discover sequences once at the same data_root to obtain plugin instance
    plugin, plugin_id, _all = runner._discover_sequences(P(data_root), time.monotonic())
    if hasattr(plugin, "name"):
        log.info("plugin=%s", plugin.name)

    # Refresh the existing-sequences set so this worker skips what other
    # workers have already ingested.
    from mosaicolabs.models.query import QuerySequence
    q = QuerySequence().with_user_metadata("dataset_id", eq="droid")
    existing = {it.sequence.name for it in client.query(q)}
    log.info("existing droid sequences in catalog: %d", len(existing))

    ok = skipped = failed = 0
    t0 = time.time()
    for i, sp in enumerate(sequence_paths, 1):
        sp_path = P(sp)
        # The file_executor inside ingest_sequence checks for skip-if-exists
        # internally; this loop only needs to relay the result status.
        try:
            res = runner.ingest_sequence(sp_path, plugin, client,
                                          existing_sequences=existing)
        except KeyboardInterrupt:
            log.warning("interrupted")
            break
        except Exception as e:
            log.exception("error on %s: %s", sp_path.name, e)
            failed += 1
            continue

        # Inspect result status
        status = getattr(res, "status", None)
        if status is not None:
            s_str = str(status)
            if "Skip" in s_str or "skip" in s_str:
                skipped += 1
            elif "Error" in s_str or "Fail" in s_str or "Interrupt" in s_str:
                failed += 1
            else:
                ok += 1
                # Track the new sequence locally to avoid re-trying its name.
                seq_name = getattr(res, "sequence_name", None)
                if seq_name:
                    existing.add(seq_name)
        else:
            ok += 1

        if i % 5 == 0 or i == len(sequence_paths):
            rate = i / max(time.time() - t0, 1)
            eta_sec = (len(sequence_paths) - i) / max(rate, 1e-6)
            log.info("progress %d/%d ok=%d skip=%d fail=%d (%.2f seq/s, ETA %.1fh)",
                     i, len(sequence_paths), ok, skipped, failed, rate, eta_sec / 3600)

    client.close()
    elapsed = time.time() - t0
    log.info("done. ok=%d skipped=%d failed=%d elapsed=%.1fs", ok, skipped, failed, elapsed)
    return {"worker_id": worker_id, "ok": ok, "skipped": skipped, "failed": failed,
            "elapsed": elapsed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="E:/datasets/droid/data")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6726)
    p.add_argument("--workers", type=int, default=6,
                   help="parallel worker processes (default 6, lower if mosaicod chokes)")
    p.add_argument("--max-sequences", type=int, default=1500,
                   help="cap on total sequences to ingest across all workers")
    args = p.parse_args()

    log = _setup_log("main")
    log.info("=== Parallel DROID ingestion ===")
    log.info("data_root=%s workers=%d max_seq=%d host=%s:%d",
             args.data_root, args.workers, args.max_sequences, args.host, args.port)

    # Discover sequences ONCE in the main process to avoid duplicating filesystem
    # work, then split the list across workers.
    from mosaicolabs.comm import MosaicoClient
    from mosaicolabs.packs.manipulation.runner.runner import ManipulationRunner
    from mosaicolabs.models.query import QuerySequence

    client = MosaicoClient.connect(args.host, args.port)
    runner = ManipulationRunner(host=args.host, port=args.port)
    log.info("discovering DROID sequences in %s ...", args.data_root)
    plugin, plugin_id, all_paths = runner._discover_sequences(
        Path(args.data_root), time.monotonic())
    log.info("plugin=%s discovered=%d", plugin_id, len(all_paths))

    # Apply cap at the very top
    paths = all_paths[: args.max_sequences]
    log.info("after cap: %d", len(paths))

    # Skip sequences already in the catalog. The runner re-checks internally,
    # so this filter is an optimization to avoid N round-trips.
    q = QuerySequence().with_user_metadata("dataset_id", eq="droid")
    existing = {it.sequence.name for it in client.query(q)}
    log.info("already in catalog: %d (will be skipped)", len(existing))
    client.close()

    # The plugin maps sequence_path to a sequence_name only inside
    # create_ingestion_plan; deriving the name up-front would require
    # invoking it per file. Pass all paths through and rely on the runner's
    # skip-if-exists check inside each worker.

    paths_str = [str(pp) for pp in paths]
    n_workers = max(1, args.workers)
    shards = [paths_str[i::n_workers] for i in range(n_workers)]
    log.info("sharded into %d workers (avg %d seqs each)",
             n_workers, len(paths_str) // n_workers if n_workers else 0)

    t_start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(
            worker,
            [(i, shard, args.host, args.port, args.data_root)
             for i, shard in enumerate(shards)],
        )

    elapsed = time.time() - t_start
    total_ok = sum(r["ok"] for r in results)
    total_skip = sum(r["skipped"] for r in results)
    total_fail = sum(r["failed"] for r in results)
    log.info("=" * 70)
    log.info("ALL WORKERS DONE in %.1fh", elapsed / 3600)
    log.info("totals: ok=%d skipped=%d failed=%d", total_ok, total_skip, total_fail)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
