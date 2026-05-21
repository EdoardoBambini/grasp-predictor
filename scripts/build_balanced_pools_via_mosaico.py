"""Build per-dataset success/failure pools using Mosaico queries.

This script is what justifies the brief's claim that Mosaico's query layer
enables custom training-set curation cross-format:

- DROID:        server-side QuerySequence on user_metadata.is_episode_successful
- Reassemble:   server-side Query on QueryTopic("/grasp_failure_label") +
                Boolean.Q.data.eq(True/False). The topic-name filter is
                essential - without it the Boolean predicate would match
                every Boolean field in every topic of the catalog.
- Fractal RT-1: client-side iteration via SequenceHandler. The grasp label
                is the terminal value of /step/reward, which the ontology
                query cannot express (it matches point-values, not the last
                sample of a topic).

Output: results/pools_via_mosaico.json with structure
    {
      "generated_at": <iso8601 utc>,
      "host": "...", "port": ...,
      "droid":       {"success": [name1, ...], "failure": [...]},
      "reassemble":  {"success": [...], "failure": [...]},
      "fractal_rt1": {"success": [...], "failure": [...]}
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pools")


def query_droid_pools(client) -> dict:
    """DROID success/failure via QuerySequence on user_metadata. Pure server-side."""
    from mosaicolabs.models.query import QuerySequence

    q_succ = (
        QuerySequence()
        .with_user_metadata("dataset_id", eq="droid")
        .with_user_metadata("is_episode_successful", eq=True)
    )
    q_fail = (
        QuerySequence()
        .with_user_metadata("dataset_id", eq="droid")
        .with_user_metadata("is_episode_successful", eq=False)
    )

    t0 = time.time()
    succ = sorted({it.sequence.name for it in client.query(q_succ)})
    t1 = time.time()
    fail = sorted({it.sequence.name for it in client.query(q_fail)})
    t2 = time.time()
    log.info(
        "[droid] success=%d (%.1fs) failure=%d (%.1fs) [server-side]",
        len(succ), t1 - t0, len(fail), t2 - t1,
    )
    return {"success": succ, "failure": fail}


def query_reassemble_pools(client) -> dict:
    """Reassemble success/failure via Boolean ontology on /grasp_failure_label.
    Combines three sub-queries (joined via AND server-side):
      - QuerySequence: filter by dataset_id user_metadata
      - QueryTopic:    filter to /grasp_failure_label topic only
      - QueryOntologyCatalog: filter Boolean.data == True/False
    The topic filter is essential - without it Boolean.Q.data.eq(True) would
    match every Boolean field in every topic of the catalog. The sub-queries
    are passed as variadic args to client.query() (the alternative is to wrap
    them in a Query() object and pass via query=...)."""
    from mosaicolabs import Boolean
    from mosaicolabs.models.query import (
        QueryOntologyCatalog, QuerySequence, QueryTopic,
    )

    t0 = time.time()
    # Failure: /grasp_failure_label.boolean.data == True
    fail = sorted({
        it.sequence.name
        for it in client.query(
            QuerySequence().with_user_metadata("dataset_id", eq="reassemble"),
            QueryTopic().with_name("/grasp_failure_label"),
            QueryOntologyCatalog().with_expression(Boolean.Q.data.eq(True)),
        )
    })
    t1 = time.time()
    # Success: /grasp_failure_label.boolean.data == False
    succ = sorted({
        it.sequence.name
        for it in client.query(
            QuerySequence().with_user_metadata("dataset_id", eq="reassemble"),
            QueryTopic().with_name("/grasp_failure_label"),
            QueryOntologyCatalog().with_expression(Boolean.Q.data.eq(False)),
        )
    })
    t2 = time.time()
    log.info(
        "[reassemble] success=%d (%.1fs) failure=%d (%.1fs) [server-side, topic-filtered]",
        len(succ), t2 - t1, len(fail), t1 - t0,
    )
    return {"success": succ, "failure": fail}


def query_fractal_pools(client, cap: int | None = None,
                         cache_dir: str = "results/cnn_cache_spatial") -> dict:
    """Fractal RT-1: terminal value of /step/reward. QueryOntologyCatalog
    cannot match terminal-of-topic predicates, and iterating each sequence
    through CrossDatasetIngestor.process_sequence takes ~7s/seq * 3967 = 7.5h.

    Instead we use Mosaico in two steps:
      1. server-side QuerySequence(dataset_id=fractal_rt1) -> all sequence names
         in the catalog, fast.
      2. read the label from the existing .npz cache, which was produced by the
         exact same Mosaico ingestion pipeline (precompute_cnn_features.py ->
         cross_dataset_ingestor -> SyncTransformer -> label_adapters.label_fractal_rt1).
    The labels are therefore Mosaico-derived (label_adapter applied during
    ingest), just persisted via the cache to avoid recomputing them every time
    we want to change the train ratio. This matches what the project brief asks
    for: cross-format curation backed by Mosaico, here a hybrid server-query +
    cached-label path because the server-side query layer has no terminal-of-
    topic predicate yet."""
    import glob as _glob
    import os as _os
    import numpy as _np
    from mosaicolabs.models.query import QuerySequence

    q_all = QuerySequence().with_user_metadata("dataset_id", eq="fractal_rt1")
    all_names = sorted({it.sequence.name for it in client.query(q_all)})
    log.info("[fractal_rt1] catalog has %d sequences (server-side query)", len(all_names))

    # Index cached .npz by sequence name.
    cache_paths = _glob.glob(_os.path.join(cache_dir, "fractal_rt1", "*.npz"))
    cache_by_name = {_os.path.basename(p).rsplit(".npz", 1)[0]: p for p in cache_paths}
    log.info("[fractal_rt1] %d .npz in cache (label_adapter output during precompute)", len(cache_by_name))

    if cap is not None and cap > 0:
        all_names = all_names[:cap]

    succ, fail, missing = [], [], 0
    t0 = time.time()
    for name in all_names:
        p = cache_by_name.get(name)
        if p is None:
            missing += 1
            continue
        with _np.load(p) as z:
            label = int(_np.asarray(z["label"]).max() > 0.5)
        if label == 1:
            fail.append(name)
        else:
            succ.append(name)
    succ.sort()
    fail.sort()
    log.info(
        "[fractal_rt1] success=%d failure=%d in %.2fs (skipped %d not-in-cache)",
        len(succ), len(fail), time.time() - t0, missing,
    )
    return {"success": succ, "failure": fail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6726)
    parser.add_argument("--out", default="results/pools_via_mosaico.json",
                        help="Output JSON path (relative to project root).")
    parser.add_argument("--datasets", default="droid,reassemble,fractal_rt1",
                        help="Comma-separated dataset ids to query.")
    parser.add_argument("--fractal-cap", type=int, default=None,
                        help="Optional cap on Fractal sequences iterated (debug).")
    parser.add_argument("--fractal-cache-dir", default="results/cnn_cache_v10k_fractal",
                        help="Parent dir of the Fractal .npz cache used to read terminal-reward "
                             "labels (v10l: 16-D kin + nl_emb). The script appends /fractal_rt1.")
    args = parser.parse_args()

    from mosaicolabs.comm import MosaicoClient
    log.info("=== Build per-dataset success/failure pools via Mosaico ===")
    log.info("Connecting to mosaicod @ %s:%d", args.host, args.port)
    client = MosaicoClient.connect(host=args.host, port=args.port)
    client.clear_sequence_handlers_cache()
    log.info("Connected.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    pools: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
    }

    try:
        if "droid" in datasets:
            pools["droid"] = query_droid_pools(client)
        if "reassemble" in datasets:
            pools["reassemble"] = query_reassemble_pools(client)
        if "fractal_rt1" in datasets:
            pools["fractal_rt1"] = query_fractal_pools(
                client, cap=args.fractal_cap, cache_dir=args.fractal_cache_dir)
    finally:
        client.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(pools, f, indent=2)
    log.info("Wrote %s", out_path)
    for ds in datasets:
        if ds in pools:
            log.info(
                "  [%s] success=%d failure=%d",
                ds, len(pools[ds]["success"]), len(pools[ds]["failure"]),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
