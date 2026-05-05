"""
One-time backfill: copy `segments_info` from each Reassemble .h5 into Mosaico
as a new topic `grasp_failure_label` (Boolean, 1=failure, 0=success).

The ReassemblePlugin does not persist segments_info in sequence_metadata.
After running this script the training pipeline pulls labels via MosaicoClient
only; the .h5 side-car is never touched from training.

Only needed for catalogs populated by the unpatched plugin: the patched plugin
(see README) emits the label topic inline at ingest time.

Idempotent: skips sequences that already have `/grasp_failure_label`.

Usage:
    python scripts/backfill_reassemble_labels.py
    python scripts/backfill_reassemble_labels.py \
        --h5-root "D:/datasets/reassemble/data" --host 127.0.0.1 --port 6726
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py

from mosaicolabs import Boolean, Message, MosaicoClient
from mosaicolabs.models.platform import Sequence
from mosaicolabs.models.query import QuerySequence

TOPIC = "/grasp_failure_label"


def read_segments_from_h5(h5_path: Path) -> list[dict]:
    """Flatten segments_info/{n}/{start,end,success} + low_level/{k}/... into a single list."""
    segments: list[dict] = []
    with h5py.File(str(h5_path), "r") as f:
        grp = f.get("segments_info")
        if grp is None:
            return segments
        for key in sorted(grp.keys(), key=lambda k: int(k)):
            s = grp[key]
            if not all(k in s for k in ("start", "end", "success")):
                continue
            segments.append({
                "start_ns": int(float(s["start"][()]) * 1e9),
                "end_ns": int(float(s["end"][()]) * 1e9),
                "success": bool(s["success"][()]),
                "level": "high",
            })
            # finer-grained low_level segments, if present
            ll = s.get("low_level")
            if ll is None:
                continue
            for lk in sorted(ll.keys(), key=lambda k: int(k)):
                ls = ll[lk]
                if not all(k in ls for k in ("start", "end", "success")):
                    continue
                segments.append({
                    "start_ns": int(float(ls["start"][()]) * 1e9),
                    "end_ns": int(float(ls["end"][()]) * 1e9),
                    "success": bool(ls["success"][()]),
                    "level": "low",
                })
    return segments


def push_labels(updater, segments: list[dict]) -> int:
    """Push 2 Boolean records per segment (start + end boundary). Two
    records let SyncTransformer(SyncHold) carry the value across the whole
    segment interval."""
    if updater.topic_writer_exists(TOPIC):
        writer = updater.get_topic_writer(TOPIC)
    else:
        writer = updater.topic_create(
            topic_name=TOPIC,
            metadata={
                "source": "backfill_h5_segments_info",
                "ontology": "Boolean",
                "semantics": "grasp_failure (1=failure, 0=success)",
                "n_segments": len(segments),
            },
            ontology_type=Boolean,
        )

    pushed = 0
    for seg in segments:
        failure = not seg["success"]
        writer.push(Message(data=Boolean(data=failure), timestamp_ns=seg["start_ns"]))
        writer.push(Message(data=Boolean(data=failure), timestamp_ns=seg["end_ns"]))
        pushed += 2
    return pushed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5-root", default="D:/datasets/reassemble/data",
                        help="directory containing the original Reassemble .h5 files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6726)
    args = parser.parse_args()

    h5_root = Path(args.h5_root)

    client = MosaicoClient.connect(host=args.host, port=args.port)
    print(f"Connected to mosaicod @ {args.host}:{args.port}")

    q = QuerySequence().with_expression(Sequence.Q.user_metadata["dataset_id"].eq("reassemble"))
    resp = client.query(q)
    seq_names = sorted([it.sequence.name for it in resp])
    print(f"Reassemble sequences: {len(seq_names)}")

    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, name in enumerate(seq_names, 1):
        h = client.sequence_handler(name)
        # Idempotence: skip sequences that already carry the label topic.
        if TOPIC in list(h.topics):
            h.close()
            n_skip += 1
            if i % 20 == 0 or i == len(seq_names):
                print(f"[{i}/{len(seq_names)}] skip (already labeled): {name}")
            continue

        # source_file was saved by the Reassemble plugin at ingest time.
        source_file = (h.user_metadata or {}).get("source_file")
        if not source_file:
            print(f"[{i}] SKIP {name}: no source_file in metadata")
            h.close()
            n_fail += 1
            continue

        h5_path = h5_root / source_file
        if not h5_path.exists():
            print(f"[{i}] SKIP {name}: {h5_path} not found")
            h.close()
            n_fail += 1
            continue

        try:
            segs = read_segments_from_h5(h5_path)
            if not segs:
                print(f"[{i}] SKIP {name}: no segments_info in h5")
                h.close()
                n_fail += 1
                continue

            # h.update() = write transaction. Used to trigger the multi-session
            # bug (topic_create opened a new session instead of appending);
            # fixed upstream in the Reassemble plugin.
            with h.update() as updater:
                pushed = push_labels(updater, segs)

            print(f"[{i}/{len(seq_names)}] {name}: {len(segs)} segs, {pushed} records")
            n_ok += 1
        except Exception as e:
            print(f"[{i}] FAIL {name}: {type(e).__name__}: {e}")
            n_fail += 1
        finally:
            try:
                h.close()
            except Exception:
                pass

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.0f}s | ok={n_ok} skipped={n_skip} failed={n_fail}"
    )

    # Without this refresh, any later reconnect wouldn't see the new topics
    # until the process was restarted.
    client.clear_sequence_handlers_cache()
    client.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
