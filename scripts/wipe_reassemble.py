"""Delete all reassemble_* sequences via MosaicoClient (SDK-only, no SQL).

Use before re-ingesting Reassemble with the patched plugin so that
`/grasp_failure_label` is written inside the same session as the 14 robot topics.
"""
from __future__ import annotations

import argparse
import sys
import time

from mosaicolabs import MosaicoClient
from mosaicolabs.models.platform import Sequence
from mosaicolabs.models.query import QuerySequence

HOST = "127.0.0.1"
PORT = 6726
DATASET_ID = "reassemble"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Actually delete (otherwise dry-run)")
    args = parser.parse_args()

    client = MosaicoClient.connect(host=HOST, port=PORT)
    print(f"Connected to mosaicod @ {HOST}:{PORT}")

    q = QuerySequence().with_expression(
        Sequence.Q.user_metadata["dataset_id"].eq(DATASET_ID)
    )
    resp = client.query(q)
    names = sorted([it.sequence.name for it in resp])
    print(f"Sequences with dataset_id={DATASET_ID!r}: {len(names)}")
    for n in names[:3]:
        print(f"  e.g. {n}")
    print(f"  ...")
    for n in names[-3:]:
        print(f"       {n}")

    if not args.yes:
        print()
        print("DRY RUN, re-run with --yes to actually delete.")
        client.close()
        return 0

    print()
    print(f"Deleting {len(names)} sequences...")
    t0 = time.time()
    ok = fail = 0
    for i, name in enumerate(names, 1):
        try:
            client.sequence_delete(name)
            ok += 1
            if i % 10 == 0 or i == len(names):
                print(f"  [{i}/{len(names)}] deleted")
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(names)}] FAIL {name}: {type(e).__name__}: {e}")
    elapsed = time.time() - t0
    print()
    print(f"Done: ok={ok} fail={fail} elapsed={elapsed:.1f}s")
    client.close()
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
