"""List topics on a healthy sequence per dataset and flag the image-like ones."""
from __future__ import annotations

import sys

from mosaicolabs import MosaicoClient
from mosaicolabs.models.query import QuerySequence


IMAGE_KEYWORDS = ("image", "camera", "frame", "rgb", "video")


def probe_dataset(dsid: str, host: str, port: int, max_tries: int = 30) -> None:
    c = MosaicoClient.connect(host=host, port=port)
    try:
        q = QuerySequence().with_user_metadata("dataset_id", eq=dsid)
        resp = list(c.query(q))
        print(f"=== {dsid} ({len(resp)} seq in catalog) ===", flush=True)
        if not resp:
            print("  no sequences", flush=True)
            return
        for i, it in enumerate(resp[:max_tries]):
            try:
                h = c.sequence_handler(it.sequence.name)
            except Exception as e:
                print(f"  [{i}] {it.sequence.name}: handler error {type(e).__name__}: {e}", flush=True)
                continue
            if h is None:
                continue
            topics = sorted(h.topics)
            imgs = [t for t in topics if any(k in t.lower() for k in IMAGE_KEYWORDS)]
            print(f"  sample {it.sequence.name}: {len(topics)} topics total", flush=True)
            print(f"  image-like topics ({len(imgs)}):", flush=True)
            for t in imgs:
                print(f"    {t}", flush=True)
            print(f"  all topics:", flush=True)
            for t in topics:
                print(f"    {t}", flush=True)
            try:
                h.close()
            except Exception:
                pass
            return
        print(f"  sequence_handler returned None for all {max_tries} tries "
              f"(daemon in bad state? consider restarting mosaicod)", flush=True)
    finally:
        c.close()


def main() -> int:
    host, port = "127.0.0.1", 6726
    for dsid in ("reassemble", "droid", "fractal_rt1"):
        probe_dataset(dsid, host, port)
        print("", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
