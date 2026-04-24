"""Sanity check SDK-side: verify /grasp_failure_label topic is present
on reassemble sequences after the backfill."""
from mosaicolabs import MosaicoClient
from mosaicolabs.models.platform import Sequence
from mosaicolabs.models.query import QuerySequence


def main():
    client = MosaicoClient.connect(host="127.0.0.1", port=6726)

    q = QuerySequence().with_expression(
        Sequence.Q.user_metadata["dataset_id"].eq("reassemble")
    )
    resp = client.query(q)
    items = list(resp)
    names = sorted([it.sequence.name for it in items])
    print(f"reassemble sequences found: {len(names)}")
    if not names:
        return

    sample = names[0]
    print(f"inspecting sequence: {sample!r}")
    h = client.sequence_handler(sample)
    if h is None:
        print("handler is None, trying second sample")
        sample = names[1]
        print(f"inspecting sequence: {sample!r}")
        h = client.sequence_handler(sample)
        if h is None:
            print("STILL None, aborting")
            return
    topics = sorted(h.topics)
    print(f"topic count: {len(topics)}")
    has_label = "/grasp_failure_label" in topics
    print(f"/grasp_failure_label present on sample: {has_label}")
    if not has_label:
        print("sample topics (first 25):")
        for t in topics[:25]:
            print(f"  {t}")
    h.close()

    with_label = 0
    missing = 0
    for nm in names:
        hh = client.sequence_handler(nm)
        if hh is None:
            missing += 1
            continue
        if "/grasp_failure_label" in hh.topics:
            with_label += 1
        hh.close()
    print(f"reassemble sequences with /grasp_failure_label: {with_label}/{len(names)} "
          f"(handler None for {missing})")


if __name__ == "__main__":
    main()
