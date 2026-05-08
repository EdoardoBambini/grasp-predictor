"""Rebuild the mosaicod Postgres catalog from existing MinIO blobs. Used
after a catalog DB wipe when the blob store on disk is still intact."""
import argparse
import json
import logging
import uuid
from pathlib import Path

import psycopg2
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rebuild_catalog")


def extract_json_from_xlmeta(path: Path) -> dict | None:
    """Extract the JSON payload embedded in a MinIO xl.meta binary file."""
    raw = path.read_bytes()
    start = raw.find(b'{"properties')
    if start < 0:
        start = raw.find(b'{"user_metadata')
    if start < 0:
        return None
    depth = 0
    end = start
    for i in range(start, len(raw)):
        if raw[i] == ord("{"):
            depth += 1
        elif raw[i] == ord("}"):
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(raw[start:end].decode("utf-8"))


def find_parquet_part(topic_dir: Path) -> Path | None:
    """Find the part.1 parquet file inside a topic's data directory."""
    for item in topic_dir.iterdir():
        if item.is_dir() and item.name != "manifest.json":
            parquet_sub = item / "data-00000.parquet"
            if parquet_sub.exists():
                for sub in parquet_sub.iterdir():
                    part = sub / "part.1"
                    if part.exists():
                        return part
    return None


def get_data_file_name(topic_dir: Path) -> str | None:
    """Get the data:HASH/data-00000.parquet path for the chunk record."""
    for item in topic_dir.iterdir():
        if item.is_dir() and item.name != "manifest.json":
            # Convert Windows Unicode colon back to real colon for mosaicod
            dir_name = item.name.replace("\uf03a", ":")
            return f"{dir_name}/data-00000.parquet"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-root", default="D:/mosaico-store/mosaico",
                        help="MinIO store root containing per-sequence dirs")
    parser.add_argument("--dsn", default="postgresql://postgres:password@localhost:5432/mosaico",
                        help="Postgres DSN for the mosaicod catalog DB")
    args = parser.parse_args()

    store_root = Path(args.store_root)

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    # Clean existing data
    cur.execute("DELETE FROM column_chunk_numeric_t")
    cur.execute("DELETE FROM column_chunk_textual_t")
    cur.execute("DELETE FROM chunk_t")
    cur.execute("DELETE FROM column_t")
    cur.execute("DELETE FROM topic_t")
    cur.execute("DELETE FROM session_t")
    cur.execute("DELETE FROM sequence_t")
    conn.commit()
    log.info("Cleaned existing catalog")

    # Reset sequences
    cur.execute("ALTER SEQUENCE sequence_t_sequence_id_seq RESTART WITH 1")
    cur.execute("ALTER SEQUENCE session_t_session_id_seq RESTART WITH 1")
    cur.execute("ALTER SEQUENCE topic_t_topic_id_seq RESTART WITH 1")
    cur.execute("ALTER SEQUENCE chunk_t_chunk_id_seq RESTART WITH 1")
    conn.commit()

    sequence_dirs = sorted(
        [d for d in store_root.iterdir() if d.is_dir() and d.name.startswith("reassemble_")]
    )
    log.info("Found %d sequence directories", len(sequence_dirs))

    total_sequences = 0
    total_topics = 0
    total_chunks = 0

    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name

        # --- Sequence metadata from metadata.json/xl.meta ---
        meta_xlmeta = seq_dir / "metadata.json" / "xl.meta"
        seq_user_metadata = {}
        if meta_xlmeta.exists():
            meta_json = extract_json_from_xlmeta(meta_xlmeta)
            if meta_json and "user_metadata" in meta_json:
                seq_user_metadata = meta_json["user_metadata"]

        # --- Session UUID ---
        session_uuid = None
        for f in seq_dir.iterdir():
            if f.name.startswith("session-") and f.name.endswith(".json"):
                session_uuid = f.name.replace("session-", "").replace(".json", "")
                break
        if not session_uuid:
            session_uuid = str(uuid.uuid4())

        # --- Insert sequence ---
        seq_uuid = str(uuid.uuid4())
        # Use the first topic's created_at as sequence creation time
        creation_ts = 0

        robot_state_dir = seq_dir / "robot_state"
        if not robot_state_dir.exists():
            log.warning("No robot_state dir in %s, skipping", seq_name)
            continue

        # Pre-scan topics to get creation timestamp
        topic_dirs = sorted([d for d in robot_state_dir.iterdir() if d.is_dir()])
        if not topic_dirs:
            log.warning("No topics in %s, skipping", seq_name)
            continue

        first_manifest = topic_dirs[0] / "manifest.json" / "xl.meta"
        if first_manifest.exists():
            manifest_data = extract_json_from_xlmeta(first_manifest)
            if manifest_data:
                creation_ts = manifest_data["properties"]["created_at"]

        cur.execute(
            """INSERT INTO sequence_t (sequence_uuid, locator_name, user_metadata, creation_unix_tstamp)
               VALUES (%s, %s, %s, %s) RETURNING sequence_id""",
            (seq_uuid, seq_name, json.dumps(seq_user_metadata), creation_ts),
        )
        sequence_id = cur.fetchone()[0]

        # --- Insert session ---
        completion_ts = creation_ts
        cur.execute(
            """INSERT INTO session_t (session_uuid, sequence_id, creation_unix_tstamp, completion_unix_tstamp)
               VALUES (%s, %s, %s, %s) RETURNING session_id""",
            (session_uuid, sequence_id, creation_ts, completion_ts),
        )
        session_id = cur.fetchone()[0]

        # --- Insert topics ---
        for topic_dir in topic_dirs:
            topic_short = f"robot_state/{topic_dir.name}"
            topic_name = f"{seq_name}/{topic_short}"
            manifest_path = topic_dir / "manifest.json" / "xl.meta"

            ontology_tag = "unknown"
            serialization_format = "default"
            topic_user_metadata = {}
            topic_created = creation_ts
            topic_completed = creation_ts

            if manifest_path.exists():
                manifest_data = extract_json_from_xlmeta(manifest_path)
                if manifest_data:
                    props = manifest_data.get("properties", {})
                    onto = manifest_data.get("ontology_metadata", {}).get("properties", {})
                    ontology_tag = onto.get("ontology_tag", "unknown")
                    serialization_format = onto.get("serialization_format", "default")
                    topic_user_metadata = manifest_data.get("ontology_metadata", {}).get("user_metadata", {})
                    topic_created = props.get("created_at", creation_ts)
                    topic_completed = props.get("completed_at", creation_ts)

            # Read parquet metadata for stats
            part_file = find_parquet_part(topic_dir)
            data_file_name = get_data_file_name(topic_dir)
            row_count = 0
            size_bytes = 0
            start_ts = 0
            end_ts = 0

            if part_file:
                try:
                    meta = pq.read_metadata(str(part_file))
                    row_count = meta.num_rows
                    size_bytes = part_file.stat().st_size
                    rg = meta.row_group(0)
                    for i in range(rg.num_columns):
                        col = rg.column(i)
                        if "timestamp_ns" in col.path_in_schema:
                            if col.statistics and col.statistics.has_min_max:
                                start_ts = col.statistics.min
                                end_ts = col.statistics.max
                            break
                except Exception as e:
                    log.warning("Failed to read parquet metadata for %s/%s: %s", seq_name, topic_name, e)

            topic_uuid = str(uuid.uuid4())
            cur.execute(
                """INSERT INTO topic_t (topic_uuid, sequence_id, session_id, locator_name,
                   user_metadata, serialization_format, ontology_tag, creation_unix_tstamp,
                   chunks_number, total_bytes, start_index_timestamp, end_index_timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING topic_id""",
                (
                    topic_uuid, sequence_id, session_id, topic_name,
                    json.dumps(topic_user_metadata), serialization_format, ontology_tag,
                    topic_created, 1 if row_count > 0 else 0, size_bytes, start_ts, end_ts,
                ),
            )
            topic_id = cur.fetchone()[0]
            total_topics += 1

            # --- Insert chunk ---
            if data_file_name and row_count > 0:
                chunk_uuid = str(uuid.uuid4())
                cur.execute(
                    """INSERT INTO chunk_t (chunk_uuid, topic_id, data_file, size_bytes, row_count)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (chunk_uuid, topic_id, data_file_name, size_bytes, row_count),
                )
                total_chunks += 1

        total_sequences += 1
        if total_sequences % 10 == 0:
            log.info("Processed %d/%d sequences...", total_sequences, len(sequence_dirs))

    conn.commit()

    log.info("=" * 60)
    log.info("CATALOG REBUILD COMPLETE")
    log.info("  Sequences: %d", total_sequences)
    log.info("  Topics:    %d", total_topics)
    log.info("  Chunks:    %d", total_chunks)
    log.info("=" * 60)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
