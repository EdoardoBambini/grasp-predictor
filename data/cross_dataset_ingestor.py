"""
Cross-dataset ingestor. For each of the 4 Mosaico-ingested datasets
(reassemble, droid, mml, fractal_rt1) pulls the sequences, syncs them at
50Hz (SyncHold), projects them onto the canonical 8-feature schema and
attaches the grasp_failure label. Only Mosaico SDK, no direct h5/parquet/bag.
"""
from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd

from mosaicolabs import MosaicoClient
from mosaicolabs.ml import DataFrameExtractor, SyncHold, SyncTransformer
from mosaicolabs.models.platform import Sequence
from mosaicolabs.models.query import QuerySequence

from . import feature_mapper, label_adapters

logger = logging.getLogger(__name__)

DEFAULT_SYNC_FPS = 50.0
DEFAULT_WINDOW_SEC = 5.0


class CrossDatasetIngestor:
    """Pulls sequences from Mosaico for all 4 datasets and yields labeled canonical DataFrames."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6726,
        sync_target_fps: float = DEFAULT_SYNC_FPS,
        window_sec: float = DEFAULT_WINDOW_SEC,
        datasets: Optional[List[str]] = None,
        cap_per_dataset: Optional[int] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._fps = sync_target_fps
        self._window_sec = window_sec
        self._datasets = datasets or list(feature_mapper.TOPICS_PER_DATASET.keys())
        self._cap = cap_per_dataset
        self._client: Optional[MosaicoClient] = None

    def connect(self) -> "CrossDatasetIngestor":
        self._client = MosaicoClient.connect(host=self._host, port=self._port)
        # Refresh the topic cache: after a recent backfill/ingest the client
        # would otherwise miss the new topics until the process restarts.
        self._client.clear_sequence_handlers_cache()
        logger.info("Connected to mosaicod @ %s:%d", self._host, self._port)
        return self

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def __enter__(self) -> "CrossDatasetIngestor":
        return self.connect()

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------

    def _query_names(self, dsid: str) -> List[str]:
        assert self._client is not None
        # Filter runs on the DB: the expression gets translated into a
        # WHERE user_metadata->>'dataset_id' = dsid. No client-side scan.
        q = QuerySequence().with_expression(Sequence.Q.user_metadata["dataset_id"].eq(dsid))
        resp = self._client.query(q)
        return sorted([it.sequence.name for it in resp])

    def process_sequence(
        self, dsid: str, seq_name: str
    ) -> Optional[pd.DataFrame]:
        """Pull + sync + project + label for a single sequence."""
        assert self._client is not None
        h = self._client.sequence_handler(seq_name)
        if h is None:
            logger.warning("Sequence %s not found", seq_name)
            return None

        try:
            # Topics can differ between sequences of the same dataset, so
            # intersect the requested list with what's actually present.
            available = set(h.topics)
            wanted = feature_mapper.TOPICS_PER_DATASET[dsid]
            topics = [t for t in wanted if t in available]
            if not topics:
                logger.warning("[%s] %s: no requested topics available; skipping", dsid, seq_name)
                return None

            ex = DataFrameExtractor(h)
            # SyncHold (forward-fill) is safe on bool/float/arrays; interp would
            # break quaternions.
            sync = SyncTransformer(target_fps=self._fps, policy=SyncHold())

            # Workaround: sync.transform() chunk-by-chunk breaks on columns with
            # ndarray cells (droid/mml joint_states). Accumulate sparse chunks,
            # concat, then a single fit_transform on the whole thing.
            sparse_parts: List[pd.DataFrame] = []
            for sparse in ex.to_pandas_chunks(topics=topics, window_sec=self._window_sec):
                if sparse is None or sparse.empty:
                    continue
                sparse_parts.append(sparse)

            if not sparse_parts:
                logger.warning("[%s] %s: empty from extractor", dsid, seq_name)
                return None

            sparse_all = pd.concat(sparse_parts, ignore_index=True)
            dense = sync.fit_transform(sparse_all)
            if dense is None or dense.empty:
                logger.warning("[%s] %s: empty after sync", dsid, seq_name)
                return None

            # Label BEFORE projection: /step/reward, /step/is_terminal and
            # /grasp_failure_label need to be still in the dense df.
            dense = label_adapters.label(dsid, dense, h)

            projected = feature_mapper.project(dsid, dense)
            projected[label_adapters.LABEL_COL] = dense[label_adapters.LABEL_COL].values

            # Drop NaN rows first (sync edges), otherwise finite-diff pollutes
            # the whole sequence with zeros.
            feat_cols = list(feature_mapper.CANONICAL_FEATURES)
            projected = projected.dropna(subset=feat_cols).reset_index(drop=True)
            if projected.empty:
                return projected

            projected = feature_mapper.add_derived_features(projected)
            return projected
        finally:
            try:
                h.close()
            except Exception:
                pass

    def iter_sequences(
        self,
    ) -> Generator[Tuple[str, str, pd.DataFrame], None, None]:
        """Yield (dataset_id, sequence_name, labeled canonical DataFrame)."""
        if self._client is None:
            raise RuntimeError("Call connect() first.")

        for dsid in self._datasets:
            names = self._query_names(dsid)
            if self._cap is not None:
                names = names[: self._cap]
            logger.info("[%s] processing %d sequences", dsid, len(names))

            ok = fail = empty = 0
            for i, name in enumerate(names, 1):
                try:
                    df = self.process_sequence(dsid, name)
                except Exception as e:
                    logger.exception("[%s] %s failed: %s", dsid, name, e)
                    fail += 1
                    continue
                if df is None or df.empty:
                    empty += 1
                    continue
                ok += 1
                if i % 10 == 0 or i == len(names):
                    logger.info(
                        "[%s] %d/%d done | ok=%d empty=%d fail=%d",
                        dsid, i, len(names), ok, empty, fail,
                    )
                yield dsid, name, df

            logger.info("[%s] total ok=%d empty=%d fail=%d", dsid, ok, empty, fail)


def summarize_sequence(dsid: str, name: str, df: pd.DataFrame) -> Dict[str, float]:
    """Small summary used by the training script for logging."""
    return {
        "dsid": dsid,
        "name": name,
        "rows": len(df),
        "pos_ratio": float(df[label_adapters.LABEL_COL].mean()) if len(df) else 0.0,
        "n_features": len(feature_mapper.EXTENDED_FEATURES),
    }
