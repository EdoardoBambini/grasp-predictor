"""
Per-dataset label adapters: each one returns a grasp_failure Series
(0=success, 1=failure) aligned to the df timestamps. All 4 work on data
already pulled via Mosaico (no h5/parquet/bag side-cars). Reassemble labels
come from /grasp_failure_label (populated by backfill_reassemble_labels.py).
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COL = "grasp_failure"


def _empty_label(df: pd.DataFrame, value: int = 0) -> pd.Series:
    """Create a constant-value label Series aligned to the length of df."""
    return pd.Series(value, index=df.index, name=LABEL_COL, dtype=np.float32)


def label_reassemble(df: pd.DataFrame, seq_handler) -> pd.Series:
    """Label from /grasp_failure_label (built from segments_info in the h5)."""
    # The ".boolean.data" suffix is DataFrameExtractor's naming convention
    # for Boolean-ontology topics; older backfills used ".data".
    col_candidates = [
        "/grasp_failure_label.boolean.data",
        "/grasp_failure_label.data",
    ]
    col = next((c for c in col_candidates if c in df.columns), None)
    if col is None:
        # Fall back to success rather than crashing: lets training skip past
        # incomplete sequences without losing the whole run.
        logger.warning(
            "Reassemble label topic column not found in df. "
            "Run scripts/backfill_reassemble_labels.py to populate it. Falling back to success."
        )
        return _empty_label(df, value=0)
    v = df[col]
    # ffill covers gaps SyncHold misses at sparse boundaries;
    # bfill covers samples that precede the first label record.
    v = v.ffill().bfill()
    labels = (v.astype(float) > 0.5).astype(np.float32)
    labels.name = LABEL_COL
    return labels


def label_droid(df: pd.DataFrame, seq_handler) -> pd.Series:
    """DROID -> episode-level label from user_metadata.is_episode_successful."""
    meta = seq_handler.user_metadata or {}
    is_ok = meta.get("is_episode_successful")
    if is_ok is None:
        logger.debug("DROID seq %s missing is_episode_successful, assuming success", seq_handler.name)
        return _empty_label(df, value=0)
    failure = 0 if bool(is_ok) else 1
    return _empty_label(df, value=failure)


def label_mml(df: pd.DataFrame, seq_handler) -> pd.Series:
    """MML -> teleop demos, no failure annotation. Everything is labeled
    success. Known label noise — MML only contributes domain coverage."""
    # TODO: Tekscan pressure or IIWA joint effort could infer failures.
    return _empty_label(df, value=0)


def label_fractal_rt1(df: pd.DataFrame, seq_handler) -> pd.Series:
    """Fractal RT1 -> episode-level label from the terminal /step/reward.
    RT1 keeps reward=0 until the terminal step, then 1 (success) or 0."""
    reward_cols = [c for c in df.columns if c.startswith("/step/reward")]
    is_terminal_cols = [c for c in df.columns if c.startswith("/step/is_terminal")]
    reward_col = next((c for c in reward_cols if c.endswith(".data")), None)
    term_col = next((c for c in is_terminal_cols if c.endswith(".data")), None)

    if reward_col is None:
        logger.debug("Fractal seq %s missing /step/reward, assuming success", seq_handler.name)
        return _empty_label(df, value=0)

    reward = pd.to_numeric(df[reward_col], errors="coerce").fillna(0.0)

    # Reward exactly at the last is_terminal=True is the true episode label.
    if term_col is not None:
        term = pd.to_numeric(df[term_col], errors="coerce").fillna(0.0).astype(bool)
        if term.any():
            final_reward = float(reward[term].iloc[-1])
        else:
            final_reward = float(reward.iloc[-1]) if len(reward) else 0.0
    else:
        final_reward = float(reward.iloc[-1]) if len(reward) else 0.0

    # Episode-level: same label on every sample. Can't tell WHEN the failure
    # started, only that the episode ended badly.
    failure = 0 if final_reward > 0 else 1
    return _empty_label(df, value=failure)


ADAPTERS: Dict[str, callable] = {
    "reassemble": label_reassemble,
    "droid": label_droid,
    "mml": label_mml,
    "fractal_rt1": label_fractal_rt1,
}


def label(dsid: str, df: pd.DataFrame, seq_handler) -> pd.DataFrame:
    """Add the `grasp_failure` column to df using the dataset-specific adapter."""
    if dsid not in ADAPTERS:
        raise ValueError(f"Unknown dataset {dsid!r}. Known: {list(ADAPTERS.keys())}")
    labels = ADAPTERS[dsid](df, seq_handler)
    df = df.copy()
    df[LABEL_COL] = labels.values
    return df
