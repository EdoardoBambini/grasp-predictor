"""Per-dataset adapters returning a grasp_failure Series (0=success, 1=failure)
aligned to the df timestamps."""
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
    """Label from /grasp_failure_label (built from segments_info in the h5).

    Supports two ontology schemas, checked in order:

    1) Boolean ontology. Columns /grasp_failure_label.boolean.data or
       /grasp_failure_label.data, value interpreted as 1=failure, 0=success.
       Populated by the SDK ingestion plugin.

    2) SegmentInfo ontology. Column /grasp_failure_label.segment_info.success,
       value interpreted as 1=success, 0=failure (polarity inverted vs Boolean).

    The first schema with a present column wins.
    """
    # Schema 1: Boolean ontology.
    bool_candidates = [
        "/grasp_failure_label.boolean.data",
        "/grasp_failure_label.data",
    ]
    bool_col = next((c for c in bool_candidates if c in df.columns), None)
    if bool_col is not None:
        v = df[bool_col].ffill().bfill()
        labels = (v.astype(float) > 0.5).astype(np.float32)
        labels.name = LABEL_COL
        return labels

    # Schema 2: SegmentInfo ontology. The `success` field carries the outcome
    # flag with inverted polarity (success=True means not-a-failure), so the
    # failure label is the boolean complement.
    seg_col = "/grasp_failure_label.segment_info.success"
    if seg_col in df.columns:
        v = df[seg_col].ffill().bfill()
        success = v.astype(float)
        labels = (1.0 - (success > 0.5).astype(np.float32)).astype(np.float32)
        labels.name = LABEL_COL
        return labels

    # Neither schema present: fall back to success rather than crashing, so a
    # training run can skip past incomplete sequences without losing the rest.
    logger.warning(
        "Reassemble label topic column not found in df. "
        "Expected /grasp_failure_label.boolean.data or "
        "/grasp_failure_label.segment_info.success. "
        "Falling back to success."
    )
    return _empty_label(df, value=0)


def label_droid(df: pd.DataFrame, seq_handler) -> pd.Series:
    """DROID -> episode-level label from user_metadata.is_episode_successful."""
    meta = seq_handler.user_metadata or {}
    is_ok = meta.get("is_episode_successful")
    if is_ok is None:
        logger.debug("DROID seq %s missing is_episode_successful, assuming success", seq_handler.name)
        return _empty_label(df, value=0)
    failure = 0 if bool(is_ok) else 1
    return _empty_label(df, value=failure)


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

    # Episode-level supervision: same label on every sample. The exact
    # frame at which the failure started is not annotated.
    failure = 0 if final_reward > 0 else 1
    return _empty_label(df, value=failure)


ADAPTERS: Dict[str, callable] = {
    "reassemble": label_reassemble,
    "droid": label_droid,
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
