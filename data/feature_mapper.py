"""Canonical projection -> 8-dim base schema + 7 finite-difference derivatives, plus optional RGB topic for the multimodal path."""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_FEATURES: Tuple[str, ...] = (
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_rot_qx", "ee_rot_qy", "ee_rot_qz", "ee_rot_qw",
    "gripper_state",
)

# Finite-diff derivatives of the 8 canonical features. qw is intentionally
# skipped: it barely moves during a grasp, and keeps the total at 15.
DERIVED_FEATURES: Tuple[str, ...] = (
    "ee_vel_x", "ee_vel_y", "ee_vel_z",
    "ee_rot_vel_x", "ee_rot_vel_y", "ee_rot_vel_z",
    "gripper_vel",
)

EXTENDED_FEATURES: Tuple[str, ...] = CANONICAL_FEATURES + DERIVED_FEATURES  # 15

# Topics required per dataset to make the canonical projection.
TOPICS_PER_DATASET: Dict[str, List[str]] = {
    "reassemble": [
        "/robot_state/pose",
        "/robot_state/end_effector",
        "/grasp_failure_label",
    ],
    "droid": [
        "/observation/state/cartesian_position",
        "/observation/state/gripper_position",
    ],
    "fractal_rt1": [
        "/step/observation/base_pose_tool_reached",
        "/step/observation/gripper_closed",
        "/step/observation/gripper_closedness_commanded",
        "/step/observation/natural_language_embedding",
        "/step/reward",
        "/step/is_terminal",
    ],
}

# v10l Fractal extras. The 16th kinematic channel is the raw gripper residual
# (commanded minus sensed), populated only for Fractal RT-1; DROID has no
# commanded signal and stays 15-D, zero-padded to 16 at training load time.
# The natural-language embedding is the 512-D Universal Sentence Encoder vector
# of the episode instruction, constant across the episode, carried per-episode
# into the cache for the pre-pool language concat.
FRACTAL_GRIPPER_COMMANDED_COL = "/step/observation/gripper_closedness_commanded.floating32.data"
FRACTAL_GRIPPER_SENSED_COL = "/step/observation/gripper_closed.floating32.data"
FRACTAL_NL_EMB_COL = "/step/observation/natural_language_embedding.text_embedding.values"
GRIPPER_RESIDUAL_FEATURE = "gripper_residual"  # 16th kin channel (projected df)
NL_EMB_COLUMN = "nl_emb"                        # per-episode 512-D, carried object col
NL_EMB_DIM = 512

# Optional RGB image topic per dataset (multimodal path). None means the
# dataset has no RGB available and goes through kinematic-only. The exact
# topic names must match what the Mosaico plugin publishes at ingest time.
# Reassemble: /hand wrist camera blob.
# DROID:      3 cameras available; wrist_left picked because it tracks the
#             grasp point through the whole trajectory with minimal occlusion.
# Fractal:    single image topic from the RLDS parser.
IMAGE_TOPIC_PER_DATASET: Dict[str, Optional[str]] = {
    "reassemble":  "/hand",
    "droid":       "/observation/images/wrist_left",
    "fractal_rt1": "/step/observation/image",
}

# Target resolution for CNN input. 112 keeps per-window memory reasonable
# (50 * 3 * 112 * 112 uint8 = 1.88 MB per window) and works well with
# MobileNetV3-small.
IMAGE_TARGET_HW: Tuple[int, int] = (112, 112)

# ImageNet normalization constants (backbone is pretrained).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Per-dataset thresholds for canonicalizing gripper_state to a binary {0, 1}
# signal. Each dataset records the gripper on an incompatible scale
# (Reassemble: Franka finger position in meters, range 0..0.04; DROID:
# normalized [0,1], bimodal at 0 and 1; Fractal: already quasi binary 0/1).
# Without this binarization gripper_state and gripper_vel encode dataset
# identity, leaking through the cross-dataset bridge.
# Thresholds calibrated from a histogram analysis of the cached gripper values.
#
# Convention: 1.0 = "open" (high raw value), 0.0 = "closed" (low raw value).
# The bimodal nature of the raw signals makes the binary projection a
# faithful summary of the underlying control mode in each format.
GRIPPER_THRESHOLD_PER_DATASET: Dict[str, float] = {
    "reassemble":  0.025,  # Franka 2-finger position (meters); range 0..0.04, midpoint
    "droid":       0.5,    # normalized [0,1] bimodal at 0 and 1
    "fractal_rt1": 0.5,    # already quasi-binary; clip to {0,1}
}


def _binarize_gripper(values: np.ndarray, dsid: str) -> np.ndarray:
    """Apply the per-dataset threshold defined in GRIPPER_THRESHOLD_PER_DATASET.
    NaN values are preserved as NaN (downstream NaN-row drop in
    cross_dataset_ingestor handles them); finite values are mapped to {0.0, 1.0}.
    """
    if dsid not in GRIPPER_THRESHOLD_PER_DATASET:
        raise ValueError(
            f"No gripper threshold defined for dataset {dsid!r}. "
            f"Known: {list(GRIPPER_THRESHOLD_PER_DATASET.keys())}"
        )
    thr = GRIPPER_THRESHOLD_PER_DATASET[dsid]
    arr = np.asarray(values, dtype=np.float32)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    out[finite] = (arr[finite] > thr).astype(np.float32)
    return out

# Mosaico exposes the Pose ontology as <topic>.pose.position.{x,y,z}
# and <topic>.pose.orientation.{x,y,z,w}.
def _pose_cols(topic: str) -> Dict[str, str]:
    """Columns for a topic with Pose ontology."""
    base = f"{topic}.pose"
    return {
        "pos_x": f"{base}.position.x",
        "pos_y": f"{base}.position.y",
        "pos_z": f"{base}.position.z",
        "rot_qx": f"{base}.orientation.x",
        "rot_qy": f"{base}.orientation.y",
        "rot_qz": f"{base}.orientation.z",
        "rot_qw": f"{base}.orientation.w",
    }


def _mean_of_array_col(series: pd.Series) -> np.ndarray:
    """Mean per cell. Used to collapse a multi-finger gripper array into one scalar."""
    out = np.full(len(series), np.nan, dtype=np.float32)
    for i, v in enumerate(series.values):
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float32)
            if arr.size > 0:
                out[i] = float(arr.mean())
        except Exception:
            # Malformed cells become NaN, dropped downstream.
            continue
    return out


def _first_of_array_col(series: pd.Series) -> np.ndarray:
    """First element per cell. DROID gripper_position is a 1-elem array."""
    out = np.full(len(series), np.nan, dtype=np.float32)
    for i, v in enumerate(series.values):
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float32)
            if arr.size > 0:
                out[i] = float(arr[0])
        except Exception:
            continue
    return out


def project_reassemble(df: pd.DataFrame) -> pd.DataFrame:
    """Reassemble -> canonical. Pose from /robot_state/pose, gripper = mean
    of the 2 Franka finger positions."""
    out = pd.DataFrame({"timestamp_ns": df["timestamp_ns"].to_numpy()})
    pcols = _pose_cols("/robot_state/pose")
    for canon, src in zip(
        ("ee_pos_x", "ee_pos_y", "ee_pos_z",
         "ee_rot_qx", "ee_rot_qy", "ee_rot_qz", "ee_rot_qw"),
        ("pos_x", "pos_y", "pos_z", "rot_qx", "rot_qy", "rot_qz", "rot_qw"),
    ):
        col = pcols[src]
        # Missing column -> NaN, defensive against partial sequences.
        out[canon] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    grip_col = "/robot_state/end_effector.end_effector.positions"
    if grip_col in df.columns:
        raw = _mean_of_array_col(df[grip_col])
        out["gripper_state"] = _binarize_gripper(raw, "reassemble")
    else:
        out["gripper_state"] = np.nan
    return out


def project_droid(df: pd.DataFrame) -> pd.DataFrame:
    """DROID -> canonical."""
    out = pd.DataFrame({"timestamp_ns": df["timestamp_ns"].to_numpy()})
    pcols = _pose_cols("/observation/state/cartesian_position")
    for canon, src in zip(
        ("ee_pos_x", "ee_pos_y", "ee_pos_z",
         "ee_rot_qx", "ee_rot_qy", "ee_rot_qz", "ee_rot_qw"),
        ("pos_x", "pos_y", "pos_z", "rot_qx", "rot_qy", "rot_qz", "rot_qw"),
    ):
        col = pcols[src]
        out[canon] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    grip_col = "/observation/state/gripper_position.end_effector.positions"
    if grip_col in df.columns:
        raw = _first_of_array_col(df[grip_col])
        out["gripper_state"] = _binarize_gripper(raw, "droid")
    else:
        out["gripper_state"] = np.nan
    return out


def project_fractal_rt1(df: pd.DataFrame) -> pd.DataFrame:
    """Fractal RT1 -> canonical. Pose from base_pose_tool_reached, gripper
    from gripper_closed (float scalar)."""
    out = pd.DataFrame({"timestamp_ns": df["timestamp_ns"].to_numpy()})
    pcols = _pose_cols("/step/observation/base_pose_tool_reached")
    for canon, src in zip(
        ("ee_pos_x", "ee_pos_y", "ee_pos_z",
         "ee_rot_qx", "ee_rot_qy", "ee_rot_qz", "ee_rot_qw"),
        ("pos_x", "pos_y", "pos_z", "rot_qx", "rot_qy", "rot_qz", "rot_qw"),
    ):
        col = pcols[src]
        out[canon] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    grip_col = "/step/observation/gripper_closed.floating32.data"
    if grip_col in df.columns:
        raw = pd.to_numeric(df[grip_col], errors="coerce").to_numpy(dtype=np.float32)
        out["gripper_state"] = _binarize_gripper(raw, "fractal_rt1")
    else:
        out["gripper_state"] = np.nan
    return out


PROJECTORS = {
    "reassemble": project_reassemble,
    "droid": project_droid,
    "fractal_rt1": project_fractal_rt1,
}


def project(dsid: str, df: pd.DataFrame) -> pd.DataFrame:
    """Project a source DataFrame onto the canonical 8-feature schema."""
    if dsid not in PROJECTORS:
        raise ValueError(f"Unknown dataset id {dsid!r}. Known: {list(PROJECTORS.keys())}")
    projected = PROJECTORS[dsid](df)
    ordered = ["timestamp_ns"] + list(CANONICAL_FEATURES)
    return projected[ordered]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Finite-diff derivatives of the 8 canonical columns. Output has 15
    numeric features + timestamp.

    For DROID/Fractal the label is episode-level, so the derivatives are
    the only per-window signal that lets the LSTM tell stable grasps from
    failing ones.
    """
    out = df.copy().reset_index(drop=True)

    # SyncTransformer already gives ~constant dt at 50Hz, but compute it from
    # real timestamps anyway (cheap and catches ROS clock shifts).
    t_ns = out["timestamp_ns"].to_numpy()
    dt = np.diff(t_ns) / 1e9
    # Non-monotonic timestamps happen in ROS bags; turn into NaN, _fd handles it.
    dt[dt <= 0] = np.nan

    def _fd(series: pd.Series) -> np.ndarray:
        v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
        d = np.diff(v) / dt
        # Pad first sample with 0 (NaN would break the LSTM).
        out_arr = np.zeros_like(v, dtype=np.float32)
        out_arr[1:] = np.where(np.isnan(d), 0.0, d).astype(np.float32)
        return out_arr

    out["ee_vel_x"] = _fd(out["ee_pos_x"])
    out["ee_vel_y"] = _fd(out["ee_pos_y"])
    out["ee_vel_z"] = _fd(out["ee_pos_z"])
    # Cheap proxy for rotation rate (not true angular velocity, which would
    # require 2 * q_dot * q_conjugate). Sufficient signal for a classifier.
    out["ee_rot_vel_x"] = _fd(out["ee_rot_qx"])
    out["ee_rot_vel_y"] = _fd(out["ee_rot_qy"])
    out["ee_rot_vel_z"] = _fd(out["ee_rot_qz"])
    out["gripper_vel"] = _fd(out["gripper_state"])
    return out


# ----------------------------------------------------------------------------
# Fractal RT-1 v10l extras: 16th gripper-residual channel + NL embedding
# ----------------------------------------------------------------------------

def fractal_gripper_residual(dense: pd.DataFrame) -> Optional[np.ndarray]:
    """Raw gripper residual = commanded - sensed, per frame, from the synced
    dense DataFrame. Returns a (T,) float32 array aligned to dense rows, or
    None if either source column is absent. NaNs become 0.0."""
    if (FRACTAL_GRIPPER_COMMANDED_COL not in dense.columns
            or FRACTAL_GRIPPER_SENSED_COL not in dense.columns):
        return None
    cmd = pd.to_numeric(dense[FRACTAL_GRIPPER_COMMANDED_COL], errors="coerce").to_numpy(dtype=np.float32)
    sns = pd.to_numeric(dense[FRACTAL_GRIPPER_SENSED_COL], errors="coerce").to_numpy(dtype=np.float32)
    res = cmd - sns
    return np.nan_to_num(res, nan=0.0).astype(np.float32)


def extract_nl_emb(dense: pd.DataFrame, dim: int = NL_EMB_DIM) -> Optional[np.ndarray]:
    """The episode's Universal Sentence Encoder embedding (constant across the
    episode). Returns the first valid (dim,) float32 vector found in the
    TextEmbedding column, or None if absent/malformed."""
    if FRACTAL_NL_EMB_COL not in dense.columns:
        return None
    for v in dense[FRACTAL_NL_EMB_COL].values:
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float32).flatten()
        except Exception:
            continue
        if arr.size == dim:
            return arr
    return None


# ----------------------------------------------------------------------------
# Image extraction helpers (multimodal path)
# ----------------------------------------------------------------------------

def image_column_name(topic: str) -> str:
    """The column where DataFrameExtractor lands CompressedImage.data bytes.
    The Mosaico CompressedImage ontology projects to
    `<topic>.compressed_image.{data,format,frame_id,...}` after
    DataFrameExtractor + SyncTransformer."""
    return f"{topic}.compressed_image.data"


def decode_jpeg_to_chw(
    jpeg_bytes: Optional[bytes], target_hw: Tuple[int, int] = IMAGE_TARGET_HW,
) -> np.ndarray:
    """Decode one CompressedImage payload to a (3, H, W) uint8 tensor.
    Returns a zero frame if bytes are None/empty (placeholder that the
    model can mask downstream via a has_image flag)."""
    from PIL import Image  # lazy import, Pillow only needed for multimodal

    H, W = target_hw
    if jpeg_bytes is None or (hasattr(jpeg_bytes, "__len__") and len(jpeg_bytes) == 0):
        return np.zeros((3, H, W), dtype=np.uint8)
    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize((W, H), Image.BILINEAR)
    except Exception:
        return np.zeros((3, H, W), dtype=np.uint8)
    arr = np.asarray(img, dtype=np.uint8)           # (H, W, 3)
    return np.transpose(arr, (2, 0, 1))             # (3, H, W)


def extract_image_series(
    df: pd.DataFrame, image_topic: Optional[str],
) -> Optional[pd.Series]:
    """Return the object-dtype Series of JPEG bytes aligned to df rows, or
    None if the dataset has no image topic or the column is missing."""
    if image_topic is None:
        return None
    col = image_column_name(image_topic)
    if col not in df.columns:
        return None
    return df[col]
