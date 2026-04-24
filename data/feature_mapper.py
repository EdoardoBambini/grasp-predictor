"""
Projects dataset-specific DataFrameExtractor outputs onto a single canonical
8-dim schema (ee xyz + quaternion + gripper scalar).

These 8 are the only features shared across all 4 source formats (HDF5
Reassemble, parquet DROID, ROS bag MML, TFRecord RT1). Tactile and effort
live in one or two datasets only, so they'd break the cross-dataset bridge.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

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
    "mml": [
        "/iiwa/eePose",
        "/allegro_hand_right/joint_states",
    ],
    "fractal_rt1": [
        "/step/observation/base_pose_tool_reached",
        "/step/observation/gripper_closed",
        "/step/reward",
        "/step/is_terminal",
    ],
}

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
        out["gripper_state"] = _mean_of_array_col(df[grip_col])
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
        out["gripper_state"] = _first_of_array_col(df[grip_col])
    else:
        out["gripper_state"] = np.nan
    return out


def project_mml(df: pd.DataFrame) -> pd.DataFrame:
    """MML -> canonical. Pose from /iiwa/eePose; gripper = mean of the 16
    Allegro joints (crude, but needed to collapse to a single scalar)."""
    out = pd.DataFrame({"timestamp_ns": df["timestamp_ns"].to_numpy()})
    pcols = _pose_cols("/iiwa/eePose")
    for canon, src in zip(
        ("ee_pos_x", "ee_pos_y", "ee_pos_z",
         "ee_rot_qx", "ee_rot_qy", "ee_rot_qz", "ee_rot_qw"),
        ("pos_x", "pos_y", "pos_z", "rot_qx", "rot_qy", "rot_qz", "rot_qw"),
    ):
        col = pcols[src]
        out[canon] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    grip_col = "/allegro_hand_right/joint_states.robot_joint.positions"
    if grip_col in df.columns:
        out["gripper_state"] = _mean_of_array_col(df[grip_col])
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
        out["gripper_state"] = pd.to_numeric(df[grip_col], errors="coerce").astype(np.float32)
    else:
        out["gripper_state"] = np.nan
    return out


PROJECTORS = {
    "reassemble": project_reassemble,
    "droid": project_droid,
    "mml": project_mml,
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
    # Not real angular velocity (would need 2 * q_dot * q_conjugate), just a
    # cheap proxy for rotation rate — good enough for a classifier.
    out["ee_rot_vel_x"] = _fd(out["ee_rot_qx"])
    out["ee_rot_vel_y"] = _fd(out["ee_rot_qy"])
    out["ee_rot_vel_z"] = _fd(out["ee_rot_qz"])
    out["gripper_vel"] = _fd(out["gripper_state"])
    return out
