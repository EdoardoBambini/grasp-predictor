"""
Module C — MuJoCo closed-loop simulation with Mosaico-sourced data.

Two modes:

* replay (default): arm follows the recorded Reassemble EE trajectory via IK.
  Brief-strict implementation: state init + per-tick target both come from
  Mosaico. Visually the arm replays an assembly motion with no reference to
  the lab scene objects (Mosaico does not carry object poses for Reassemble).

* demo: arm runs a scripted pick-and-place on the cube_red / sphere_blue
  scene objects. Initial EE pose still comes from the first Mosaico sample
  (Module C.2). In parallel the LSTM consumes the real Mosaico feature
  tensor in its training distribution. Used for the demonstration video.

In both modes the LSTM runs on the normalized 15-feature window built from
Mosaico (same pipeline as training) and P(fail) is compared against
/grasp_failure_label ground truth.

Usage:
    py -3.13 run_simulation.py                              # replay, auto-pick failure seq
    py -3.13 run_simulation.py --mode demo                  # scripted pick+place, LSTM on Mosaico
    py -3.13 run_simulation.py --sequence reassemble_xxx    # specific sequence
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from data import feature_mapper, label_adapters
from data.cross_dataset_ingestor import CrossDatasetIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulation")

# ── Paths ──────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "results", "crossdataset_run", "checkpoints", "best_model.pt",
)
SCALER_NPZ = os.path.join(
    os.path.dirname(__file__),
    "results", "crossdataset_run", "scaler.npz",
)
SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "simulation", "assets", "mujoco_menagerie",
    "franka_emika_panda", "grasp_lab.xml",
)
STATS_JSON = os.path.join(os.path.dirname(__file__), "outputs", "sim_replay_stats.json")

# ── Model architecture (must match crossdataset_run checkpoint) ───────
N_FEATURES = 15
SEQ_LEN = 50
HIDDEN = 256
LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True

# ── Sim ────────────────────────────────────────────────────────────────
SIM_TIMESTEP = 0.002
MOSAICO_FPS = 50.0
MOSAICO_DT = 1.0 / MOSAICO_FPS      # 0.02s
STEPS_PER_TICK = int(round(MOSAICO_DT / SIM_TIMESTEP))  # = 10
FAILURE_THRESHOLD = 0.5

# ── Gripper mapping (Reassemble -> Franka ctrl) ───────────────────────
# Reassemble gripper_state is the mean of two finger positions in meters:
# fully open ~0.04m -> ctrl 255; fully closed ~0.0m -> ctrl 0.
GRIP_OPEN_M = 0.04
GRIP_CLOSED_M = 0.0

# ── Frame mismatch shim ────────────────────────────────────────────────
# Reassemble EE poses are in the real-robot base frame, where the Franka
# sits on a workbench at ~table height. In the Menagerie scene the Franka
# base is on the floor (z=0) and the lab table top is at z=0.415. Feeding
# dataset z-values as world targets pushes the EE through the table.
# The visual IK target is shifted upward so the lowest point of the
# trajectory clears the table. The LSTM still sees the untouched feature
# tensor (training distribution preserved).
TABLE_TOP_Z = 0.415
MIN_TABLE_CLEARANCE = 0.03

# ── Demo-mode scripted waypoints (MuJoCo world coords) ────────────────
HOME_POS   = np.array([0.35, 0.00, 0.55])
CUBE_POS   = np.array([0.50, -0.12, 0.44])
SPHERE_POS = np.array([0.50,  0.05, 0.437])
DROP_POS   = np.array([0.50,  0.32, 0.418])
GRIPPER_RAMP_S = 0.4


# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

class VanillaGraspLSTM(nn.Module):
    """Mirrors the vanilla-fallback architecture saved in the
    crossdataset_run checkpoint (no tsai backbone)."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super().__init__()
        directions = 2 if bidirectional else 1
        self._lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional, batch_first=True,
        )
        self._dropout = nn.Dropout(p=dropout)
        self._head = nn.Linear(hidden_size * directions, 1)

    def forward(self, x):
        out, _ = self._lstm(x)
        return self._head(self._dropout(out[:, -1, :])).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


def load_model() -> VanillaGraspLSTM:
    model = VanillaGraspLSTM(
        input_size=N_FEATURES, hidden_size=HIDDEN, num_layers=LAYERS,
        dropout=DROPOUT, bidirectional=BIDIRECTIONAL,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("LSTM loaded -- %d parameters", sum(p.numel() for p in model.parameters()))
    return model


def load_scaler() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(SCALER_NPZ):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_NPZ}. "
            "Run `python scripts/dump_scaler.py` first (mosaicod must be up)."
        )
    npz = np.load(SCALER_NPZ, allow_pickle=True)
    mean = npz["mean"].astype(np.float32)
    std = npz["std"].astype(np.float32)
    log.info("Scaler loaded -- mean[:3]=%s std[:3]=%s", mean[:3], std[:3])
    return mean, std


# ══════════════════════════════════════════════════════════════════════
# Mosaico: pick + fetch sequence
# ══════════════════════════════════════════════════════════════════════

DSID = "reassemble"


def _query_reassemble_names(ing: CrossDatasetIngestor) -> list[str]:
    # connect() was already called; reuse the private helper that has the
    # dataset_id filter baked in.
    return ing._query_names(DSID)


def pick_sequence(ing: CrossDatasetIngestor, explicit: Optional[str]) -> Tuple[str, pd.DataFrame]:
    """Return (sequence_name, labeled_dense_df). If explicit is None, picks
    the first sequence that contains at least one failure sample."""
    if explicit:
        log.info("Fetching explicit sequence: %s", explicit)
        df = ing.process_sequence(DSID, explicit)
        if df is None or df.empty:
            raise RuntimeError(f"Sequence {explicit!r} not found or empty.")
        log.info("  n_samples=%d pos_ratio=%.3f",
                 len(df), float(df[label_adapters.LABEL_COL].mean()))
        return explicit, df

    names = _query_reassemble_names(ing)
    log.info("Scanning %d reassemble sequences for a failure case...", len(names))
    for i, name in enumerate(names, 1):
        df = ing.process_sequence(DSID, name)
        if df is None or df.empty:
            continue
        if float(df[label_adapters.LABEL_COL].max()) > 0:
            log.info("Picked sequence %s (failure=1, n_samples=%d, pos_ratio=%.3f) "
                     "after scanning %d",
                     name, len(df), float(df[label_adapters.LABEL_COL].mean()), i)
            return name, df
    raise RuntimeError("No reassemble sequence with a failure label found.")


# ══════════════════════════════════════════════════════════════════════
# IK controller (damped least-squares, 3-DoF translational)
# ══════════════════════════════════════════════════════════════════════

def ik_step(
    m: mujoco.MjModel, d: mujoco.MjData,
    site_id: int, target_pos: np.ndarray,
    step_size: float = 0.5,
    damping: float = 1e-4,
) -> np.ndarray:
    ee_pos = d.site_xpos[site_id].copy()
    error = target_pos - ee_pos

    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, site_id)
    J = jacp[:, :7]

    JJT = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(JJT, error)

    q_target = d.qpos[:7].copy() + step_size * dq
    for i in range(7):
        lo, hi = m.jnt_range[i]
        if lo < hi:
            q_target[i] = np.clip(q_target[i], lo, hi)
    return q_target


def init_arm_to_pose(m: mujoco.MjModel, d: mujoco.MjData,
                     ee_site_id: int, target_pos: np.ndarray,
                     max_iters: int = 400, tol: float = 3e-3) -> float:
    """Spin IK until the EE is within tol of target_pos. Returns final error."""
    err = float("inf")
    for _ in range(max_iters):
        q = ik_step(m, d, ee_site_id, target_pos, step_size=0.8)
        d.ctrl[:7] = q
        d.qpos[:7] = q
        mujoco.mj_forward(m, d)
        err = float(np.linalg.norm(d.site_xpos[ee_site_id] - target_pos))
        if err < tol:
            break
    return err


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def grip_meters_to_ctrl(grip_m: float) -> float:
    t = (grip_m - GRIP_CLOSED_M) / max(GRIP_OPEN_M - GRIP_CLOSED_M, 1e-6)
    t = max(0.0, min(1.0, t))
    return float(255.0 * t)


# ══════════════════════════════════════════════════════════════════════
# Demo-mode waypoints (scripted pick-and-place on scene objects)
# ══════════════════════════════════════════════════════════════════════

def build_demo_waypoints():
    """(target_xyz, gripper_ctrl, duration_sec, phase_name)"""
    return [
        (HOME_POS,                              255, 2.0, "INTRO"),
        (CUBE_POS   + np.array([0, 0, 0.16]),   255, 2.0, "P1_APPROACH"),
        (CUBE_POS   + np.array([0, 0, 0.00]),   255, 2.0, "P1_DESCEND"),
        (CUBE_POS   + np.array([0, 0, 0.00]),     0, 1.8, "P1_GRASP"),
        (CUBE_POS   + np.array([0, 0, 0.20]),     0, 2.2, "P1_LIFT"),
        (DROP_POS   + np.array([0, 0, 0.20]),     0, 3.0, "P1_TRANSPORT"),
        (DROP_POS   + np.array([0, 0, 0.05]),     0, 2.0, "P1_LOWER"),
        (DROP_POS   + np.array([0, 0, 0.05]),   255, 1.0, "P1_RELEASE"),
        (DROP_POS   + np.array([0, 0, 0.22]),   255, 1.4, "P1_RETREAT"),
        (SPHERE_POS + np.array([0, 0, 0.16]),   255, 2.0, "P2_APPROACH"),
        (SPHERE_POS + np.array([0, 0, 0.00]),   255, 2.0, "P2_DESCEND"),
        (SPHERE_POS + np.array([0, 0, 0.00]),     0, 1.8, "P2_GRASP"),
        (SPHERE_POS + np.array([0, 0, 0.20]),     0, 2.0, "P2_LIFT"),
        (DROP_POS   + np.array([0, 0, 0.20]),     0, 2.5, "P2_TRANSPORT"),
        (DROP_POS   + np.array([0, 0, 0.05]),     0, 1.5, "P2_LOWER"),
        (DROP_POS   + np.array([0, 0, 0.05]),   255, 0.8, "P2_RELEASE"),
        (HOME_POS,                              255, 3.0, "OUTRO"),
    ]


def resolve_demo_tick(t_idx: int, waypoints, start_pos: np.ndarray):
    """Return (target_pos, grip_ctrl, phase_name, prev_grip, ramp_alpha).

    Intra-phase smoothstep for smooth EE motion; gripper ramp is
    independent (fixed-time close/open, avoids tearing the grasp).
    """
    # convert tick index to cumulative time
    t_sec = t_idx * MOSAICO_DT
    cum = 0.0
    prev_pos = start_pos.copy()
    prev_grip = 255
    for i, (target, grip, dur, phase) in enumerate(waypoints):
        if t_sec < cum + dur or i == len(waypoints) - 1:
            phase_t = max(0.0, min(1.0, (t_sec - cum) / max(dur, 1e-6)))
            alpha = smoothstep(phase_t)
            interp_target = prev_pos + alpha * (target - prev_pos)
            ramp_steps = max(1, int(GRIPPER_RAMP_S / MOSAICO_DT))
            ramp_alpha = smoothstep(min(1.0, (t_sec - cum) / (ramp_steps * MOSAICO_DT)))
            grip_ctrl = prev_grip + ramp_alpha * (grip - prev_grip)
            return interp_target, float(grip_ctrl), phase
        cum += dur
        prev_pos = np.asarray(target, dtype=np.float64)
        prev_grip = grip
    # past end: park at last target
    last = waypoints[-1]
    return np.asarray(last[0], dtype=np.float64), float(last[1]), last[3]


def demo_total_duration(waypoints) -> float:
    return float(sum(w[2] for w in waypoints))


# ══════════════════════════════════════════════════════════════════════
# HUD + alarm strip
# ══════════════════════════════════════════════════════════════════════

def hud_tr(seq_name, t_idx, n_samples, lstm_p, gt, acc, latency_ms, phase=None):
    lstm_str = "  -- " if lstm_p is None else f"{lstm_p:.2f}"
    gt_str = "  -- " if gt is None else ("FAIL" if gt >= 0.5 else "ok")
    if lstm_p is None:
        status = "warming up"
    elif lstm_p < 0.3:
        status = "nominal"
    elif lstm_p < FAILURE_THRESHOLD:
        status = "caution"
    else:
        status = "FAILURE"
    if phase is None:
        title = "sequence\nprogress\nLSTM P(fail)\nGT label\nstatus\nrunning acc\nlatency ms"
        value = (f"{seq_name}\n{t_idx}/{n_samples}\n{lstm_str}\n{gt_str}\n"
                 f"{status}\n{acc:.2f}\n{latency_ms:.1f}")
    else:
        title = "sequence\nphase\nprogress\nLSTM P(fail)\nGT label\nstatus\nrunning acc\nlatency ms"
        value = (f"{seq_name}\n{phase}\n{t_idx}/{n_samples}\n{lstm_str}\n{gt_str}\n"
                 f"{status}\n{acc:.2f}\n{latency_ms:.1f}")
    return [(1, 0, title, value)]


def alarm_color(p: float, sim_time: float) -> np.ndarray:
    import math
    if p < 0.3:
        return np.array([0.20, 0.85, 0.30, 1.0], dtype=np.float32)
    if p < FAILURE_THRESHOLD:
        t = (p - 0.3) / 0.2
        return np.array([0.85 + 0.10 * t, 0.78 - 0.10 * t,
                         0.10, 1.0], dtype=np.float32)
    pulse = 0.5 + 0.5 * math.sin(sim_time * 9.0)
    return np.array([0.95, 0.10 + 0.18 * pulse, 0.08, 1.0], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["replay", "demo"], default="replay",
                        help="replay = arm follows Mosaico trajectory (brief-strict). "
                             "demo = scripted pick-and-place + LSTM on Mosaico features (video).")
    parser.add_argument("--sequence", default=None, help="Reassemble sequence name (optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6726)
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        log.error("Checkpoint not found: %s", CHECKPOINT)
        return 1
    if not os.path.exists(SCENE_XML):
        log.error("Scene not found: %s", SCENE_XML)
        return 1

    model = load_model()
    mean, std = load_scaler()

    # ── Mosaico: pull the replay sequence ──
    with CrossDatasetIngestor(host=args.host, port=args.port,
                              datasets=[DSID]) as ing:
        seq_name, df = pick_sequence(ing, args.sequence)

    feat_cols = list(feature_mapper.EXTENDED_FEATURES)
    raw = df[feat_cols].to_numpy(dtype=np.float32)
    np.nan_to_num(raw, copy=False, nan=0.0)
    # Apply training-time z-score (critical: the LSTM expects this distribution).
    normalized = ((raw - mean) / std).astype(np.float32)
    labels = df[label_adapters.LABEL_COL].to_numpy(dtype=np.float32)
    n_samples = len(df)
    if n_samples < SEQ_LEN + 10:
        log.error("Sequence too short (%d < %d) to run a window.", n_samples, SEQ_LEN + 10)
        return 1

    # EE pose trajectory for IK (use the raw, un-normalized columns).
    # NOTE: ee_xyz is shifted in z for MuJoCo visualization only — the LSTM
    # still receives the unshifted `normalized` tensor above.
    ee_xyz = raw[:, 0:3].copy()
    z_min = float(ee_xyz[:, 2].min())
    z_offset = (TABLE_TOP_Z + MIN_TABLE_CLEARANCE) - z_min
    if z_offset > 0:
        ee_xyz[:, 2] += z_offset
        log.info("Shifted IK z-targets by +%.3fm to clear table (min z %.3f -> %.3f)",
                 z_offset, z_min, z_min + z_offset)
    grip_m = raw[:, 7]    # gripper_state is column 7 in the canonical schema

    # ── MuJoCo scene ──
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_data = mujoco.MjData(mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "lab_home")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    mujoco.mj_forward(mj_model, mj_data)

    ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    alarm_geom_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("alarm_strip_front", "alarm_strip_back")
    ]

    # State re-init from Mosaico: spin IK until the EE matches the first
    # recorded EE position. This is the Module-C.2 hook (both modes).
    err0 = init_arm_to_pose(mj_model, mj_data, ee_site_id, ee_xyz[0])
    log.info("Init IK to first Mosaico pose: residual=%.4fm", err0)
    mj_data.ctrl[7] = grip_meters_to_ctrl(float(grip_m[0]))

    # ── Mode setup ──
    if args.mode == "demo":
        waypoints = build_demo_waypoints()
        demo_duration = demo_total_duration(waypoints)
        max_ticks = int(demo_duration * MOSAICO_FPS)
        # Align Mosaico feature stream so the failure zone falls inside the
        # scripted run (better visuals on the HUD during the pick-and-place).
        fail_idx = int(np.argmax(labels > 0.5)) if labels.max() > 0 else 0
        lead_ticks = int(2.0 * MOSAICO_FPS)   # show 2s of pre-failure context
        mosaico_offset = max(0, min(fail_idx - lead_ticks,
                                    max(0, n_samples - max_ticks)))
        log.info("Demo: %d scripted ticks (%.1fs), Mosaico offset=%d (fail_idx=%d)",
                 max_ticks, demo_duration, mosaico_offset, fail_idx)
    else:
        waypoints = None
        max_ticks = n_samples
        mosaico_offset = 0

    # ── Buffers ──
    buffer = collections.deque(maxlen=SEQ_LEN)
    predictions = [None] * max_ticks
    gt_tick = [None] * max_ticks
    inference_latencies_ms: list[float] = []
    n_alerts = 0
    running_acc_count = 0
    running_acc_correct = 0

    log.info("Launching viewer (mode=%s, %d ticks = %.1fs @ %.0fHz)...",
             args.mode, max_ticks, max_ticks * MOSAICO_DT, MOSAICO_FPS)
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 155
        viewer.cam.elevation = -28
        viewer.cam.distance = 1.55
        viewer.cam.lookat[:] = [0.40, 0.05, 0.42]

        step = 0
        demo_start_pos = mj_data.site_xpos[ee_site_id].copy()
        phase_name = None
        try:
            for t_idx in range(max_ticks):
                if not viewer.is_running():
                    break
                loop_t0 = time.perf_counter()

                # ── IK target + gripper ctrl: mode-dependent ──
                if args.mode == "demo":
                    target, grip_ctrl, phase_name = resolve_demo_tick(
                        t_idx, waypoints, demo_start_pos,
                    )
                    mj_data.ctrl[7] = grip_ctrl
                else:
                    target = ee_xyz[t_idx]
                    mj_data.ctrl[7] = grip_meters_to_ctrl(float(grip_m[t_idx]))

                q_arm = ik_step(mj_model, mj_data, ee_site_id, target, step_size=0.5)
                mj_data.ctrl[:7] = q_arm

                # Physics for the MOSAICO_DT window.
                for _ in range(STEPS_PER_TICK):
                    mujoco.mj_step(mj_model, mj_data)
                step += STEPS_PER_TICK

                # ── LSTM inference: always on Mosaico feature stream ──
                mosaico_idx = mosaico_offset + t_idx
                failure_prob = None
                gt_here = None
                if mosaico_idx < n_samples:
                    buffer.append(normalized[mosaico_idx])
                    gt_here = float(labels[mosaico_idx])
                    gt_tick[t_idx] = gt_here
                    if len(buffer) == SEQ_LEN:
                        window = np.stack(list(buffer))
                        x = torch.from_numpy(window).unsqueeze(0)
                        t_inf = time.perf_counter()
                        with torch.no_grad():
                            failure_prob = float(model.predict_proba(x).item())
                        inference_latencies_ms.append((time.perf_counter() - t_inf) * 1000)
                        predictions[t_idx] = failure_prob
                        if failure_prob >= FAILURE_THRESHOLD:
                            n_alerts += 1
                        pred_binary = 1.0 if failure_prob >= FAILURE_THRESHOLD else 0.0
                        running_acc_count += 1
                        if pred_binary == gt_here:
                            running_acc_correct += 1

                running_acc = (running_acc_correct / running_acc_count) if running_acc_count else 0.0

                # Alarm strip color
                p_for_color = failure_prob if failure_prob is not None else 0.0
                col = alarm_color(p_for_color, step * SIM_TIMESTEP)
                for gid in alarm_geom_ids:
                    mj_model.geom_rgba[gid] = col

                # HUD at 10Hz
                viewer.user_scn.ngeom = 0
                if t_idx % 5 == 0:
                    last_lat = inference_latencies_ms[-1] if inference_latencies_ms else 0.0
                    viewer.set_texts(hud_tr(
                        seq_name, t_idx, max_ticks, failure_prob,
                        gt_here, running_acc, last_lat,
                        phase=phase_name,
                    ))
                viewer.sync()

                # Real-time pacing
                elapsed = time.perf_counter() - loop_t0
                sleep_time = MOSAICO_DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            log.info("Stopped at tick %d/%d", t_idx, max_ticks)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    valid = [(p, gt_tick[i]) for i, p in enumerate(predictions)
             if p is not None and gt_tick[i] is not None]
    preds_arr = np.array([p for p, _ in valid], dtype=np.float64)
    labs_arr = np.array([l for _, l in valid], dtype=np.float64)
    if len(preds_arr):
        bin_pred = (preds_arr >= FAILURE_THRESHOLD).astype(np.float64)
        tp = int(((bin_pred == 1) & (labs_arr == 1)).sum())
        tn = int(((bin_pred == 0) & (labs_arr == 0)).sum())
        fp = int(((bin_pred == 1) & (labs_arr == 0)).sum())
        fn = int(((bin_pred == 0) & (labs_arr == 1)).sum())
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        p_avg = float(preds_arr.mean())
        p_min = float(preds_arr.min())
        p_max = float(preds_arr.max())
    else:
        tp = tn = fp = fn = 0
        acc = prec = rec = f1 = 0.0
        p_avg = p_min = p_max = 0.0

    print()
    print("=" * 64)
    print(f"  SIM SUMMARY ({args.mode}) -- {seq_name}")
    print("=" * 64)
    print(f"  Ticks run            : {max_ticks}  ({max_ticks * MOSAICO_DT:.2f}s @ {MOSAICO_FPS:.0f}Hz)")
    print(f"  Mosaico seq length   : {n_samples}  (offset used: {mosaico_offset})")
    print(f"  LSTM predictions     : {len(preds_arr)}  (first {SEQ_LEN-1} skipped: warmup)")
    print(f"  Failure alerts       : {n_alerts}  (threshold = {FAILURE_THRESHOLD})")
    print(f"  P(failure) avg/max/min : {p_avg:.3f} / {p_max:.3f} / {p_min:.3f}")
    if inference_latencies_ms:
        lat = np.array(inference_latencies_ms)
        print(f"  Inference latency    : mean={lat.mean():.2f}ms  "
              f"p50={np.percentile(lat,50):.2f}ms  "
              f"p95={np.percentile(lat,95):.2f}ms  "
              f"max={lat.max():.2f}ms")
    print("-" * 64)
    print("  PER-TIMESTEP CLASSIFICATION vs GROUND TRUTH")
    print(f"    accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("=" * 64)

    os.makedirs(os.path.dirname(STATS_JSON), exist_ok=True)
    out = {
        "mode": args.mode,
        "sequence": seq_name,
        "checkpoint": os.path.relpath(CHECKPOINT, os.path.dirname(__file__)),
        "n_samples_sequence": n_samples,
        "n_ticks_run": max_ticks,
        "mosaico_offset": mosaico_offset,
        "replay_fps": MOSAICO_FPS,
        "failure_threshold": FAILURE_THRESHOLD,
        "metrics": {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_predicted": len(preds_arr),
            "n_alerts": n_alerts,
            "p_avg": p_avg, "p_min": p_min, "p_max": p_max,
        },
        "latency_ms": {
            "mean": float(np.mean(inference_latencies_ms)) if inference_latencies_ms else None,
            "p50":  float(np.percentile(inference_latencies_ms, 50)) if inference_latencies_ms else None,
            "p95":  float(np.percentile(inference_latencies_ms, 95)) if inference_latencies_ms else None,
            "max":  float(np.max(inference_latencies_ms)) if inference_latencies_ms else None,
        },
        "predictions": [None if p is None else float(p) for p in predictions],
        "labels_aligned": [None if l is None else float(l) for l in gt_tick],
    }
    with open(STATS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Stats written to: {os.path.relpath(STATS_JSON, os.path.dirname(__file__))}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
