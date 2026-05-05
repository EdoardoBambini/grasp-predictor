"""
Closed loop demo (multi-act version).

Runs N acts in sequence. Each act consists of a scripted Franka Panda
trajectory in MuJoCo plus parallel inference of the trained classifier on a
real Mosaico DROID sequence (held out from training and validation). SUCCESS
acts use the sphere trajectory plus anchorage (overcomes friction=0). FAILURE
acts use the cube trajectory; when the classifier triggers ABORT mid-act, the
gripper opens and the cube falls under physical gravity.

Acts are described by a JSON config: a list of dicts with `label`, `sequence`,
`offset`, `duration`. Between acts the robot smoothly returns to HOME, the
scene is reset (cube/sphere replaced), and a title card announces the next act.

Output: full-HD MP4 with overlay HUD showing real-time model state.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cached_lstm import LateFusionLSTM
from data import feature_mapper
from data.cross_dataset_ingestor import CrossDatasetIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shadow_loop")

SEQ_LEN = 50
KIN_DIM = 15
GRASP_LAB_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation", "assets", "mujoco_menagerie", "franka_emika_panda", "grasp_lab.xml",
)

# ----------------------------------------------------------------------------
# Waypoints solved via IK against grasp_lab.xml geometry
# ----------------------------------------------------------------------------
HOME = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, -0.785])

# Cube target: (0.50, -0.12, 0.44)
CUBE_APPROACH = np.array([-0.1044, -0.1368, -0.1063, -1.5479, -0.0415, 1.5147, -0.7850])
CUBE_ABOVE    = np.array([-0.1064, -0.0682, -0.1094, -1.6359, -0.0409, 1.4809, -0.7850])
CUBE_DESCEND  = np.array([-0.1079,  0.0244, -0.1113, -1.7096, -0.0402, 1.4763, -0.7850])
CUBE_LIFT     = np.array([-0.1038, -0.1493, -0.1066, -1.5226, -0.0422, 1.5252, -0.7850])

# Sphere target: (0.50, +0.05, 0.437)
SPH_APPROACH = np.array([ 0.0437, -0.1560,  0.0445, -1.5642,  0.0173, 1.4896, -0.7850])
SPH_ABOVE    = np.array([ 0.0448, -0.0874,  0.0461, -1.6539,  0.0172, 1.4553, -0.7850])
SPH_DESCEND  = np.array([ 0.0454,  0.0103,  0.0470, -1.7315,  0.0169, 1.4516, -0.7850])
SPH_LIFT     = np.array([ 0.0436, -0.1694,  0.0448, -1.5385,  0.0177, 1.4994, -0.7850])
DROP_OVER    = np.array([ 0.2594,  0.0443,  0.2761, -1.4855,  0.1124, 1.6703, -0.7850])

# Trajectories are decoupled from the act's expected outcome: any pickup target
# (cube or sphere) can be paired with any sequence (success/failure label). The
# Mosaico sequence drives the model output; the trajectory only determines what
# the Franka physically attempts.
#
# (t_start, t_end, qpos_target, gripper_target). 255=open, 50=closed-firm.
SPHERE_SCRIPT = [  # sphere reach + grasp + transport to drop zone + release
    (0.0,  2.5,  SPH_APPROACH,  255.0),
    (2.5,  4.0,  SPH_ABOVE,     255.0),
    (4.0,  5.5,  SPH_DESCEND,   255.0),
    (5.5,  6.7,  SPH_DESCEND,    50.0),
    (6.7,  8.5,  SPH_LIFT,       50.0),
    (8.5, 11.0,  DROP_OVER,      50.0),
    (11.0, 12.0, DROP_OVER,     255.0),
]
CUBE_SCRIPT = [  # cube reach + grasp + transport to drop zone + release
    (0.0,  2.5,  CUBE_APPROACH, 255.0),
    (2.5,  4.0,  CUBE_ABOVE,    255.0),
    (4.0,  5.5,  CUBE_DESCEND,  255.0),
    (5.5,  6.7,  CUBE_DESCEND,   50.0),
    (6.7,  8.5,  CUBE_LIFT,      50.0),
    (8.5, 11.0,  DROP_OVER,      50.0),
    (11.0, 12.0, DROP_OVER,     255.0),
]


def script_target(t: float, script):
    if t <= script[0][0]:
        return HOME.copy(), 255.0
    if t >= script[-1][1]:
        return script[-1][2].copy(), script[-1][3]
    for i, (ts, te, qpos, grip) in enumerate(script):
        if ts <= t < te:
            prev = script[i - 1] if i > 0 else (0.0, 0.0, HOME, 255.0)
            a = (t - ts) / max(te - ts, 1e-6)
            return (1 - a) * prev[2] + a * qpos, (1 - a) * prev[3] + a * grip
    return HOME.copy(), 255.0


# ----------------------------------------------------------------------------
# Model loader (reads attn_pool / hidden / dropout from results.json if present)
# ----------------------------------------------------------------------------
def load_model(model_dir: str):
    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    n_components = int(sc["ipca_n_components"])
    # Read config from results.json to detect attn_pool / hidden / dropout
    cfg = {}
    cfg_path = os.path.join(model_dir, "results.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f).get("config", {})
        except Exception:
            cfg = {}
    hidden = int(cfg.get("hidden", 256))
    dropout = float(cfg.get("dropout", 0.35))
    attn_pool = bool(cfg.get("attn_pool", False))
    log.info("Model config: hidden=%d dropout=%.2f attn_pool=%s ipca=%d",
             hidden, dropout, attn_pool, n_components)
    model = LateFusionLSTM(
        kin_dim=KIN_DIM, cnn_dim=n_components,
        kin_hidden=hidden // 4, vis_hidden=hidden // 4,
        num_layers=1, dropout=dropout, bidirectional=True, fc_dropout=dropout,
        attn_pool=attn_pool,
    )
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "checkpoints", "best_model.pt"),
        map_location="cpu", weights_only=True))
    model.eval()
    return (model, sc["mean"], sc["std"], sc["cnn_mean"], sc["cnn_std"],
            sc["ipca_components"], sc["ipca_mean"], n_components)




def load_droid_wrist_images(seq_names, host="127.0.0.1", port=6726):
    """Load DROID wrist-cam frames for a list of sequence names.

    Returns a dict {seq_name: [bgr_frames]}. Callers that want a flat list
    (e.g. for a concat act spanning multiple source sequences) can stitch
    them together in order on their own.
    """
    out: dict[str, list] = {}
    img_topic = feature_mapper.IMAGE_TOPIC_PER_DATASET["droid"]
    img_col = feature_mapper.image_column_name(img_topic)
    H, W = feature_mapper.IMAGE_TARGET_HW
    with CrossDatasetIngestor(host=host, port=port,
                              datasets=["droid"], cap_per_dataset=None) as ing:
        for name in seq_names:
            log.info("  loading wrist images for %s ...", name)
            df = ing.process_sequence("droid", name, include_images=True)
            if df is None or df.empty or img_col not in df.columns:
                log.warning("  no image data for %s", name)
                out[name] = []
                continue
            frames = []
            for jpeg_bytes in df[img_col].values:
                chw = feature_mapper.decode_jpeg_to_chw(jpeg_bytes, (H, W))
                rgb = chw.transpose(1, 2, 0)
                frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            out[name] = frames
            log.info("    %s -> %d frames", name, len(frames))
    log.info("  loaded wrist frames for %d sequences", len(out))
    return out


def model_proba(model, kin_w, cnn_w, kin_mean, kin_std, cnn_mean, cnn_std,
                ipca_comp, ipca_mean):
    k = (kin_w - kin_mean) / kin_std
    k = np.nan_to_num(k, nan=0.0).astype(np.float32)
    c = (cnn_w.astype(np.float32) - cnn_mean) / cnn_std
    c = np.nan_to_num(c, nan=0.0)
    c = ((c - ipca_mean) @ ipca_comp.T).astype(np.float32)
    with torch.no_grad():
        return float(model.predict_proba(
            torch.from_numpy(k[None]), torch.from_numpy(c[None])).item())


# ─────────────────────────────────────────────────────────────────────────
# Sim setup
# ─────────────────────────────────────────────────────────────────────────
def setup_franka_hd(width: int, height: int):
    mj_model = mujoco.MjModel.from_xml_path(GRASP_LAB_XML)
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height

    mj_data = mujoco.MjData(mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "lab_home")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    arm_acts = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
        for i in range(1, 8)
    ]
    gripper_act = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    ee_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    cube_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_free")
    sphere_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_blue_free")
    cube_qadr = mj_model.jnt_qposadr[cube_jid]
    sphere_qadr = mj_model.jnt_qposadr[sphere_jid]
    arm_joint_qadrs = [
        mj_model.jnt_qposadr[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
        for i in range(1, 8)
    ]

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    return (mj_model, mj_data, renderer, arm_acts, gripper_act, ee_site,
            cube_qadr, sphere_qadr, key_id, arm_joint_qadrs)


# ─────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────
def draw_hud(frame_bgr, *, sim_t, dur, prob, threshold, mode, gt_label,
             abort_t, prob_history, act_label, wrist_image=None,
             show_gt=False, reveal_text=None, reveal_color=None,
             disclaimer_lines=None, seq_name=None,
             plot_y_lo: float = 0.20, plot_y_hi: float = 0.45,
             passive: bool = False, passive_alert_thr: float = 0.7):
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    # ── Top disclaimer banner (always visible) ─────────────────────────
    banner_h = 0
    if disclaimer_lines:
        banner_h = 22 * len(disclaimer_lines) + 10
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 10, 16), -1)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
        cv2.line(out, (0, banner_h), (w, banner_h), (90, 100, 130), 1)
        for i, line in enumerate(disclaimer_lines):
            cv2.putText(out, line, (16, 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 215, 225), 1, cv2.LINE_AA)
        y0 = banner_h + 18
    else:
        y0 = 30

    cv2.putText(out, act_label, (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 230), 1, cv2.LINE_AA)
    cv2.putText(out, f"sim {sim_t:5.2f}s / {dur:.1f}s",
                (w - 180, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 230), 1, cv2.LINE_AA)
    if seq_name:
        (tw, _), _ = cv2.getTextSize(seq_name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(out, f"seq: {seq_name}", (w // 2 - (tw + 32) // 2, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 200), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 20, h - 80, 320, 22
    cv2.rectangle(out, (bar_x - 2, bar_y - 2), (bar_x + bar_w + 2, bar_y + bar_h + 2),
                  (60, 70, 90), 1)
    if not np.isnan(prob):
        fill = int(bar_w * max(0.0, min(1.0, prob)))
        col = (60, 200, 60) if prob < threshold else (60, 70, 230)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + max(fill, 2), bar_y + bar_h), col, -1)
        cv2.putText(out, f"P(failure) = {prob:.3f}",
                    (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    thr_x = bar_x + int(bar_w * threshold)
    cv2.line(out, (thr_x, bar_y - 4), (thr_x, bar_y + bar_h + 4), (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"thr {threshold:.2f}",
                (thr_x - 24, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1, cv2.LINE_AA)

    plot_x, plot_y, plot_w, plot_h = w - 320 - 20, h - 90, 320, 64
    cv2.rectangle(out, (plot_x - 2, plot_y - 2), (plot_x + plot_w + 2, plot_y + plot_h + 2),
                  (60, 70, 90), 1)
    cv2.rectangle(out, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (16, 18, 26), -1)
    # Y-range for the live P(failure) plot. The released model covers the
    # full [0.0, 1.0] range and uses these bounds via plot_y_lo / plot_y_hi.
    Y_LO, Y_HI = plot_y_lo, plot_y_hi
    def _y_for(p):
        return plot_y + int(plot_h * (1 - (max(Y_LO, min(Y_HI, p)) - Y_LO) / (Y_HI - Y_LO)))
    # Y-axis ticks
    for tick in (Y_LO, (Y_LO + Y_HI) / 2, Y_HI):
        ty = _y_for(tick)
        cv2.line(out, (plot_x, ty), (plot_x + 4, ty), (90, 100, 130), 1)
        cv2.putText(out, f"{tick:.2f}", (plot_x + 6, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 120, 140), 1, cv2.LINE_AA)
    thr_line_y = _y_for(threshold)
    cv2.line(out, (plot_x, thr_line_y), (plot_x + plot_w, thr_line_y), (255, 200, 0), 1, cv2.LINE_AA)
    if len(prob_history) >= 2:
        hist = prob_history[-plot_w:]
        x_step = plot_w / max(len(hist) - 1, 1)
        prev = None
        for i, p in enumerate(hist):
            if np.isnan(p):
                prev = None
                continue
            x = plot_x + int(i * x_step)
            y = _y_for(p)
            if prev is not None:
                col = (60, 200, 60) if p < threshold else (60, 70, 230)
                cv2.line(out, prev, (x, y), col, 2, cv2.LINE_AA)
            prev = (x, y)
    cv2.putText(out, f"P(failure) history  [{Y_LO:.2f}-{Y_HI:.2f}]",
                (plot_x + 6, plot_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 190), 1, cv2.LINE_AA)

    pill_w, pill_h = 220, 24
    pill_x = (w - pill_w) // 2
    pill_y = max(50, banner_h + 24)
    if mode == "EXECUTING":
        bg, fg = (40, 80, 40), (60, 220, 60)
        text = "EXECUTING"
    else:
        bg, fg = (60, 30, 30), (60, 60, 240)
        text = "ABORTED"
    pill_full_w = pill_w if not show_gt else pill_w + 110
    cv2.rectangle(out, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), bg, -1)
    cv2.putText(out, text, (pill_x + 14, pill_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)
    if show_gt:
        gt_text = "F" if gt_label == 1 else ("S" if gt_label == 0 else "-")
        gt_col = (60, 70, 230) if gt_label == 1 else (60, 200, 60) if gt_label == 0 else (130, 130, 130)
        cv2.putText(out, f"  Mosaico GT: {gt_text}",
                    (pill_x + 100, pill_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, gt_col, 1, cv2.LINE_AA)

    # Passive-mode small badge: highlights that the LSTM has detected a
    # failure while the robot is not allowed to react. Positioned BELOW the
    # title bar / seconds counter / pill so it does not overlap the timer
    # ("non copre i secondi"). Latching: once the model has crossed the
    # alert threshold even briefly, the badge stays in 'detected' state for
    # the rest of the act so a viewer following P(failure) doesn't miss it.
    if passive and not np.isnan(prob):
        # Latch on the running max of the trajectory -> once it crosses the
        # alert threshold even for a single frame, the badge stays red.
        finite_hist = [p for p in prob_history if not np.isnan(p)]
        peak_so_far = max(finite_hist) if finite_hist else float(prob)
        triggered = peak_so_far >= passive_alert_thr
        badge_text = ("LSTM detected failure - robot unaware"
                      if triggered else "LSTM observing - no abort")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (tw, th_), _ = cv2.getTextSize(badge_text, font, scale, thick)
        pad_x, pad_y = 10, 6
        bx1 = w - 16
        # Place badge clearly below the seconds counter / pill / disclaimer
        # banner. Rendering order: banner -> seconds (y0) -> pill (pill_y+24)
        # -> badge. The offset banner_h + 60 fits 0 to 3 disclaimer lines.
        by0 = banner_h + 60
        bx0 = bx1 - tw - 2 * pad_x
        by1 = by0 + th_ + 2 * pad_y
        if triggered:
            pulse = 0.55 + 0.45 * abs(np.sin(sim_t * 4.0))
            border_col = (40, 40, int(120 + 130 * pulse))   # red, pulsing
            text_col = (60, 70, int(120 + 130 * pulse))
        else:
            border_col = (90, 100, 130)
            text_col = (180, 190, 210)
        overlay = out.copy()
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (12, 14, 22), -1)
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
        cv2.rectangle(out, (bx0, by0), (bx1, by1), border_col, 1)
        cv2.putText(out, badge_text, (bx0 + pad_x, by1 - pad_y),
                    font, scale, text_col, thick, cv2.LINE_AA)

    if abort_t >= 0 and mode == "ABORTED" and reveal_text is None:
        # Big red ABORT card. Pulses for the first 2 seconds after trigger,
        # then steady. Sized and positioned to remain visually distinct from
        # the gripper opening itself (which looks identical to a delivery
        # drop frame-by-frame).
        elapsed_since_abort = max(0.0, sim_t - abort_t)
        if elapsed_since_abort < 2.0:
            pulse = 0.55 + 0.45 * abs(np.sin(elapsed_since_abort * 6.28))
        else:
            pulse = 0.85
        text = f"ABORT @ {abort_t:.2f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 1.15, 3
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        pad = 16
        rx0 = w // 2 - (tw + 2 * pad) // 2
        ry0 = max(banner_h + 10, 86)
        rx1 = rx0 + tw + 2 * pad
        ry1 = ry0 + th_ + 2 * pad
        overlay = out.copy()
        cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (rx0, ry0), (rx1, ry1),
                      (40, 40, int(80 + 175 * pulse)), 3)
        cv2.putText(out, text, (rx0 + pad, ry1 - pad),
                    font, scale, (60, 60, int(80 + 175 * pulse)), thick, cv2.LINE_AA)

    # Big centered reveal card (used at the end of each test, never during)
    if reveal_text is not None:
        rcol = reveal_color or (220, 220, 220)
        rect_w, rect_h = 760, 160
        rx = (w - rect_w) // 2
        ry = (h - rect_h) // 2 - 40
        overlay = out.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rect_w, ry + rect_h), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (rx, ry), (rx + rect_w, ry + rect_h), rcol, 3)
        lines = reveal_text.split("\n")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scales = [1.5, 1.0]
        thicknesses = [4, 2]
        colors = [rcol, (220, 220, 220)]
        y = ry + 70
        for li, line in enumerate(lines[:2]):
            sc = scales[li] if li < len(scales) else 0.7
            th = thicknesses[li] if li < len(thicknesses) else 1
            co = colors[li] if li < len(colors) else (200, 200, 200)
            (tw, _), _ = cv2.getTextSize(line, font, sc, th)
            cv2.putText(out, line, (w // 2 - tw // 2, y),
                        font, sc, co, th, cv2.LINE_AA)
            y += int(40 * sc + 20)

    if wrist_image is not None:
        pip_h = 180
        pip_w = 180
        pip_resized = cv2.resize(wrist_image, (pip_w, pip_h),
                                 interpolation=cv2.INTER_AREA)
        px, py = w - pip_w - 16, 86
        cv2.rectangle(out, (px - 3, py - 3), (px + pip_w + 3, py + pip_h + 3),
                      (240, 240, 240), 2)
        out[py:py + pip_h, px:px + pip_w] = pip_resized
        cv2.rectangle(out, (px, py + pip_h + 4), (px + pip_w, py + pip_h + 26),
                      (10, 12, 18), -1)
        cv2.putText(out, "DROID wrist cam (input to LSTM)",
                    (px + 4, py + pip_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 220), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Title card: 1.5s of static-robot frames with large overlay announcing
# the next act. Robot is at HOME (post-pause or fresh from keyframe reset).
# ─────────────────────────────────────────────────────────────────────────
def render_title_card(vw, mj_model, mj_data, renderer, cam,
                      act_index, total_acts, act_label, seq_basename,
                      width, height, fps, duration_sec):
    n_frames = int(duration_sec * fps)
    col_label = (60, 200, 60) if act_label == "SUCCESS" else (60, 60, 240)
    line1 = f"ACT {act_index + 1} / {total_acts}"
    line2 = act_label
    line3 = seq_basename
    rect_w, rect_h = 720, 200
    rx = (width - rect_w) // 2
    ry = (height - rect_h) // 2
    for _ in range(n_frames):
        renderer.update_scene(mj_data, camera=cam)
        frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rect_w, ry + rect_h), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (rx, ry), (rx + rect_w, ry + rect_h), col_label, 3)

        (tw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame_bgr, line1, (width // 2 - tw1 // 2, ry + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 3, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        cv2.putText(frame_bgr, line2, (width // 2 - tw2 // 2, ry + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col_label, 4, cv2.LINE_AA)
        (tw3, _), _ = cv2.getTextSize(line3, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame_bgr, line3, (width // 2 - tw3 // 2, ry + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 200), 1, cv2.LINE_AA)
        vw.write(frame_bgr)


def reset_scene_keep_arm(mj_model, mj_data, key_id, arm_joint_qadrs):
    """Reset cube + sphere via keyframe, but preserve current arm pose so the
    transition from inter-act-pause is seamless."""
    saved_arm = [float(mj_data.qpos[adr]) for adr in arm_joint_qadrs]
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    for adr, q in zip(arm_joint_qadrs, saved_arm):
        mj_data.qpos[adr] = q
    mujoco.mj_forward(mj_model, mj_data)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-config", required=True,
                   help="JSON list of acts: [{label, sequence, offset, duration, [pip_seqs], [mode]}]")
    p.add_argument("--model-dir", default="results/multimodal_indist_v9_sharp")
    p.add_argument("--output", default="results/module_c_franka/shadow_loop.mp4")
    p.add_argument("--threshold", type=float, default=0.806,
                   help="P(failure) threshold to trigger ABORT in active mode (val-tuned: 0.806)")
    p.add_argument("--n-consecutive", type=int, default=3)
    p.add_argument("--passive-acts", type=int, default=0,
                   help="Number of acts to run in passive (no-abort) mode at the start. "
                        "LSTM still computes P(failure) but the robot ignores it. "
                        "Use this for a 'monitoring-only' Part 1 of the video.")
    p.add_argument("--plot-y-lo", type=float, default=0.0,
                   help="P(failure) plot lower bound (default 0.0)")
    p.add_argument("--plot-y-hi", type=float, default=1.0,
                   help="P(failure) plot upper bound (default 1.0)")
    p.add_argument("--part-card-seconds", type=float, default=3.5,
                   help="Duration of part-transition disclaimer card between Parts 1 and 2")
    p.add_argument("--part1-disclaimer", default=(
        "PART 1 - PASSIVE MONITORING||"
        "The LSTM only OBSERVES the trajectory; the robot does NOT react to its predictions.||"
        "When P(failure) rises, watch the model 'see' the failure even though the robot keeps going."
    ), help="Disclaimer overlaid during passive acts. Use '||' to separate lines.")
    p.add_argument("--part2-disclaimer", default=(
        "PART 2 - LSTM IN CONTROL||"
        "The LSTM is now allowed to ABORT the grasp if it predicts failure.||"
        "When P(failure) crosses the threshold, the gripper opens to release the object safely."
    ), help="Disclaimer overlaid during active (abort-enabled) acts.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--inter-act-pause", type=float, default=1.5)
    p.add_argument("--title-card-seconds", type=float, default=1.5)
    p.add_argument("--aftermath-seconds", type=float, default=2.0)
    p.add_argument("--no-pip", action="store_true",
                   help="disable wrist-camera Picture-in-Picture (skips Mosaico image fetch)")
    p.add_argument("--disclaimer", default=(
        "DROID test split - sequences NEVER seen by the model (held out from training and validation).||"
        "The model only receives the kinematics + wrist camera frames; it does NOT see the ground-truth label.||"
        "P(failure) below is the model's live prediction. The episode outcome is shown so you can judge if the model gets it right or wrong."),
        help="Top banner disclaimer text. Use '||' to separate lines, or empty string to disable.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.acts_config) as f:
        acts = json.load(f)
    log.info("=== Shadow-Loop Demo: %d acts ===", len(acts))
    for i, a in enumerate(acts):
        log.info("  [%d/%d] %s  %s  offset=%d  duration=%.1fs", i + 1, len(acts),
                 a["label"], os.path.basename(a["sequence"]),
                 a.get("offset", 0), a.get("duration", 12.0))

    (model, kin_mean, kin_std, cnn_mean, cnn_std, ipca_c, ipca_m, n_comp) = load_model(args.model_dir)
    log.info("model loaded (IPCA=%d)", n_comp)

    (mj_model, mj_data, renderer, arm_acts, gripper_act,
     ee_site, cube_qadr, sphere_qadr, key_id, arm_joint_qadrs) = setup_franka_hd(
         args.width, args.height)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 135.0
    cam.elevation = -22.0
    cam.distance = 1.6
    cam.lookat[:] = [0.45, 0.0, 0.42]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    if not vw.isOpened():
        out = os.path.splitext(args.output)[0] + ".avi"
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"XVID"), args.fps,
                             (args.width, args.height))
        args.output = out
    log.info("Video: %s @ %d fps (%dx%d)", args.output, args.fps, args.width, args.height)

    SUBSTEPS = 10
    disclaimer_lines = [s for s in (args.disclaimer or "").split("||") if s.strip()]

    # PiP wrist-cam: collect every DROID seq referenced by any act, both via
    # explicit `pip_seqs` (concat case) and via the `sequence` .npz basename
    # (single-sequence case). For single-sequence acts the DROID name is
    # inferred from the cache filename (results/cnn_cache_spatial/droid/<name>.npz).
    pip_cache: dict[str, list] = {}  # {seq_name: [bgr frames]}
    if not args.no_pip:
        wanted = set()
        for a in acts:
            wanted.update(a.get("pip_seqs", []))
            seq_path = a.get("sequence", "").replace("\\", "/")
            if "cnn_cache_spatial/droid/" in seq_path:
                wanted.add(os.path.splitext(os.path.basename(seq_path))[0])
        if wanted:
            log.info("Loading wrist-camera images via Mosaico SDK for %d sequences...", len(wanted))
            try:
                pip_cache = load_droid_wrist_images(sorted(wanted))
            except Exception as e:
                log.warning("PiP image load failed: %s -- continuing without", e)

    def run_act(act_index, total_acts, act_label, target, sequence_path, offset,
                duration_sec, pip_images=None, passive: bool = False,
                disclaimer_override=None, slip: bool = False,
                slip_at: float = -1.0, grip_close: float = 50.0):
        """Run one act. target in {'cube','sphere'} chooses the trajectory and
        anchorage independently from act_label. The act_label only determines
        which Mosaico sequence is fed to the model (and thus whether ABORT is
        expected to trigger).

        passive=True disables the ABORT logic entirely: the LSTM still computes
        P(failure) and the HUD plots it, but the robot ignores the prediction
        and finishes the scripted trajectory regardless. Used for Part 1
        (passive monitoring) where detection is shown without actuation.

        slip=True disables the sphere anchor for the WHOLE act so a
        frictionless ball never gets attached to the gripper - the gripper
        closes around empty air.

        slip_at>0 enables a DELAYED slip: the sphere is anchored normally
        when the gripper closes (so the robot 'grasps' the ball), then the
        anchor is released at sim_t == slip_at, letting the ball fall mid-
        lift while the robot continues unaware. Use this for a clean
        'gripper succeeds, then loses grip' visual in passive failure acts.
        """
        seq_basename = os.path.basename(sequence_path)
        with np.load(sequence_path) as z:
            kin_seq = z["kin"].astype(np.float32)
            cnn_seq = z["cnn"]
            label_seq = z["label"].astype(np.float32)
        log.info("Act %d/%d (%s on %s): seq T=%d  offset=%d", act_index + 1, total_acts,
                 act_label, target, len(kin_seq), offset)

        if target == "sphere":
            script = SPHERE_SCRIPT
            # Default: anchor sphere to gripper while closed (friction=0).
            # slip=True disables the anchor for the entire act -> ball stays
            # behind from the moment the gripper closes.
            # slip_at>0 keeps the anchor live until sim_t reaches slip_at,
            # then releases it -> ball falls mid-lift after a clean grasp.
            anchor_qadr = None if slip else sphere_qadr
        elif target == "cube":
            script = CUBE_SCRIPT
            anchor_qadr = None         # real friction holds the cube
        else:
            raise ValueError(f"Unknown target: {target}")

        # Per-act override of the gripper close value. Default 50 = firm grip
        # (sufficient for cube friction during lift). Higher values (e.g. 130)
        # produce a weaker grip: the fingers do not close fully around the
        # cube, so the contact normal force during lift is lower and friction
        # may not overcome the lift acceleration -> physical slip. This is a
        # static physical parameter (gripper command), not a scripted event:
        # MuJoCo physics decides if and when the cube slips.
        if grip_close != 50.0:
            log.info("  grip_close override: %.1f (weak grip)", grip_close)
            script = [
                (ts, te, qpos, (grip_close if grip < 200.0 else grip))
                for ts, te, qpos, grip in script
            ]

        # Allow per-act duration override (script is parametric in time)
        script = [(ts, te, q, g) for (ts, te, q, g) in script
                  if ts < duration_sec]
        if script and script[-1][1] > duration_sec:
            ts, _, q, g = script[-1]
            script[-1] = (ts, duration_sec, q, g)

        n_ticks = int(duration_sec / 0.02)
        consec = 0
        mode = "EXECUTING"
        abort_t = -1.0
        abort_pose = None
        prob_hist: list[float] = []
        hud_label = f"TEST {act_index + 1}/{total_acts}"

        for k in range(n_ticks):
            sim_t = k * 0.02

            if mode == "EXECUTING":
                qpos_t, grip_t = script_target(sim_t, script)
                for j, act in enumerate(arm_acts):
                    mj_data.ctrl[act] = float(qpos_t[j])
                mj_data.ctrl[gripper_act] = float(grip_t)
            else:
                for j, act in enumerate(arm_acts):
                    mj_data.ctrl[act] = float(abort_pose[j])
                mj_data.ctrl[gripper_act] = 255.0

            # Sphere anchor: hold the frictionless ball in the gripper while
            # closed. If slip_at is set and the simulation time has passed it,
            # release the anchor so the ball drops mid-lift (delayed slip visual).
            anchor_active = (anchor_qadr is not None and mode == "EXECUTING"
                             and (slip_at < 0.0 or sim_t < slip_at))
            if anchor_active:
                _, current_grip = script_target(sim_t, script)
                if current_grip < 200.0:
                    ee_pos = mj_data.site_xpos[ee_site].copy()
                    mj_data.qpos[anchor_qadr:anchor_qadr + 3] = ee_pos
                    mj_data.qpos[anchor_qadr + 3:anchor_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                    mj_data.qvel[anchor_qadr:anchor_qadr + 6] = 0.0

            prob = float("nan")
            gt_label = -1
            seq_idx = offset + k
            if 0 <= seq_idx and seq_idx + SEQ_LEN <= len(kin_seq):
                kw = kin_seq[seq_idx:seq_idx + SEQ_LEN]
                cw = cnn_seq[seq_idx:seq_idx + SEQ_LEN]
                prob = model_proba(model, kw, cw, kin_mean, kin_std, cnn_mean, cnn_std,
                                   ipca_c, ipca_m)
                gt_label = int(label_seq[seq_idx + SEQ_LEN - 1] > 0.5)
                if mode == "EXECUTING" and not passive:
                    if prob >= args.threshold:
                        consec += 1
                        if consec >= args.n_consecutive:
                            mode = "ABORTED"
                            abort_t = sim_t
                            abort_pose = np.array(
                                [float(mj_data.qpos[adr]) for adr in arm_joint_qadrs])
                            log.info(">>> ABORT @ sim_t=%.2fs (P=%.3f)", sim_t, prob)
                    else:
                        consec = 0
            prob_hist.append(prob)

            for _ in range(SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)

            renderer.update_scene(mj_data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            wrist = None
            if pip_images is not None and len(pip_images) > 0:
                pip_idx = min(seq_idx + SEQ_LEN - 1, len(pip_images) - 1)
                if 0 <= pip_idx:
                    wrist = pip_images[pip_idx]
            disclaimer_for_frame = (disclaimer_override
                                    if disclaimer_override is not None
                                    else disclaimer_lines)
            frame_bgr = draw_hud(
                frame_bgr, sim_t=sim_t, dur=duration_sec,
                prob=prob, threshold=args.threshold, mode=mode,
                gt_label=gt_label, abort_t=abort_t, prob_history=prob_hist,
                act_label=hud_label, wrist_image=wrist,
                show_gt=False, reveal_text=None,
                disclaimer_lines=disclaimer_for_frame, seq_name=seq_basename,
                plot_y_lo=args.plot_y_lo, plot_y_hi=args.plot_y_hi,
                passive=passive,
            )
            vw.write(frame_bgr)
            if k % 100 == 0:
                log.info("  t=%5.2fs mode=%-9s P=%5.3f gt=%d %s",
                         sim_t, mode, prob, gt_label,
                         "(passive)" if passive else "")
        return mode, abort_t, prob_hist

    def run_aftermath(duration_sec, hud_label, mode, abort_t, last_prob, gt_label, prob_hist,
                      reveal_text=None, reveal_color=None, hud_seq_name=None,
                      disclaimer_override=None, passive: bool = False):
        n = int(duration_sec / 0.02)
        for _ in range(n):
            for _ in range(SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            disclaimer_for_frame = (disclaimer_override
                                    if disclaimer_override is not None
                                    else disclaimer_lines)
            frame_bgr = draw_hud(
                frame_bgr, sim_t=duration_sec, dur=duration_sec,
                prob=last_prob, threshold=args.threshold, mode=mode,
                gt_label=gt_label, abort_t=abort_t, prob_history=prob_hist,
                act_label=hud_label, show_gt=True,
                reveal_text=reveal_text, reveal_color=reveal_color,
                disclaimer_lines=disclaimer_for_frame, seq_name=hud_seq_name,
                plot_y_lo=args.plot_y_lo, plot_y_hi=args.plot_y_hi,
                passive=passive,
            )
            vw.write(frame_bgr)

    def run_inter_act_pause(duration_sec, disclaimer_override=None):
        n = int(duration_sec / 0.02)
        for k in range(n):
            a = k / max(n - 1, 1)
            cur_pose = np.array([float(mj_data.qpos[adr]) for adr in arm_joint_qadrs])
            target = (1 - a) * cur_pose + a * HOME
            for j, act in enumerate(arm_acts):
                mj_data.ctrl[act] = float(target[j])
            mj_data.ctrl[gripper_act] = 255.0
            for _ in range(SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            disclaimer_for_frame = (disclaimer_override
                                    if disclaimer_override is not None
                                    else disclaimer_lines)
            frame_bgr = draw_hud(
                frame_bgr, sim_t=k * 0.02, dur=duration_sec,
                prob=float("nan"), threshold=args.threshold, mode="EXECUTING",
                gt_label=-1, abort_t=-1.0, prob_history=[],
                act_label="Transition: returning to home",
                disclaimer_lines=disclaimer_for_frame, seq_name=None,
                plot_y_lo=args.plot_y_lo, plot_y_hi=args.plot_y_hi,
            )
            vw.write(frame_bgr)

    def render_part_card(duration_sec, headline, body_lines, accent_color):
        """Big full-screen card announcing a transition between video parts.
        Robot stays at HOME (no physics); only the static overlay changes."""
        n = int(duration_sec * args.fps)
        h, w = args.height, args.width
        for _ in range(n):
            renderer.update_scene(mj_data, camera=cam)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (5, 7, 12), -1)
            cv2.addWeighted(overlay, 0.88, frame_bgr, 0.12, 0, frame_bgr)
            # Headline
            (tw, _), _ = cv2.getTextSize(headline, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
            cv2.putText(frame_bgr, headline, (w // 2 - tw // 2, h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, accent_color, 4, cv2.LINE_AA)
            cv2.line(frame_bgr, (w // 2 - 200, h // 2 - 50),
                     (w // 2 + 200, h // 2 - 50), accent_color, 2)
            # Body
            for i, line in enumerate(body_lines):
                (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                cv2.putText(frame_bgr, line,
                            (w // 2 - tw // 2, h // 2 + 10 + i * 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 225, 235),
                            1, cv2.LINE_AA)
            vw.write(frame_bgr)

    total_acts = len(acts)
    # Resolve per-part disclaimers (Part 1 = passive, Part 2 = active with abort).
    # The CLI default disclaimer is preserved as a fallback for single-part runs
    # where --passive-acts == 0.
    part1_disc = [s for s in (args.part1_disclaimer or "").split("||") if s.strip()]
    part2_disc = [s for s in (args.part2_disclaimer or "").split("||") if s.strip()]
    n_passive = max(0, min(args.passive_acts, total_acts))

    for i, act in enumerate(acts):
        if i > 0:
            reset_scene_keep_arm(mj_model, mj_data, key_id, arm_joint_qadrs)

        # Part-transition card BEFORE the first active act, only when a
        # passive part is configured (--passive-acts > 0).
        if i == n_passive and n_passive > 0:
            log.info("=== Rendering Part 2 transition card ===")
            render_part_card(
                args.part_card_seconds,
                headline="PART 2 - LSTM IN CONTROL",
                body_lines=[
                    "We now hand the same LSTM the safety override.",
                    "If P(failure) crosses the threshold for several consecutive",
                    "frames, the gripper opens to release the object safely.",
                    "Same model, same sequences, only the control loop changes.",
                ],
                accent_color=(60, 70, 230),  # red-ish for 'control / abort'
            )

        passive = (i < n_passive)
        disclaimer_override = part1_disc if passive else part2_disc

        pip_imgs = None
        if not args.no_pip:
            if "pip_seqs" in act:
                # Concat case: stitch frames of listed seqs in order so the
                # offset within the .npz aligns with the flattened pip list.
                stitched = []
                for s in act["pip_seqs"]:
                    stitched.extend(pip_cache.get(s, []))
                if stitched:
                    pip_imgs = stitched
            else:
                seq_path = act.get("sequence", "").replace("\\", "/")
                if "cnn_cache_spatial/droid/" in seq_path:
                    seq_name = os.path.splitext(os.path.basename(seq_path))[0]
                    frames = pip_cache.get(seq_name)
                    if frames:
                        pip_imgs = frames
        # Default target so old configs keep working:
        # SUCCESS -> sphere, FAILURE -> cube. Override per-act with `target`.
        default_target = "sphere" if act["label"] == "SUCCESS" else "cube"
        target = act.get("target", default_target)
        mode, abort_t, prob_hist = run_act(
            act_index=i, total_acts=total_acts,
            act_label=act["label"], target=target,
            sequence_path=act["sequence"],
            offset=int(act.get("offset", 0)),
            duration_sec=float(act.get("duration", 12.0)),
            pip_images=pip_imgs,
            passive=passive,
            disclaimer_override=disclaimer_override,
            slip=bool(act.get("slip", False)),
            slip_at=float(act.get("slip_at", -1.0)),
            grip_close=float(act.get("grip_close", 50.0)),
        )

        last_prob = prob_hist[-1] if prob_hist else float("nan")
        gt_for_aftermath = 1 if act["label"] == "FAILURE" else 0
        # Reveal: show OUTCOME only after the test is over, never before.
        # In passive mode the LSTM never aborts, so the reveal shows the
        # ground-truth outcome and what the model PREDICTED.
        if passive:
            peak = max((p for p in prob_hist if not np.isnan(p)), default=float("nan"))
            gt_word = "FAILURE" if act["label"] == "FAILURE" else "SUCCESS"
            reveal_text = (
                f"TEST {i + 1}/{total_acts} - GT: {gt_word}\n"
                f"Peak P(failure) = {peak:.2f}"
            )
            reveal_color = (60, 60, 240) if act["label"] == "FAILURE" else (60, 200, 60)
        elif mode == "ABORTED":
            reveal_text = f"TEST {i + 1}/{total_acts}\nFAILURE  (ABORT @ {abort_t:.2f}s)"
            reveal_color = (60, 60, 240)
        else:
            reveal_text = f"TEST {i + 1}/{total_acts}\nSUCCESS"
            reveal_color = (60, 200, 60)
        aftermath_label = f"TEST {i + 1}/{total_acts}"
        run_aftermath(args.aftermath_seconds, aftermath_label, mode, abort_t,
                      last_prob, gt_for_aftermath, prob_hist,
                      reveal_text=reveal_text, reveal_color=reveal_color,
                      hud_seq_name=os.path.basename(act["sequence"]),
                      disclaimer_override=disclaimer_override)

        if i < total_acts - 1:
            # When transitioning into Part 2 the part-card already covers it.
            next_disc = (part1_disc if (i + 1) < n_passive else part2_disc)
            run_inter_act_pause(args.inter_act_pause,
                                disclaimer_override=next_disc)

    vw.release()
    renderer.close()
    log.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    sys.exit(main())
