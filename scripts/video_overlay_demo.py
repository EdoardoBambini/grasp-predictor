"""Render per-act MP4s: source video frames overlaid with the classifier's
per-frame P(failure) and an optional abort-on-threshold freeze."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cached_lstm import LateFusionLSTM

SEQ_LEN = 50
KIN_DIM = 15
CACHE_FPS = 50.0
THRESHOLD_DEFAULT = 0.806

CANVAS_W = 1280
CANVAS_H = 720
BANNER_H = 50
HUD_H = 150
VIDEO_H = CANVAS_H - BANNER_H - HUD_H

BG_COLOR = (12, 14, 22)
PANEL_COLOR = (20, 24, 36)
SAFE_COLOR = (60, 220, 80)
WARN_COLOR = (60, 165, 240)
ALARM_COLOR = (60, 80, 240)

FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


def _font_cache():
    cache = {}
    def get(path: str, size: int):
        key = (path, size)
        if key not in cache:
            cache[key] = ImageFont.truetype(path, size)
        return cache[key]
    return get


_font = _font_cache()


def load_model(model_dir: str):
    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    n_components = int(sc["ipca_n_components"])
    cfg = {}
    cfg_path = os.path.join(model_dir, "results.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f).get("config", {})
    hidden = int(cfg.get("hidden", 256))
    dropout = float(cfg.get("dropout", 0.35))
    attn_pool = bool(cfg.get("attn_pool", False))
    print(f"  model: hidden={hidden} dropout={dropout} attn_pool={attn_pool} ipca={n_components}")
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


def open_reassemble_video(h5_path: str, blob_key: str = "hand"):
    with h5py.File(h5_path, "r") as f:
        blob = f[blob_key][()]
        raw = blob.tobytes() if hasattr(blob, "tobytes") else bytes(blob)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        raise RuntimeError(f"could not open extracted MP4 from {h5_path}/{blob_key}")
    return cap, tmp.name


def open_droid_video(parquet_path: str, episode_id: int,
                     camera: str = "exterior_1_left"):
    parquet_dir = Path(parquet_path).parent
    file_stem = Path(parquet_path).stem
    chunk = parquet_dir.name
    droid_root = parquet_dir.parent.parent
    mp4 = droid_root / "videos" / f"observation.images.{camera}" / chunk / f"{file_stem}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"DROID MP4 not found: {mp4}")
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {mp4}")

    import pyarrow.parquet as pq
    df = pq.read_table(str(parquet_path),
                       columns=["episode_index", "index"]).to_pandas()
    ep_mask = df["episode_index"] == episode_id
    if not ep_mask.any():
        raise ValueError(f"episode {episode_id} not in {parquet_path}")
    local_start = int(df[ep_mask].index[0])
    parquet_total = len(df)
    mp4_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mp4_start = int(round(local_start * mp4_total / parquet_total))
    print(f"  droid ep{episode_id}: parquet_local_idx={local_start}/{parquet_total}, "
          f"mp4_start={mp4_start}/{mp4_total}")
    return cap, mp4_start


def color_for_p(p: float, threshold: float) -> tuple[int, int, int]:
    if p < threshold * 0.5:
        return SAFE_COLOR
    if p < threshold:
        a = (p - threshold * 0.5) / (threshold * 0.5)
        return (
            int(SAFE_COLOR[0] * (1 - a) + WARN_COLOR[0] * a),
            int(SAFE_COLOR[1] * (1 - a) + WARN_COLOR[1] * a),
            int(SAFE_COLOR[2] * (1 - a) + WARN_COLOR[2] * a),
        )
    return ALARM_COLOR


def _bgr_to_rgb_tuple(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


def render_canvas(video_frame: np.ndarray, *, t_sec: float, dur_sec: float,
                  p: float, threshold: float, p_history: list, gt_label: int,
                  abort_at: float | None, show_gt: bool, dataset: str,
                  seq_name: str, act_label: str, part_label: str) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    p_color = color_for_p(p, threshold)
    is_alarm = p >= threshold
    in_abort = abort_at is not None and t_sec >= abort_at

    # === video area ===
    video_y0 = BANNER_H
    video_y1 = video_y0 + VIDEO_H
    src_h, src_w = video_frame.shape[:2]
    scale = min(CANVAS_W / src_w, VIDEO_H / src_h)
    disp_w = int(src_w * scale)
    disp_h = int(src_h * scale)
    disp = cv2.resize(video_frame, (disp_w, disp_h),
                      interpolation=cv2.INTER_LINEAR)
    vx0 = (CANVAS_W - disp_w) // 2
    vy0 = video_y0 + (VIDEO_H - disp_h) // 2
    canvas[vy0:vy0 + disp_h, vx0:vx0 + disp_w] = disp

    border_color = ALARM_COLOR if (is_alarm or in_abort) else (60, 70, 100)
    cv2.rectangle(canvas, (vx0 - 2, vy0 - 2),
                  (vx0 + disp_w + 1, vy0 + disp_h + 1), border_color, 2)

    # === switch to PIL for clean text rendering ===
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Banner top
    draw.rectangle([0, 0, CANVAS_W, BANNER_H], fill=(20, 24, 36))
    draw.line([0, BANNER_H, CANVAS_W, BANNER_H], fill=(40, 50, 80), width=1)
    draw.text((16, 8),
              f"Mosaico SDK   |   {dataset}",
              font=_font(FONT_REG, 16), fill=(220, 220, 230))
    draw.text((16, 28),
              f"{seq_name}   |   act {act_label}   |   t {t_sec:5.2f} / {dur_sec:.1f} s",
              font=_font(FONT_MONO, 12), fill=(160, 175, 200))
    pl_text = part_label
    pl_font = _font(FONT_BOLD, 15)
    bbox = draw.textbbox((0, 0), pl_text, font=pl_font)
    pl_w = bbox[2] - bbox[0]
    draw.text((CANVAS_W - pl_w - 16, 17), pl_text,
              font=pl_font, fill=(180, 200, 230))

    if show_gt:
        gt_color = ALARM_COLOR if gt_label else SAFE_COLOR
        gt_text = f"GT: {'FAILURE' if gt_label else 'SUCCESS'}"
        gt_font = _font(FONT_BOLD, 14)
        bbox_gt = draw.textbbox((0, 0), gt_text, font=gt_font)
        draw.text((CANVAS_W - (bbox_gt[2] - bbox_gt[0]) - 16, 32),
                  gt_text, font=gt_font,
                  fill=_bgr_to_rgb_tuple(gt_color))

    # === bottom HUD: full-width timeline only ===
    hud_y0 = video_y1
    draw.rectangle([0, hud_y0, CANVAS_W, CANVAS_H], fill=(20, 24, 36))
    draw.line([0, hud_y0, CANVAS_W, hud_y0], fill=(40, 50, 80), width=1)

    draw.text((24, hud_y0 + 8),
              "P(failure) timeline   |   live LSTM inference @ 50 Hz",
              font=_font(FONT_BOLD, 13), fill=(170, 185, 215))

    plot_x0 = 24
    plot_x1 = CANVAS_W - 24
    plot_y0 = hud_y0 + 30
    plot_y1 = CANVAS_H - 16
    plot_h = plot_y1 - plot_y0

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], fill=(8, 10, 16))

    # threshold line
    thr_y = int(plot_y1 - threshold * plot_h)
    draw.line([plot_x0, thr_y, plot_x1, thr_y],
              fill=_bgr_to_rgb_tuple(WARN_COLOR), width=1)
    draw.text((plot_x0 + 8, thr_y - 16),
              f"threshold = {threshold:.3f}",
              font=_font(FONT_MONO, 11),
              fill=_bgr_to_rgb_tuple(WARN_COLOR))

    # P trace: time-synced x position
    n_pts = len(p_history)
    if n_pts >= 2 and dur_sec > 0:
        # P is sampled once per output frame, so the i-th sample maps to time
        # i * dt with dt = t_sec / (n_pts - 1).
        dt = t_sec / (n_pts - 1) if n_pts > 1 else 0.0
        pts = []
        for i in range(n_pts):
            t_i = i * dt
            x = plot_x0 + int((t_i / dur_sec) * (plot_x1 - plot_x0))
            y = int(plot_y1 - p_history[i] * plot_h)
            pts.append((x, y))
        for i in range(1, n_pts):
            seg_color = (ALARM_COLOR if p_history[i] >= threshold
                         else SAFE_COLOR)
            draw.line([pts[i - 1], pts[i]],
                      fill=_bgr_to_rgb_tuple(seg_color), width=2)
        cur = pts[-1]
        draw.ellipse([cur[0] - 5, cur[1] - 5, cur[0] + 5, cur[1] + 5],
                     fill=_bgr_to_rgb_tuple(p_color))

        # P value floating label near cursor
        p_label = f"P = {p:.3f}"
        p_lbl_font = _font(FONT_BOLD, 18)
        bbox_p = draw.textbbox((0, 0), p_label, font=p_lbl_font)
        lbl_w = bbox_p[2] - bbox_p[0]
        lbl_x = min(cur[0] + 12, plot_x1 - lbl_w - 6)
        lbl_y = max(plot_y0 + 4, cur[1] - 22)
        draw.rectangle([lbl_x - 4, lbl_y - 2, lbl_x + lbl_w + 4, lbl_y + 22],
                       fill=(8, 10, 16))
        draw.text((lbl_x, lbl_y), p_label,
                  font=p_lbl_font, fill=_bgr_to_rgb_tuple(p_color))

    # cursor (vertical line at current time)
    if dur_sec > 0:
        cur_x = plot_x0 + int(t_sec / dur_sec * (plot_x1 - plot_x0))
        draw.line([cur_x, plot_y0, cur_x, plot_y1],
                  fill=(200, 200, 220), width=1)

    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if in_abort:
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (CANVAS_W, video_y1), (0, 0, 200), -1)
        canvas = cv2.addWeighted(ov, 0.18, canvas, 0.82, 0)
        pil2 = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d2 = ImageDraw.Draw(pil2)
        msg = "ABORT INITIATED BY MOSAICO ML"
        msg_font = _font(FONT_BOLD, 36)
        bbox = d2.textbbox((0, 0), msg, font=msg_font)
        msg_w = bbox[2] - bbox[0]
        msg_x = (CANVAS_W - msg_w) // 2
        msg_y = video_y0 + 24
        d2.rectangle([msg_x - 18, msg_y - 8, msg_x + msg_w + 18, msg_y + 50],
                     fill=(0, 0, 0))
        d2.text((msg_x, msg_y), msg, font=msg_font, fill=(255, 255, 255))
        canvas = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)

    return canvas


def process_act(act: dict, model_bundle, *, threshold: float, active_abort: bool,
                out_path: str, part_label: str) -> int:
    seq_path = act["sequence"]
    offset = int(act["offset"])
    duration = float(act["duration"])
    label = act["label"]

    print(f"\n[act {label}] sequence={seq_path}, offset={offset}, "
          f"duration={duration}s, abort={active_abort}")

    cache = np.load(seq_path)
    kin_seq = cache["kin"].astype(np.float32)
    cnn_seq = cache["cnn"]
    label_seq = cache["label"].astype(np.float32)

    is_droid = "droid/" in seq_path.replace("\\", "/")
    is_reassemble = "reassemble/" in seq_path.replace("\\", "/")

    tmp_files = []
    video_start = 0
    if is_reassemble:
        date = Path(seq_path).stem.replace("reassemble_", "")
        h5_path = f"D:/datasets/reassemble/data/{date}.h5"
        camera = act.get("camera", "hand")
        cap, tmp = open_reassemble_video(h5_path, camera)
        tmp_files.append(tmp)
        dataset = "Reassemble  (NIST Assembly Task Board)"
        seq_name = f"reassemble_{date}"
        # Reassemble offset is in 50Hz npz units; map to video fps
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_start = int(round(offset / CACHE_FPS * video_fps))
    elif is_droid:
        stem = Path(seq_path).stem
        ep_id = int(stem.split("_ep")[-1])
        file_stem = stem.split("_ep")[0].replace("droid_", "")
        parquet_path = f"E:/datasets/droid/data/chunk-000/{file_stem}.parquet"
        cap, ep_mp4_start = open_droid_video(parquet_path, ep_id, "exterior_1_left")
        dataset = "DROID  (in-the-wild manipulation)"
        seq_name = stem
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # Within-episode offset: npz at 50Hz, video at video_fps.
        video_start = ep_mp4_start + int(round(offset / CACHE_FPS * video_fps))
    else:
        raise ValueError(f"unknown dataset for {seq_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  source: {src_w}x{src_h} @ {video_fps:.1f}fps, video_start={video_start}")

    out_fps = 30
    n_video_frames = int(round(duration * out_fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, out_fps, (CANVAS_W, CANVAS_H))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")

    p_history = []
    last_p = 0.0
    abort_at = None
    aborted_freeze_frame = None
    (model, kin_mean, kin_std, cnn_mean, cnn_std,
     ipca_c, ipca_m, _) = model_bundle
    is_failure_act = label.upper().startswith("FAIL")

    for k in range(n_video_frames):
        t_sec = k / out_fps

        npz_idx = offset + int(round(t_sec * CACHE_FPS))
        if npz_idx >= len(kin_seq):
            npz_idx = len(kin_seq) - 1
        if npz_idx >= SEQ_LEN:
            kw = kin_seq[npz_idx - SEQ_LEN:npz_idx]
            cw = cnn_seq[npz_idx - SEQ_LEN:npz_idx]
            p = model_proba(model, kw, cw, kin_mean, kin_std, cnn_mean, cnn_std,
                            ipca_c, ipca_m)
            last_p = p
        else:
            p = last_p
        p_history.append(p)
        gt = int(label_seq[min(npz_idx, len(label_seq) - 1)] > 0.5)

        if active_abort and is_failure_act and abort_at is None and p >= threshold:
            abort_at = t_sec
            print(f"  [ACTIVE ABORT] triggered at t={t_sec:.2f}s, P={p:.3f}")

        video_frame_idx = video_start + int(round(t_sec * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((src_h, src_w, 3), dtype=np.uint8)

        if abort_at is not None and t_sec >= abort_at:
            if aborted_freeze_frame is None:
                aborted_freeze_frame = frame.copy()
            frame = aborted_freeze_frame.copy()

        # The Reassemble label topic stores only single-frame boundary
        # markers (failure on at the start of a failed segment, off again
        # right after), so reading label_seq frame-by-frame would flicker.
        # The act-level label is authoritative; reveal it after the midpoint.
        if is_failure_act:
            gt_for_display = 1 if t_sec >= duration / 2 else 0
        else:
            gt_for_display = 0

        canvas = render_canvas(
            frame, t_sec=t_sec, dur_sec=duration,
            p=p, threshold=threshold, p_history=p_history,
            gt_label=gt_for_display, abort_at=abort_at,
            show_gt=(t_sec >= duration - 1.5),
            dataset=dataset, seq_name=seq_name, act_label=label,
            part_label=part_label,
        )
        vw.write(canvas)

    vw.release()
    cap.release()
    for tmp in tmp_files:
        try: os.unlink(tmp)
        except OSError: pass
    print(f"  wrote {out_path} ({n_video_frames} frames @ {out_fps}fps)")
    return n_video_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-config",
        default="results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json")
    ap.add_argument("--model-dir", default="results/multimodal_indist_v9_sharp")
    ap.add_argument("--act-idx", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--out-dir", default="results/video_overlay_demo")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading model")
    model_bundle = load_model(args.model_dir)

    print(f"\n[2/3] loading acts config {args.acts_config}")
    with open(args.acts_config) as f:
        acts = json.load(f)

    part_labels = ["PART 1 - Passive Monitoring",
                   "PART 1 - Passive Monitoring",
                   "PART 2 - Active Control (Abort Authority)",
                   "PART 2 - Active Control (Abort Authority)"]

    print(f"\n[3/3] rendering")
    indices = range(len(acts)) if args.act_idx < 0 else [args.act_idx]
    for i in indices:
        act = acts[i]
        active_abort = (i >= 2)
        out_path = os.path.join(args.out_dir, f"act_{i+1}_{act['label']}.mp4")
        process_act(act, model_bundle, threshold=args.threshold,
                    active_abort=active_abort,
                    out_path=out_path,
                    part_label=part_labels[i])
    print("\ndone.")


if __name__ == "__main__":
    sys.exit(main())
