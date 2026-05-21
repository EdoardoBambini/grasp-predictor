"""Render per-act MP4s: source video frames overlaid with the classifier's
per-frame P(failure) trace, in passive-monitoring mode."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mil_attention import LateFusionLSTMMIL

SEQ_LEN = 50
KIN_DIM = 15
CACHE_FPS = 50.0
THRESHOLD_DEFAULT = 0.55  # v10l per-dataset DROID operating threshold

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

# Default font paths (Windows). On Linux/macOS the renderer falls back to
# PIL's bundled default font; override via environment variables if you have
# specific fonts you'd like to use (e.g. DejaVu on Linux).
FONT_REG = os.environ.get("FONT_REG", "C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = os.environ.get("FONT_BOLD", "C:/Windows/Fonts/segoeuib.ttf")
FONT_MONO = os.environ.get("FONT_MONO", "C:/Windows/Fonts/consola.ttf")


def _font_cache():
    cache = {}
    def get(path: str, size: int):
        key = (path, size)
        if key not in cache:
            try:
                cache[key] = ImageFont.truetype(path, size)
            except (OSError, IOError):
                # Path doesn't exist (e.g. Linux/macOS without Windows fonts).
                # Fall back to PIL's bundled default so the renderer still runs.
                cache[key] = ImageFont.load_default()
        return cache[key]
    return get


_font = _font_cache()


def load_model(model_dir: str):
    sc = np.load(os.path.join(model_dir, "scaler.npz"))
    cfg = {}
    cfg_path = os.path.join(model_dir, "results.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f).get("config", {})
    hidden = int(cfg.get("hidden", 256))
    dropout = float(cfg.get("dropout", 0.35))
    cnn_dim = int(cfg.get("cnn_dim", 360))
    kin_dim = int(cfg.get("kin_dim_runtime", 16))
    bidir = cfg.get("lstm_direction", "uni") == "bi"
    pool_mode = cfg.get("mil_pool_mode", "gated")
    use_nl_concat = bool(cfg.get("use_nl_concat", True))
    nl_concat_dim = int(cfg.get("nl_concat_dim", 64))
    print(f"  model: LateFusionLSTMMIL hidden={hidden} dropout={dropout} "
          f"cnn_dim={cnn_dim} kin_dim={kin_dim} pool={pool_mode} "
          f"lstm={'bi' if bidir else 'uni'} nl_concat={use_nl_concat}")
    model = LateFusionLSTMMIL(
        kin_dim=kin_dim, cnn_dim=cnn_dim,
        kin_hidden=hidden // 4, vis_hidden=hidden // 4,
        num_layers=1, dropout=dropout, bidirectional=bidir,
        attn_pool=bool(cfg.get("attn_pool", True)),
        mil_attn_hidden=hidden // 2, head_hidden=hidden // 2,
        pool_mode=pool_mode,
        use_film=bool(cfg.get("use_film", False)),
        use_nl_concat=use_nl_concat, nl_concat_dim=nl_concat_dim,
    )
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "checkpoints", "best_model.pt"),
        map_location="cpu", weights_only=True))
    model.eval()
    return (model, sc["mean"], sc["std"], sc["cnn_mean"], sc["cnn_std"],
            kin_dim, cnn_dim)


def _prep_kin(kin_w, kin_mean, kin_std, kin_dim):
    # Pad the kinematic window to the model's runtime kin dim (DROID cache is
    # 15-D; the model expects 16-D with the trailing channel zero-padded), then
    # z-score with the saved scaler.
    kw = kin_w.astype(np.float32)
    if kw.shape[1] < kin_dim:
        kw = np.concatenate(
            [kw, np.zeros((kw.shape[0], kin_dim - kw.shape[1]), np.float32)], axis=1)
    elif kw.shape[1] > kin_dim:
        kw = kw[:, :kin_dim]
    return np.nan_to_num((kw - kin_mean) / kin_std, nan=0.0).astype(np.float32)


def _prep_cnn(cnn_w, cnn_mean, cnn_std):
    return np.nan_to_num((cnn_w.astype(np.float32) - cnn_mean) / cnn_std,
                         nan=0.0).astype(np.float32)


def prefix_bag_proba(model, pk_list, pc_list, n):
    """Bag-level probability over the first ``n`` windows of the episode.

    This is the model's running prediction for the sequence observed so far. It
    is the faithful online signal for a bag-level MIL classifier: a single
    per-window readout is unreliable because the head and pool are trained on
    full bags, not on one window in isolation. DROID carries no language
    instruction, so the null token is selected via is_droid=True.
    """
    xk = np.stack(pk_list[:n])[None]   # (1, n, T, kin_dim)
    xc = np.stack(pc_list[:n])[None]   # (1, n, T, cnn_dim)
    mask = torch.ones((1, n), dtype=torch.bool)
    nl = torch.zeros((1, 512), dtype=torch.float32)
    is_d = torch.ones((1,), dtype=torch.bool)
    probs, _ = model.predict_proba(torch.from_numpy(xk), torch.from_numpy(xc),
                                   mask, nl_emb=nl, is_droid=is_d)
    return float(probs[0].item())


def bag_proba(model, pk_list, pc_list, lo, hi):
    """Bag-level P(failure) over the contiguous window slice ``[lo:hi]``.

    Same readout as ``prefix_bag_proba`` but over an arbitrary window range, so
    FAILURE acts can be scored on a trailing tail window (the model's
    droid_tail_fraction=0.3 evaluation regime) while SUCCESS acts keep the full
    prefix bag. DROID carries no language instruction, so the null token is
    selected via is_droid=True.
    """
    n = hi - lo
    xk = np.stack(pk_list[lo:hi])[None]   # (1, n, T, kin_dim)
    xc = np.stack(pc_list[lo:hi])[None]   # (1, n, T, cnn_dim)
    mask = torch.ones((1, n), dtype=torch.bool)
    nl = torch.zeros((1, 512), dtype=torch.float32)
    is_d = torch.ones((1,), dtype=torch.bool)
    probs, _ = model.predict_proba(torch.from_numpy(xk), torch.from_numpy(xc),
                                   mask, nl_emb=nl, is_droid=is_d)
    return float(probs[0].item())


def open_reassemble_video(h5_path: str, blob_key: str = "hand"):
    import h5py  # lazy import: only the Reassemble path needs it
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


def find_shot(cap, center: int, half: int = 200, thresh: float = 35.0,
              min_len: int = 12):
    """Return the frames of the single continuous camera shot containing
    ``center``.

    DROID per-file mp4s concatenate many episodes back-to-back, and the
    parquet->mp4 frame mapping is only approximate, so a fixed-length read from
    an estimated start can splice in a neighboring episode (a visible scene
    cut). We read a window around the estimate, detect hard cuts via frame
    differencing, and clip to the shot that contains the anchor frame.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (center + half)
    s0 = max(0, center - half)
    s1 = min(total, center + half)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    frames, grays = [], []
    for _ in range(max(1, s1 - s0)):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
        grays.append(cv2.cvtColor(cv2.resize(fr, (160, 90)),
                                  cv2.COLOR_BGR2GRAY).astype(np.int16))
    if not frames:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 180
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
        return [np.zeros((h, w, 3), np.uint8)]
    diffs = [0.0] + [float(np.mean(np.abs(grays[i] - grays[i - 1])))
                     for i in range(1, len(frames))]
    c = max(0, min(center - s0, len(frames) - 1))
    left = 0
    for i in range(c, 0, -1):
        if diffs[i] > thresh:
            left = i
            break
    right = len(frames)
    for i in range(c + 1, len(frames)):
        if diffs[i] > thresh:
            right = i
            break
    shot = frames[left:right]
    if len(shot) < min_len:
        a = max(0, c - min_len)
        b = min(len(frames), c + min_len)
        shot = frames[a:b]
    return shot


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
                  show_gt: bool, dataset: str,
                  seq_name: str, act_label: str, part_label: str) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    p_color = color_for_p(p, threshold)
    is_alarm = p >= threshold

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

    border_color = ALARM_COLOR if is_alarm else (60, 70, 100)
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
              "P(failure)   |   MIL bag prediction (eval protocol)",
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
    return canvas


def process_act(act: dict, model_bundle, *, threshold: float,
                out_path: str, part_label: str,
                reassemble_h5_root: str, droid_parquet_root: str) -> int:
    seq_path = act["sequence"]
    offset = int(act["offset"])
    duration = float(act["duration"])
    label = act["label"]

    print(f"\n[act {label}] sequence={seq_path}, offset={offset}, "
          f"duration={duration}s")

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
        h5_path = os.path.join(reassemble_h5_root, f"{date}.h5")
        camera = act.get("camera", "hand")
        cap, tmp = open_reassemble_video(h5_path, camera)
        tmp_files.append(tmp)
        dataset = "Reassemble  (NIST Assembly Task Board)"
        seq_name = f"reassemble_{date}"
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_start = int(round(offset / CACHE_FPS * video_fps))
    elif is_droid:
        stem = Path(seq_path).stem
        ep_id = int(stem.split("_ep")[-1])
        file_stem = stem.split("_ep")[0].replace("droid_", "")
        parquet_path = os.path.join(droid_parquet_root, "chunk-000", f"{file_stem}.parquet")
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
    # Clip the source to the single continuous shot around the (approximate)
    # start estimate, so the act never splices in a neighboring episode. Play it
    # at ~native speed; the full P trace is fraction-synced over the same span.
    shot = find_shot(cap, video_start, half=200, thresh=35.0)
    n_shot = len(shot)
    duration = max(4.0, min(16.0, n_shot / max(video_fps, 1.0)))
    n_video_frames = int(round(duration * out_fps))
    print(f"  shot: {n_shot} frames -> duration {duration:.1f}s")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, out_fps, (CANVAS_W, CANVAS_H))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path}")

    (model, kin_mean, kin_std, cnn_mean, cnn_std,
     kin_dim, cnn_dim) = model_bundle
    is_failure_act = label.upper().startswith("FAIL")
    # Online readout to draw. Default preserves prior behavior (FAILURE ->
    # trailing-tail bag, SUCCESS -> full prefix bag); an act may override with
    # "readout": "tail" | "prefix" -- e.g. a SUCCESS act that should show the
    # moving tail-sliding signal instead of the (cumulative, self-stabilizing)
    # prefix bag. A tail readout on a success is the model's OOD regime, so the
    # trace can rise toward the alarm band mid-episode before settling low.
    readout = act.get("readout", "tail" if is_failure_act else "prefix")

    # Windows are always SEQ_LEN (50) frames long; a finer per-act stride
    # (e.g. 15-25) yields overlapping windows so partial-failure windows score
    # at intermediate values, giving a gradual failure rise instead of a single
    # jump. Defaults to SEQ_LEN (non-overlapping).
    trace_stride = int(act.get("trace_stride", SEQ_LEN))
    win_starts = list(range(0, len(kin_seq) - SEQ_LEN + 1, trace_stride))
    pk_list = [_prep_kin(kin_seq[s:s + SEQ_LEN], kin_mean, kin_std, kin_dim)
               for s in win_starts]
    pc_list = [_prep_cnn(cnn_seq[s:s + SEQ_LEN], cnn_mean, cnn_std)
               for s in win_starts]
    warmup_windows = 2
    M = len(win_starts)
    # Per-act readout, matching the model's evaluation protocol on DROID:
    #   FAILURE -> trailing-tail bag over the last K=round(0.3*M) windows. This
    #     is the regime the model was trained/evaluated on for DROID failures
    #     (droid_tail_fraction=0.3); the score rises low->high as the terminal
    #     failure enters the tail window.
    #   SUCCESS -> full prefix bag over every window seen so far (stays low).
    # The full-episode prefix bag is OOD for DROID failures (the model misses
    # them); the trailing-tail bag is OOD for successes (it false-alarms). There
    # is no single label-agnostic online readout for this bag-level model, so we
    # show each act under its evaluation protocol (disclosed on the intro card).
    tail_k = max(1, int(round(0.30 * M))) if M else 1
    p_by_n: dict = {}
    for n in range(warmup_windows, M + 1):
        if readout == "tail":
            p_by_n[n] = bag_proba(model, pk_list, pc_list, max(0, n - tail_k), n)
        else:
            p_by_n[n] = prefix_bag_proba(model, pk_list, pc_list, n)

    def _p_at(npz_i: int) -> float:
        # Continuous window position (stride = trace_stride); linearly
        # interpolate between the two surrounding window predictions for a
        # smooth trace (identical predictions, just without stair-steps).
        if not p_by_n:
            return 0.0
        x = npz_i / float(trace_stride)
        lo_n = max(warmup_windows, min(int(np.floor(x)), M))
        hi_n = max(warmup_windows, min(lo_n + 1, M))
        if lo_n == hi_n:
            return p_by_n[lo_n]
        frac = float(x - np.floor(x))
        return (1.0 - frac) * p_by_n[lo_n] + frac * p_by_n[hi_n]

    p_history = []
    for k in range(n_video_frames):
        t_sec = k / out_fps
        progress = k / max(n_video_frames - 1, 1)

        # The trace traverses the full cached episode over the act, and the
        # source shot plays once over the same span, so video and P stay aligned
        # by fraction even though the cache (50Hz) and the mp4 sit on different
        # clocks for this dataset.
        npz_idx = offset + int(round(progress * (len(kin_seq) - 1 - offset)))
        npz_idx = max(0, min(npz_idx, len(kin_seq) - 1))
        p = _p_at(npz_idx)
        p_history.append(p)

        frame = shot[min(int(round(progress * (n_shot - 1))), n_shot - 1)]

        # The act-level label is authoritative; reveal it after the midpoint.
        if is_failure_act:
            gt_for_display = 1 if t_sec >= duration / 2 else 0
        else:
            gt_for_display = 0

        canvas = render_canvas(
            frame, t_sec=t_sec, dur_sec=duration,
            p=p, threshold=threshold, p_history=p_history,
            gt_label=gt_for_display,
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
        default="results/video_overlay_demo/acts_v10l_droid_test.json")
    ap.add_argument("--model-dir", default="results/multimodal_indist_v10l_seed42")
    ap.add_argument("--act-idx", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--out-dir", default="results/video_overlay_demo")
    ap.add_argument("--reassemble-h5-root", default="./data/reassemble",
                    help="Directory containing per-recording .h5 files for Reassemble. "
                         "Each sequence resolves to <root>/<date>.h5.")
    ap.add_argument("--droid-parquet-root", default="./data/droid",
                    help="Directory containing DROID parquet shards. Resolves to "
                         "<root>/chunk-000/<file_stem>.parquet.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading model")
    model_bundle = load_model(args.model_dir)

    print(f"\n[2/3] loading acts config {args.acts_config}")
    with open(args.acts_config) as f:
        acts = json.load(f)

    # All acts run in passive-monitoring mode: the classifier observes, the
    # source video plays uninterrupted regardless of the per-frame P(failure).
    part_labels = ["Passive Monitoring"] * len(acts)

    print(f"\n[3/3] rendering")
    indices = range(len(acts)) if args.act_idx < 0 else [args.act_idx]
    for i in indices:
        act = acts[i]
        out_path = os.path.join(args.out_dir, f"act_{i+1}_{act['label']}.mp4")
        process_act(act, model_bundle, threshold=args.threshold,
                    out_path=out_path,
                    part_label=part_labels[i],
                    reassemble_h5_root=args.reassemble_h5_root,
                    droid_parquet_root=args.droid_parquet_root)
    print("\ndone.")


if __name__ == "__main__":
    sys.exit(main())
