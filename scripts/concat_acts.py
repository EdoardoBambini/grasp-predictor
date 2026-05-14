"""Concatenate the per-act MP4s produced by ``video_overlay_demo.py`` into a
single deliverable, with intro, per-act title cards, and outro."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_FPS = 30
TARGET_W = 1280
TARGET_H = 720
TITLE_DURATION_S = 2.5
INTRO_DURATION_S = 3.0
OUTRO_DURATION_S = 3.5

# Default font paths (Windows). On Linux/macOS the renderer falls back to
# PIL's bundled default font; override via environment variables if you have
# specific fonts you'd like to use (e.g. DejaVu on Linux).
FONT_REG = os.environ.get("FONT_REG", "C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = os.environ.get("FONT_BOLD", "C:/Windows/Fonts/segoeuib.ttf")
FONT_LIGHT = os.environ.get("FONT_LIGHT", "C:/Windows/Fonts/segoeuil.ttf")
_font_cache: dict = {}


def _font(path: str, size: int):
    key = (path, size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(path, size)
        except (OSError, IOError):
            # Path doesn't exist (e.g. Linux/macOS without Windows fonts).
            # Fall back to PIL's bundled default so the renderer still runs.
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def render_card(lines: list[tuple[str, int, str, tuple[int, int, int], int]],
                duration_s: float) -> list[np.ndarray]:
    """Each line: (text, font_size, font_path, rgb_color, gap_after)."""
    n = int(round(duration_s * TARGET_FPS))
    img = Image.new("RGB", (TARGET_W, TARGET_H), (12, 14, 22))
    draw = ImageDraw.Draw(img)

    draw.line([60, 60, TARGET_W - 60, 60], fill=(40, 50, 80), width=1)
    draw.line([60, TARGET_H - 60, TARGET_W - 60, TARGET_H - 60],
              fill=(40, 50, 80), width=1)

    block_h = 0
    rendered = []
    for entry in lines:
        if len(entry) == 5:
            text, sz, path, color, gap = entry
        else:
            text, sz, path, color = entry
            gap = 26
        font = _font(path, sz)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        rendered.append((text, font, tw, th, color, gap))
        block_h += th + gap

    y = (TARGET_H - block_h) // 2
    for text, font, tw, th, color, gap in rendered:
        x = (TARGET_W - tw) // 2
        draw.text((x, y), text, font=font, fill=color)
        y += th + gap

    draw.text((48, TARGET_H - 50),
              "Mosaico SDK   |   grasp integrity prediction",
              font=_font(FONT_REG, 14), fill=(160, 175, 200))
    draw.text((48, TARGET_H - 30),
              "LateFusionLSTM   |   108k params   |   AUC 0.685   |   thr 0.806",
              font=_font(FONT_REG, 12), fill=(140, 155, 180))

    base = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return [base.copy() for _ in range(n)]


def fade_in_out(frames: list[np.ndarray], fade_n: int = 6) -> list[np.ndarray]:
    fade_n = min(fade_n, len(frames) // 4)
    if fade_n <= 0:
        return frames
    out = []
    for i, f in enumerate(frames):
        if i < fade_n:
            alpha = (i + 1) / fade_n
            out.append((f.astype(np.float32) * alpha).astype(np.uint8))
        elif i >= len(frames) - fade_n:
            alpha = (len(frames) - i) / fade_n
            out.append((f.astype(np.float32) * alpha).astype(np.uint8))
        else:
            out.append(f)
    return out


def read_mp4_resampled(path: Path, native_fps: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (TARGET_H, TARGET_W):
            f = cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        frames.append(f)
    cap.release()

    if native_fps == TARGET_FPS:
        return frames
    repeat = max(1, TARGET_FPS // native_fps)
    out = []
    for f in frames:
        for _ in range(repeat):
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/video_overlay_demo")
    ap.add_argument("--acts-config",
        default="results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json")
    ap.add_argument("--out", default="results/video_overlay_demo/module_c_deliverable.mp4")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    with open(args.acts_config) as f:
        acts = json.load(f)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, TARGET_FPS, (TARGET_W, TARGET_H))
    if not vw.isOpened():
        raise RuntimeError(f"could not open {args.out}")

    print(f"[1/3] intro card")
    intro = render_card([
        ("Grasp Integrity Predictor", 64, FONT_LIGHT, (240, 240, 250), 56),
        ("Per-frame failure probability overlaid on held-out test sequences",
         26, FONT_REG, (180, 200, 230), 44),
        ("4 acts   |   passive monitoring",
         18, FONT_REG, (140, 170, 200), 0),
    ], INTRO_DURATION_S)
    for f in fade_in_out(intro, 8):
        vw.write(f)

    print(f"[2/3] writing acts")
    part_titles = ["Passive Monitoring"] * len(acts)

    for i, act in enumerate(acts):
        seq_path = act["sequence"]
        is_droid = "droid/" in seq_path.replace("\\", "/")
        ds = "DROID" if is_droid else "Reassemble"
        seq_name = Path(seq_path).stem
        act_mp4_for_probe = in_dir / f"act_{i+1}_{act['label']}.mp4"
        cap_p = cv2.VideoCapture(str(act_mp4_for_probe))
        native_fps = int(round(cap_p.get(cv2.CAP_PROP_FPS) or 30))
        cap_p.release()

        outcome = act["label"]
        outcome_color = ((90, 220, 110) if outcome == "SUCCESS"
                         else (240, 90, 90))

        print(f"  act {i+1}: {outcome} {ds} {seq_name}")
        title = render_card([
            (f"ACT {i+1}", 96, FONT_LIGHT, (240, 240, 250), 56),
            (part_titles[i], 24, FONT_BOLD, (200, 215, 235), 48),
            (f"{outcome}   |   {ds}", 36, FONT_BOLD, outcome_color, 44),
            (seq_name, 18, FONT_REG, (160, 175, 200), 18),
            (f"offset {act['offset']}   |   duration {act['duration']:.0f} s   @ 50 Hz",
             14, FONT_REG, (140, 155, 180), 0),
        ], TITLE_DURATION_S)
        for f in fade_in_out(title, 8):
            vw.write(f)

        act_mp4 = in_dir / f"act_{i+1}_{outcome}.mp4"
        for f in read_mp4_resampled(act_mp4, native_fps):
            vw.write(f)

    print(f"[3/3] outro card")
    outro = render_card([
        ("Built on Mosaico", 64, FONT_LIGHT, (240, 240, 250), 50),
        ("Cross-format ingest   |   sync 50 Hz   |   canonical schema",
         24, FONT_REG, (180, 200, 230), 44),
        ("HDF5  +  Parquet  +  TFRecord   →   single LateFusionLSTM",
         20, FONT_REG, (140, 170, 200), 0),
    ], OUTRO_DURATION_S)
    for f in fade_in_out(outro, 12):
        vw.write(f)

    vw.release()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
