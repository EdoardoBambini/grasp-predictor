"""
Pre-compute frozen CNN features (MobileNetV3 Small, ImageNet weights) for
every sequence in the Mosaico catalog that has an RGB image topic. One .npz
per sequence containing {kin: (n,15), cnn: (n,cnn_dim) fp16, label: (n,)}
aligned to the 50 Hz dense timeline.

Default backbone: layer 6 of MobileNetV3 Small (40 channels) followed by
AdaptiveMaxPool2d(3), producing 360-dim spatial features that retain
geometric structure (where in the frame the gripper and object are), not
only semantic content. A GAP-pooled 576-dim variant is available via
--intermediate-layer -1 --spatial-pool 1.

Throughput on a single CPU core is image-decode bound; a multi-thousand
sequence corpus typically completes in a few hours one shot.

Idempotent: skips sequences whose .npz already exists with size > 0; safe
to resume by re-running with the same arguments.

Usage:
    # default 360-dim spatial features, all 3 datasets
    py -3.13 scripts/precompute_cnn_features.py

    # smoke test on 2 reassemble sequences
    py -3.13 scripts/precompute_cnn_features.py --datasets reassemble --cap-per-dataset 2

    # legacy 576-dim GAP variant
    py -3.13 scripts/precompute_cnn_features.py \
        --intermediate-layer -1 --spatial-pool 1 \
        --cache-dir results/cnn_cache
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data import feature_mapper, label_adapters
from data.cross_dataset_ingestor import CrossDatasetIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precache")


def build_cnn_backbone(name: str = "mobilenet_v3_small",
                       intermediate_layer: int = -1,
                       spatial_pool: int = 1) -> tuple[nn.Module, int, tuple[int, int], str]:
    """Build a frozen visual encoder. Returns (model, output_dim, input_hw, normalize_mode).

    normalize_mode is one of:
      - "imagenet"  : x = (x/255 - mean) / std (torchvision standard)
      - "r3m"       : x = x (uint8 -> float32, NO scaling, NO mean/std subtract).
                      R3M expects raw [0, 255] input; normalization is internal.

    Supported backbones (CPU-friendly):
      - mobilenet_v3_small   : ImageNet, 112x112 input. Layer-6 + MaxPool 3x3
                               -> 360-dim (recommended for grasp tasks).
      - resnet18             : ImageNet, 512-dim, 224x224 input.
      - dinov2_vits14        : DINOv2 self-supervised, 384-dim, 224x224 input. SLOW on CPU.
      - r3m_resnet18         : Meta R3M ResNet18 pretrained on Ego4D manipulation
                               videos, 512-dim, 224x224 input. ~30ms/frame on CPU.
                               Recommended manipulation-specific upgrade vs ImageNet.
      - r3m_resnet50         : R3M ResNet50, 2048-dim, 224x224. ~100ms/frame on CPU
                               (~3x slower than ResNet18). Use only if budget allows.
    """
    name = name.lower()
    normalize_mode = "imagenet"  # default for ImageNet-pretrained backbones

    if name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if intermediate_layer >= 0:
            tap = nn.Sequential(*list(backbone.features.children())[:intermediate_layer + 1])
            CHANNELS_PER_LAYER = {0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40, 6: 40,
                                  7: 48, 8: 48, 9: 48, 10: 96, 11: 96, 12: 576}
            ch = CHANNELS_PER_LAYER.get(intermediate_layer)
            if ch is None:
                raise ValueError(f"Unknown channel count for layer {intermediate_layer}")
            cnn = nn.Sequential(tap, nn.AdaptiveMaxPool2d(spatial_pool), nn.Flatten(1))
            out_dim = ch * spatial_pool * spatial_pool
        else:
            cnn = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            out_dim = 576
        hw = (112, 112)
    elif name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        cnn = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten(1))
        out_dim, hw = 512, (224, 224)
    elif name == "dinov2_vits14":
        cnn = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        out_dim, hw = 384, (224, 224)
    elif name in ("r3m_resnet18", "r3m_resnet50", "r3m_resnet34"):
        # Meta R3M: pretrained on Ego4D manipulation videos. Best generic
        # pretrained backbone for robot manipulation tasks. Weights cached in
        # ~/.cache/r3m/ after first download (~50-200 MB depending on backbone).
        try:
            from r3m import load_r3m
        except ImportError as e:
            raise ImportError(
                "R3M not installed. Run: pip install r3m\n"
                "(downloads weights from S3 on first use, ~50-200MB)"
            ) from e
        variant = name.replace("r3m_", "")  # "resnet18" / "resnet34" / "resnet50"
        log.info("Loading R3M %s (downloads weights on first use)...", variant)
        r3m_model = load_r3m(variant)
        # R3M wraps a ResNet behind a forward returning a plain (B, dim) tensor.
        # The inner R3M model already does internal normalization (assumes [0,255] input).
        cnn = r3m_model
        out_dim = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}[variant]
        hw = (224, 224)
        normalize_mode = "r3m"  # raw uint8 -> float32, NO scaling
    else:
        raise ValueError(f"Unknown backbone: {name}")

    for p in cnn.parameters():
        p.requires_grad = False
    cnn.eval()
    return cnn, out_dim, hw, normalize_mode


def normalize_imgs(arr_uint8: np.ndarray, mode: str = "imagenet") -> np.ndarray:
    """(N, 3, H, W) uint8 -> (N, 3, H, W) float32 normalized per backbone convention.

    mode='imagenet': torchvision standard, x = (x/255 - mean) / std
    mode='r3m':      raw [0, 255] float32, R3M does its own normalization internally
    """
    if mode == "r3m":
        return arr_uint8.astype(np.float32)  # NO scaling, R3M wants [0, 255]
    mean = np.asarray(feature_mapper.IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(feature_mapper.IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    x = arr_uint8.astype(np.float32) / 255.0
    return (x - mean) / std


@torch.inference_mode()
def cnn_forward_batch(cnn: nn.Module, imgs_norm: np.ndarray, out_dim: int,
                      batch_size: int) -> np.ndarray:
    """Forward (N, 3, H, W) float32 through cnn in batches; returns (N, out_dim) float32."""
    n = imgs_norm.shape[0]
    out = np.empty((n, out_dim), dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = torch.from_numpy(imgs_norm[i : i + batch_size])
        feats = cnn(chunk)
        # DINOv2 forward returns the CLS token by default (B, 384). Other backbones
        # return (B, out_dim) after the global pool. Both are 2D, no special handling.
        out[i : i + chunk.shape[0]] = feats.cpu().numpy().astype(np.float32)
    return out


def cache_path(cache_dir: str, dsid: str, seq_name: str) -> str:
    safe = seq_name.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, dsid, f"{safe}.npz")


def process_one(
    ing: CrossDatasetIngestor,
    cnn: nn.Module,
    cnn_dim: int,
    cnn_hw: tuple,
    dsid: str,
    seq_name: str,
    out_path: str,
    batch_size: int,
    normalize_mode: str = "imagenet",
) -> Optional[dict]:
    """Pull one sequence, decode JPEGs, run CNN, save .npz. Returns stats or None."""
    df = ing.process_sequence(dsid, seq_name, include_images=True)
    if df is None or df.empty:
        return None

    img_topic = feature_mapper.IMAGE_TOPIC_PER_DATASET.get(dsid)
    if img_topic is None:
        log.warning("[%s] %s: no image topic configured for dsid", dsid, seq_name)
        return None
    img_col = feature_mapper.image_column_name(img_topic)
    if img_col not in df.columns:
        log.warning("[%s] %s: image column %s missing", dsid, seq_name, img_col)
        return None

    n = len(df)
    # Decode JPEGs frame-by-frame to (n, 3, H, W) uint8 at backbone-specific resolution
    H, W = cnn_hw
    frames = np.empty((n, 3, H, W), dtype=np.uint8)
    bytes_series = df[img_col]
    for i, b in enumerate(bytes_series.values):
        frames[i] = feature_mapper.decode_jpeg_to_chw(b, (H, W))

    imgs_norm = normalize_imgs(frames, mode=normalize_mode)
    cnn_feats = cnn_forward_batch(cnn, imgs_norm, cnn_dim, batch_size)  # (n, cnn_dim) float32

    kin = df[list(feature_mapper.EXTENDED_FEATURES)].to_numpy(dtype=np.float32)
    label = df[label_adapters.LABEL_COL].to_numpy(dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        kin=kin,
        cnn=cnn_feats.astype(np.float16),
        label=label,
    )
    return {"n": n, "pos_ratio": float(label.mean()) if n else 0.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="reassemble,droid,fractal_rt1",
                   help="comma-separated dsids (default: reassemble,droid,fractal_rt1)")
    p.add_argument("--cap-per-dataset", type=int, default=None,
                   help="optional cap on sequences per dataset (None = full)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="CNN forward batch size (default 64)")
    p.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results", "cnn_cache_spatial"),
                   help="output dir for the .npz cache (default: cnn_cache_spatial)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6726)
    p.add_argument("--force", action="store_true",
                   help="rebuild even if .npz exists")
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "resnet18", "dinov2_vits14",
                            "r3m_resnet18", "r3m_resnet34", "r3m_resnet50"],
                   help="visual encoder backbone")
    p.add_argument("--intermediate-layer", type=int, default=6,
                   help="mobilenet_v3_small features[i] tap (default 6 = 40 channels). "
                        "-1 = full features + GAP (legacy 576-dim).")
    p.add_argument("--spatial-pool", type=int, default=3,
                   help="AdaptiveMaxPool2d size before flatten (default 3 = 3x3 spatial; 1 = global)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datasets: List[str] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.cache_dir, exist_ok=True)

    log.info("=== CNN feature pre-cache ===")
    log.info("datasets=%s cap=%s batch=%d cache_dir=%s backbone=%s layer=%d pool=%d",
             datasets, args.cap_per_dataset, args.batch_size, args.cache_dir,
             args.backbone, args.intermediate_layer, args.spatial_pool)
    log.info("Loading backbone %s...", args.backbone)
    cnn, cnn_dim, cnn_hw, normalize_mode = build_cnn_backbone(args.backbone,
                                              intermediate_layer=args.intermediate_layer,
                                              spatial_pool=args.spatial_pool)
    log.info("Backbone ready (%d-dim, input %dx%d, normalize=%s, frozen).",
             cnn_dim, cnn_hw[0], cnn_hw[1], normalize_mode)

    grand_t0 = time.time()
    grand_ok = grand_skip = grand_fail = 0
    grand_frames = 0

    with CrossDatasetIngestor(
        host=args.host, port=args.port,
        datasets=datasets, cap_per_dataset=None,  # cap manually below
    ) as ing:
        for dsid in datasets:
            img_topic = feature_mapper.IMAGE_TOPIC_PER_DATASET.get(dsid)
            if img_topic is None:
                log.info("[%s] no image topic configured, skipping", dsid)
                continue
            names = ing._query_names(dsid)
            if args.cap_per_dataset is not None:
                names = names[: args.cap_per_dataset]
            log.info("[%s] %d sequences to process", dsid, len(names))

            ok = skip = fail = 0
            ds_t0 = time.time()
            for i, name in enumerate(names, 1):
                out_path = cache_path(args.cache_dir, dsid, name)
                if not args.force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    skip += 1
                    if i % 50 == 0:
                        log.info("[%s] %d/%d skip=%d ok=%d fail=%d",
                                 dsid, i, len(names), skip, ok, fail)
                    continue
                t0 = time.time()
                try:
                    stats = process_one(ing, cnn, cnn_dim, cnn_hw, dsid, name, out_path,
                                          args.batch_size, normalize_mode=normalize_mode)
                except Exception as e:
                    log.exception("[%s] %s failed: %s", dsid, name, e)
                    fail += 1
                    continue
                if stats is None:
                    fail += 1
                    continue
                dt = time.time() - t0
                ok += 1
                grand_frames += stats["n"]
                if i <= 3 or i % 25 == 0 or i == len(names):
                    log.info("[%s] %d/%d %s: n=%d pos=%.2f %.1fs",
                             dsid, i, len(names), name, stats["n"], stats["pos_ratio"], dt)

            log.info("[%s] done in %.0fs | ok=%d skip=%d fail=%d",
                     dsid, time.time() - ds_t0, ok, skip, fail)
            grand_ok += ok
            grand_skip += skip
            grand_fail += fail

    elapsed = time.time() - grand_t0
    log.info("=== Pre-cache complete in %.0fs (%.2fh) ===", elapsed, elapsed / 3600)
    log.info("ok=%d skip=%d fail=%d total_frames=%d", grand_ok, grand_skip, grand_fail, grand_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
