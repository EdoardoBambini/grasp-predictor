#!/usr/bin/env bash
# End-to-end pipeline: ingest 3 datasets through Mosaico, precompute the CNN
# cache, train the LateFusionLSTM, render the deliverable video.
#
# Total runtime: roughly 3 hours wall-clock on a Ryzen 7 CPU.
#
# Requirements:
#   - .venv set up via scripts/setup.sh
#   - Mosaico daemon running:
#       docker compose -f docker/compose-training.yml up -d
#   - Raw datasets on disk (Reassemble .h5, DROID parquet, Fractal RT-1 TFRecord)
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Environment variables (override defaults):
#   PYBIN              Python executable     (default: python)
#   REASSEMBLE_ROOT    Reassemble dataset    (default: ./data/reassemble)
#   DROID_ROOT         DROID dataset         (default: ./data/droid)
#   FRACTAL_ROOT       Fractal RT-1 dataset  (default: ./data/fractal_rt1)
set -euo pipefail

PYBIN="${PYBIN:-python}"
REASSEMBLE_ROOT="${REASSEMBLE_ROOT:-./data/reassemble}"
DROID_ROOT="${DROID_ROOT:-./data/droid}"
FRACTAL_ROOT="${FRACTAL_ROOT:-./data/fractal_rt1}"

# Reuse the same dataset roots for the renderer unless overridden separately
REASSEMBLE_H5_ROOT="${REASSEMBLE_H5_ROOT:-$REASSEMBLE_ROOT}"
DROID_PARQUET_ROOT="${DROID_PARQUET_ROOT:-$DROID_ROOT}"

echo "===== [1/4] Ingesting 3 datasets into Mosaico ====="
"$PYBIN" run_ingestion_parallel.py --workers 4 --dataset-root "$REASSEMBLE_ROOT"
"$PYBIN" scripts/ingest_droid_parallel.py --data-root "$DROID_ROOT"
"$PYBIN" scripts/ingest_generic.py --data-root "$FRACTAL_ROOT"

echo ""
echo "===== [1b/4] Sanity-checking the catalog ====="
"$PYBIN" scripts/check_label_topic.py
"$PYBIN" scripts/check_catalog_images.py

echo ""
echo "===== [2/4] Precomputing the CNN feature cache (~3 h, idempotent) ====="
"$PYBIN" scripts/precompute_cnn_features.py

echo ""
echo "===== [3/4] Training the released LateFusionLSTM (~6 min on CPU) ====="
bash scripts/run_v9.sh released

echo ""
echo "===== [4/4] Rendering the deliverable video ====="
REASSEMBLE_H5_ROOT="$REASSEMBLE_H5_ROOT" \
DROID_PARQUET_ROOT="$DROID_PARQUET_ROOT" \
  bash scripts/render_video.sh

echo ""
echo "Done. Deliverable at: results/video_overlay_demo/module_c_deliverable.mp4"
