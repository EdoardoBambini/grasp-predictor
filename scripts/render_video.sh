#!/usr/bin/env bash
# Render the per-act P(failure) overlays and concatenate them into the final
# deliverable mp4. Uses the released model + scaler shipped in
# results/multimodal_indist_v9_sharp/, so it does NOT need the Mosaico daemon
# or full dataset re-ingestion.
#
# Requirements:
#   - .venv set up via scripts/setup.sh (or another Python env with the deps)
#   - Reassemble source .h5 files reachable via --reassemble-h5-root
#   - DROID parquet shards reachable via --droid-parquet-root
#   - CNN feature cache present under results/cnn_cache_spatial/ for the
#     sequences referenced in the acts config
#
# Usage:
#   bash scripts/render_video.sh
#
# Environment variables (override defaults):
#   PYBIN                Python executable                 (default: python)
#   ACTS_CONFIG          Path to acts JSON                 (default: results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json)
#   MODEL_DIR            Directory with best_model.pt etc. (default: results/multimodal_indist_v9_sharp)
#   THRESHOLD            P threshold for alarm border      (default: 0.806)
#   OUT_DIR              Output directory                  (default: results/video_overlay_demo)
#   REASSEMBLE_H5_ROOT   Reassemble .h5 root               (default: ./data/reassemble)
#   DROID_PARQUET_ROOT   DROID parquet root                (default: ./data/droid)
set -euo pipefail

PYBIN="${PYBIN:-python}"
ACTS_CONFIG="${ACTS_CONFIG:-results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json}"
MODEL_DIR="${MODEL_DIR:-results/multimodal_indist_v9_sharp}"
THRESHOLD="${THRESHOLD:-0.806}"
OUT_DIR="${OUT_DIR:-results/video_overlay_demo}"
REASSEMBLE_H5_ROOT="${REASSEMBLE_H5_ROOT:-./data/reassemble}"
DROID_PARQUET_ROOT="${DROID_PARQUET_ROOT:-./data/droid}"

echo "[1/2] Rendering per-act overlays"
"$PYBIN" -m scripts.video_overlay_demo \
  --acts-config "$ACTS_CONFIG" \
  --model-dir "$MODEL_DIR" \
  --threshold "$THRESHOLD" \
  --out-dir "$OUT_DIR" \
  --reassemble-h5-root "$REASSEMBLE_H5_ROOT" \
  --droid-parquet-root "$DROID_PARQUET_ROOT"

echo ""
echo "[2/2] Concatenating into deliverable"
"$PYBIN" -m scripts.concat_acts \
  --in-dir "$OUT_DIR" \
  --acts-config "$ACTS_CONFIG" \
  --out "$OUT_DIR/module_c_deliverable.mp4"

echo ""
echo "Done. Output: $OUT_DIR/module_c_deliverable.mp4"
