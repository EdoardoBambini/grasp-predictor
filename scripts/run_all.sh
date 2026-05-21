#!/usr/bin/env bash
# One-shot reproduction of the released LateFusionLSTMMIL pipeline on the
# Mosaico-curated DROID + Fractal RT-1 pools.
#
# Pipeline order:
#   [1] precompute frozen CNN feature cache (per dataset)
#   [2] build per-dataset success/failure pools via Mosaico queries
#   [3] train the released configuration for one or more seeds
#   [4] regenerate the figures used in the writeup
# precompute runs before the pool build on purpose: the Fractal pool reads its
# per-episode label from the .npz cache produced during precompute.
#
# Prerequisites:
#   - DROID and Fractal RT-1 already ingested into a local Mosaico instance
#     (see README; ingestion is handled by the mosaico-alchemy plugins).
#   - mosaicod reachable at MOSAICO_HOST:MOSAICO_PORT (default 127.0.0.1:6726).
#   - Dependencies installed: pip install mosaicolabs && pip install -r requirements.txt
#   - Python 3.13+.
#
# WARNING: step [3] overwrites results/multimodal_indist_v10l_seed<seed>/, including
# the shipped checkpoint and results.json. Those artifacts are tracked in git; use
# git to restore them if you only meant to re-run part of the pipeline.
#
# Usage:
#   bash scripts/run_all.sh                                  # full pipeline, seeds "7 42 123"
#   SEEDS="7" bash scripts/run_all.sh                        # single seed
#   WITH_DEMO=1 DROID_PARQUET_ROOT=/path/to/droid/data bash scripts/run_all.sh
#
# Tunables (environment variables):
#   PYBIN, MOSAICO_HOST, MOSAICO_PORT, SEEDS, POOLS_JSON,
#   DROID_CACHE, FRACTAL_CACHE, WITH_DEMO, MODEL_DIR, DROID_PARQUET_ROOT

set -euo pipefail
cd "$(dirname "$0")/.."   # always run from the repository root

PYBIN="${PYBIN:-python}"
MOSAICO_HOST="${MOSAICO_HOST:-127.0.0.1}"
MOSAICO_PORT="${MOSAICO_PORT:-6726}"
SEEDS="${SEEDS:-7 42 123}"
POOLS_JSON="${POOLS_JSON:-results/pools_via_mosaico.json}"
DROID_CACHE="${DROID_CACHE:-results/cnn_cache_spatial}"
FRACTAL_CACHE="${FRACTAL_CACHE:-results/cnn_cache_v10k_fractal}"
WITH_DEMO="${WITH_DEMO:-0}"
MODEL_DIR="${MODEL_DIR:-results/multimodal_indist_v10l_seed7}"
DROID_PARQUET_ROOT="${DROID_PARQUET_ROOT:-./data/droid}"

echo "==> [1/4] Precompute frozen MobileNetV3 CNN feature cache (long; skips existing .npz)"
"$PYBIN" scripts/precompute_cnn_features.py \
    --host "$MOSAICO_HOST" --port "$MOSAICO_PORT" \
    --datasets droid --cache-dir "$DROID_CACHE"
"$PYBIN" scripts/precompute_cnn_features.py \
    --host "$MOSAICO_HOST" --port "$MOSAICO_PORT" \
    --datasets fractal_rt1 --cache-dir "$FRACTAL_CACHE"

echo "==> [2/4] Build per-dataset success/failure pools via Mosaico queries"
if [[ -f "$POOLS_JSON" ]]; then
    echo "    pools already present at $POOLS_JSON (delete the file to rebuild)"
else
    "$PYBIN" scripts/build_balanced_pools_via_mosaico.py \
        --host "$MOSAICO_HOST" --port "$MOSAICO_PORT" \
        --datasets droid,fractal_rt1 \
        --fractal-cache-dir "$FRACTAL_CACHE" \
        --out "$POOLS_JSON"
fi

echo "==> [3/4] Train the released configuration for seeds: $SEEDS"
for s in $SEEDS; do
    echo "    -- training seed $s"
    PYBIN="$PYBIN" POOLS_JSON="$POOLS_JSON" bash scripts/train.sh "$s"
done

echo "==> [4/4] Regenerate figures from the released seed (seed 7)"
"$PYBIN" scripts/make_figures.py \
    --run-dir results/multimodal_indist_v10l_seed7 \
    --out-dir docs/figures

if [[ "$WITH_DEMO" == "1" ]]; then
    echo "==> [demo] Render per-window overlay on held-out DROID episodes, then stitch"
    "$PYBIN" scripts/video_overlay_demo.py \
        --model-dir "$MODEL_DIR" \
        --acts-config results/video_overlay_demo/acts_v10l_droid_test.json \
        --droid-parquet-root "$DROID_PARQUET_ROOT"
    "$PYBIN" scripts/concat_acts.py \
        --in-dir results/video_overlay_demo \
        --acts-config results/video_overlay_demo/acts_v10l_droid_test.json \
        --out results/video_overlay_demo/module_c_deliverable.mp4
fi

echo "==> Done."
