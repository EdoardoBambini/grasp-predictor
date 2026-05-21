#!/usr/bin/env bash
# Train the LateFusionLSTMMIL grasp-failure classifier jointly on DROID and
# Fractal RT-1 through the Mosaico-curated balanced pools.
#
# MIL bag-level model: bag = sequence, instance = 50-frame window. Pre-pool
# natural-language concat (Universal Sentence Encoder 512-D projected to 64-D,
# DROID routed through a learnable null token). Gated attention pool (Ilse 2018).
# BCE loss, Adam, ReduceLROnPlateau, SWA from epoch 8, PCGrad multi-task
# gradient surgery. DROID failure bags trimmed to their last 30% of windows.
#
# Train split balanced 65/35 success/failure per dataset; validation and test
# stay at the natural prior so the reported metrics reflect production.
#
# Usage:
#   bash scripts/train.sh [seed]      # default seed 42

set -euo pipefail

PYBIN="${PYBIN:-python}"
SEED="${1:-42}"
POOLS_JSON="${POOLS_JSON:-results/pools_via_mosaico.json}"

if [[ ! -f "$POOLS_JSON" ]]; then
    echo "Pools JSON not found at $POOLS_JSON -- generating now..." >&2
    "$PYBIN" scripts/build_balanced_pools_via_mosaico.py --out "$POOLS_JSON" \
        --datasets droid,fractal_rt1
fi

OUTDIR="results/multimodal_indist_v10l_seed${SEED}"

"$PYBIN" run_training_cached.py \
  --train-datasets droid,fractal_rt1 \
  --test-datasets  droid,fractal_rt1 \
  --cache-dir results/cnn_cache_spatial \
  --fractal-cache-dir results/cnn_cache_v10k_fractal \
  --cnn-dim 360 \
  --epochs 30 --batch-size 16 \
  --lr 1e-3 --weight-decay 1e-3 \
  --dropout 0.35 --hidden 256 \
  --stride 50 \
  --use-scheduler --num-workers 0 \
  --kin-noise-std 0.02 \
  --weighted-sampler \
  --ipca-components 0 \
  --swa-start-epoch 8 \
  --loss bce --no-pos-weight \
  --label-smoothing 0.0 \
  --mixup-alpha 0.0 \
  --attn-pool \
  --mil-attention \
  --mil-pool-mode gated \
  --lstm-direction uni \
  --droid-tail-fraction 0.30 \
  --fractal-tail-fraction 1.0 \
  --use-nl-concat \
  --nl-concat-dim 64 \
  --use-pcgrad \
  --balanced-train \
  --pools-json "$POOLS_JSON" \
  --success-ratio 0.65 \
  --seed "$SEED" \
  --output-dir "$OUTDIR" \
  2>&1 | tee "${OUTDIR}.log"
