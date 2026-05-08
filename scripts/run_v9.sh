#!/usr/bin/env bash
# Wrapper around run_training_cached.py: `released` reproduces the released run,
# the other targets are ablations. The CNN cache must already exist.

set -euo pipefail

PYBIN="${PYBIN:-python}"

COMMON=(
  --train-datasets reassemble,droid,fractal_rt1
  --test-datasets  reassemble,droid,fractal_rt1
  --cache-dir results/cnn_cache_spatial
  --cnn-dim 360
  --epochs 30 --batch-size 64
  --lr 1e-3 --weight-decay 1e-3
  --dropout 0.35 --hidden 256
  --stride 50
  --use-scheduler --num-workers 0
  --kin-noise-std 0.02
  --weighted-sampler
  --ipca-components 64
  --late-fusion
  --swa-start-epoch 8
)

case "${1:-released}" in
  released)
    # Released configuration: BCE with auto pos_weight, no label smoothing,
    # late fusion + attention pool over time.
    "$PYBIN" run_training_cached.py "${COMMON[@]}" \
      --loss bce \
      --label-smoothing 0.0 \
      --mixup-alpha 0.0 \
      --attn-pool \
      --output-dir results/multimodal_indist_v9_sharp \
      2>&1 | tee results/multimodal_indist_v9_sharp.log
    ;;
  ablation_no_attn)
    # Same recipe without the attention pool over time.
    "$PYBIN" run_training_cached.py "${COMMON[@]}" \
      --loss bce \
      --label-smoothing 0.0 \
      --mixup-alpha 0.0 \
      --output-dir results/multimodal_indist_no_attn \
      2>&1 | tee results/multimodal_indist_no_attn.log
    ;;
  ablation_focal)
    # Focal loss alternative (pre-released configuration with label smoothing).
    "$PYBIN" run_training_cached.py "${COMMON[@]}" \
      --no-pos-weight \
      --label-smoothing 0.05 \
      --mixup-alpha 0.0 \
      --attn-pool \
      --output-dir results/multimodal_indist_focal \
      2>&1 | tee results/multimodal_indist_focal.log
    ;;
  alt_seed)
    # Same recipe with an alternative seed for ensemble diversity.
    "$PYBIN" run_training_cached.py "${COMMON[@]}" \
      --loss bce \
      --label-smoothing 0.0 \
      --mixup-alpha 0.0 \
      --attn-pool \
      --seed 123 \
      --output-dir results/multimodal_indist_seed123 \
      2>&1 | tee results/multimodal_indist_seed123.log
    ;;
  *)
    echo "Usage: $0 [released|ablation_no_attn|ablation_focal|alt_seed]"
    echo "  released         (default) reproduce the released model"
    echo "  ablation_no_attn drop attention pool, keep BCE recipe"
    echo "  ablation_focal   focal loss with label smoothing 0.05"
    echo "  alt_seed         released recipe with seed=123 (ensemble diversity)"
    exit 1
    ;;
esac
