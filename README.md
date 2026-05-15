# Cross-Format Grasp Integrity Predictor on Mosaico

[![CI](https://github.com/EdoardoBambini/grasp-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/EdoardoBambini/grasp-predictor/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Built on Mosaico ≥ 0.4](https://img.shields.io/badge/built%20on-Mosaico%20%E2%89%A5%200.4-1f77b4.svg)](https://github.com/mosaico-labs/mosaico) [![PyTorch ≥ 2.2](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A cross-format grasp failure classifier trained on three heterogeneous manipulation datasets (Reassemble, DROID, Fractal RT-1). The pipeline uses [Mosaico](https://github.com/mosaico-labs/mosaico) as the data platform, so the application code never reads `.h5`, `.tfrecord` or ROS bag files directly.

A single `LateFusionLSTM` consumes 15 canonical kinematic features and a 64-dim visual descriptor per timestep (MobileNetV3 Small layer 6, post-IPCA) over 50-frame sliding windows (1 s at 50 Hz), and emits a calibrated failure probability per window.

Full technical writeup in [`BLOG_POST.md`](BLOG_POST.md).

## Quickstart

```bash
git clone https://github.com/EdoardoBambini/grasp-predictor.git
cd grasp-predictor

# 1. Create venv + install Python dependencies
bash scripts/setup.sh
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# 2. Render the deliverable video from the released model
#    Needs Reassemble .h5 files + DROID parquet shards locally:
REASSEMBLE_H5_ROOT="/path/to/reassemble" \
DROID_PARQUET_ROOT="/path/to/droid" \
  bash scripts/render_video.sh
```

Output lands at `results/video_overlay_demo/module_c_deliverable.mp4`. For a full reproduction from raw datasets (ingest → CNN cache → train → render, ~3 hours), see `scripts/run_full_pipeline.sh`.

## Overview

A single LateFusionLSTM is trained jointly on three robotic manipulation datasets ingested through Mosaico: **Reassemble** (HDF5), **DROID** (h5 / ROS bag like) and **Fractal RT-1** (TFRecord, RLDS). The model fuses 15 canonical kinematic features with a precomputed visual stream (MobileNetV3 Small, layer 6 + `AdaptiveMaxPool2d(3)`, reduced from 360 to 64 components by IncrementalPCA) and emits a calibrated failure probability per 50 frame window (1 second at 50 Hz).

Same code path for all three formats: `MosaicoClient -> QuerySequence -> SequenceHandler -> DataFrameExtractor -> SyncTransformer(50 Hz, SyncHold) -> feature_mapper.project -> add_derived_features -> label_adapters`. Mosaico exposes the catalog as a uniform DataFrame source over Apache Arrow, so application code never reads `.h5`, `.tfrecord` or ROS bag files directly.

Per-frame `P(failure)` can then be overlaid on top of held-out recorded videos as a visual demonstration of the classifier's behavior.

### Final Results

Test set, stratified per dataset 70 / 15 / 15 split (seed 42), threshold tuned on the validation split:

| Metric                                          | Value      |
|-------------------------------------------------|------------|
| Overall AUC (test)                              | **0.685**  |
| Overall F1 (test, threshold 0.806)              | **0.334**  |
| Class separation (P fail mean - P succ mean)    | **0.173**  |
| Reassemble AUC                                  | 0.626      |
| DROID AUC                                       | 0.614      |
| Fractal RT-1 AUC                                | 0.563      |
| P(failure) range observed                       | [0.027, 0.999] |
| Confident failure (P > 0.8) window fraction     | 25.1 percent |
| Confident success (P < 0.2) window fraction     | 13.8 percent |

Trainable parameters: 108547. Training time: about 6 minutes on Ryzen 7 CPU (no GPU). Full hyperparameters and per dataset metrics are persisted in [`results/multimodal_indist_v9_sharp/results.json`](results/multimodal_indist_v9_sharp/results.json).

## Architecture

- **Visual encoder**: frozen MobileNetV3 Small ImageNet pretrained, taps layer 6 + `AdaptiveMaxPool2d(3)`. 360 dim spatial features per frame, stored at fp16 in the cache.
- **Dimensionality reduction**: `IncrementalPCA(n_components=64)` fitted on training set CNN features and applied unchanged to validation and test.
- **Temporal head**: `LateFusionLSTM` with two BiLSTMs (kin: 15 -> hidden 64; visual: 64 -> hidden 64), learned softmax attention pool over 50 timesteps in each stream, concatenated outputs feeding a single linear head.
- **Loss**: `BCEWithLogitsLoss` with `pos_weight=4.522` (auto from class balance), no label smoothing.
- **Regularization**: `kin_noise_std=0.02`, `WeightedRandomSampler` on the training split, dropout 0.35, gradient clip 1.0, SWA from epoch 8, `ReduceLROnPlateau`.

Detailed motivation for the design choices, training curves, and per-dataset ROC analysis are in [`BLOG_POST.md`](BLOG_POST.md).

## Repository Structure

```
grasp_integrity_predictor/
  data/
    cross_dataset_ingestor.py       Mosaico native multi dataset iterator
    feature_mapper.py               Canonical projection, 15 kin features, gripper binarization
    label_adapters.py               Per dataset success/failure labels (Boolean + SegmentInfo)
  models/
    cached_dataset.py               Sliding window dataset over the .npz cache
    cached_lstm.py                  LateFusionLSTM with attention pool over time
    trainer.py                      FocalLoss + reproducibility helpers
  scripts/
    audit_cache.py                          Per dataset distribution audit on the .npz cache
    precompute_cnn_features.py              One shot CNN feature cache (idempotent)
    refresh_kin_in_cache.py                 Refresh kin part only via Mosaico (no CNN re run)
    video_overlay_demo.py                   Per-act renderer (source video + P trace HUD)
    concat_acts.py                          Stitches per-act MP4s into a single video
    ingest_reassemble.py                    Reassemble ingest entry point
    ingest_droid_parallel.py                DROID parallel ingest
    ingest_generic.py                       Generic ingest entry point (auto plugin selection)
    backfill_reassemble_labels.py           Reassemble label injector (boolean.data from segments_info)
    check_label_topic.py                    Sanity check for the Reassemble label topic
    check_catalog_images.py                 Sanity check that image topics decode
    rebuild_catalog.py                      Rebuild Postgres catalog from MinIO blobs
    run_v9.sh                               Wrapper for the released training command
    find_test_split.py                      Generates the per-dataset stratified test split
  results/
    multimodal_indist_v9_sharp/             Released model + scaler + metrics + acts JSON
      checkpoints/best_model.pt             LateFusionLSTM trained weights
      scaler.npz                            kin/cnn z score + IPCA 64 components
      results.json                          hyperparameters + per dataset test metrics
      acts_v9_2parts_reassemble_test.json   per-act overlay configuration
      *.png                                 ROC, loss curves, confusion matrix
    test_split.json                         stratified per-dataset 70/15/15 split (seed 42)
  docker/
    compose-training.yml                    Postgres + MinIO + mosaicod stack
  run_training_cached.py                    Training entry point
  run_ingestion_parallel.py                 Parallel Reassemble ingest entry point
  BLOG_POST.md                              Technical writeup
  README.md                                 This file
  pyproject.toml
  requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

The Mosaico daemon (Postgres + MinIO + `mosaicod`) is needed only for full reproduction from raw datasets (Path B below). It is provisioned via Docker Compose:

```bash
docker compose -f docker/compose-training.yml up -d
```

Path A (loading the released model) does **not** require the daemon, only the local `.pt` and `.npz` files under `results/multimodal_indist_v9_sharp/`.

## Usage

Two entry points depending on whether the starting point is the released model or raw data.

### Path A: Load the released model and render the per-act overlays (about 2 minutes)

The repository ships the trained model, the test split definition, and the per-act overlay configuration so the result can be inspected without re-running the full pipeline:

```
results/multimodal_indist_v9_sharp/
  checkpoints/best_model.pt                  LateFusionLSTM trained weights (108547 params)
  scaler.npz                                 kin / cnn z score + IPCA 64 components
  results.json                               hyperparameters + per dataset test metrics
  acts_v9_2parts_reassemble_test.json        per-act overlay configuration
  *.png                                      ROC, loss curves, confusion matrix
results/test_split.json                      stratified per-dataset 70/15/15 split (seed 42)
```

Render the per-act overlays (each one overlays the per-frame `P(failure)` on top of the source video for one held-out test sequence):

```bash
python -m scripts.video_overlay_demo \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json \
  --model-dir   results/multimodal_indist_v9_sharp \
  --threshold   0.806 \
  --out-dir     results/video_overlay_demo
```

Acts are configured via the JSON at `results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json`: each entry binds a held-out test sequence to an `offset` (in 50 Hz cache frames) and a `duration` (seconds), with an optional `camera` field for Reassemble acts (`"hand"` for the wrist camera, `"hama1"` / `"hama2"` for the third-person side and front views available in the source HDF5).

### Path B: Full reproduction from raw datasets (about 3 hours total)

#### 1. Ingest the three datasets into Mosaico

The `--dataset-root` / `--data-root` flags below expect the local path where you downloaded each dataset. The placeholders in angle brackets are intentional; replace them with absolute paths on your machine.

```bash
# Reassemble (149 .h5 files), parallel ingestion
python run_ingestion_parallel.py \
  --workers 4 --dataset-root "<path-to-reassemble-data>"

# DROID (h5 / ROS bag like)
python scripts/ingest_droid_parallel.py --data-root "<path-to-droid-data>"

# Fractal RT-1 (TFRecord, RLDS)
python scripts/ingest_generic.py \
  --data-root "<path-to-fractal20220817-data>"
```

Sanity check the catalog before continuing:

```bash
python scripts/check_label_topic.py
python scripts/check_catalog_images.py
```

If `check_label_topic.py` reports the Reassemble label topic missing, populate it from the source HDF5 `segments_info` field:

```bash
python scripts/backfill_reassemble_labels.py --h5-root "<path-to-reassemble-data>"
```

See [Reassemble label schema](#reassemble-label-schema) below for the column paths the loader supports.

#### 2. Pre cache CNN features (about 3 hours, one shot)

For each Mosaico sequence, the script runs the full SDK chain (`process_sequence` -> DataFrame -> 50 Hz SyncHold -> JPEG decode at 112x112) and forwards each frame through a frozen ImageNet MobileNetV3 Small (layer 6 + `AdaptiveMaxPool2d(3)`) to produce 360 dim spatial features per frame. Output shape per sequence: `(kin: T x 15, cnn: T x 360 fp16, label: T)`, saved to `results/cnn_cache_spatial/{dataset}/{sequence}.npz`.

```bash
python scripts/precompute_cnn_features.py
```

The script is idempotent: existing `.npz` files are skipped on rerun.

If the catalog is updated after the cache is built, refresh only the kin part of the cache via Mosaico without re running the expensive CNN forward pass:

```bash
python scripts/refresh_kin_in_cache.py
```

#### 3. Train (about 6 minutes on CPU)

The training script:

1. Loads all `.npz` from `results/cnn_cache_spatial/`.
2. Splits stratified per dataset 70 / 15 / 15 by sequence label, seed 42, with a positional fallback for tiny minority classes (Reassemble n=48 has only 3 pure success sequences).
3. Builds sliding windows: `seq_len = 50` ticks (1 second at 50 Hz), `stride = 50` (no overlap, no leakage between train and test windows of the same sequence). Window label = max over the window.
4. Z scores kin features (mean / std on train, applied to val + test) and CNN features chunk by chunk to keep RAM bounded.
5. Fits `IncrementalPCA(n_components=64)` on training set CNN features and applies it to all splits. Visual stream collapses 360 -> 64.
6. Trains `LateFusionLSTM` with attention pool over time, BCE with auto `pos_weight`, `WeightedRandomSampler`, SWA, gradient clipping, dropout.
7. Saves `best_model.pt`, `scaler.npz` (kin / cnn z score + IPCA components), `results.json` (config + per dataset metrics), and ROC / loss / confusion matrix plots.

Wrapper script for the released configuration:

```bash
bash scripts/run_v9.sh released
```

Direct invocation:

```bash
python run_training_cached.py \
  --train-datasets reassemble,droid,fractal_rt1 \
  --test-datasets  reassemble,droid,fractal_rt1 \
  --cache-dir results/cnn_cache_spatial \
  --cnn-dim 360 \
  --output-dir results/multimodal_indist_v9_sharp \
  --epochs 30 --batch-size 64 \
  --lr 1e-3 --weight-decay 1e-3 \
  --dropout 0.35 --hidden 256 \
  --stride 50 \
  --use-scheduler --num-workers 0 \
  --loss bce \
  --label-smoothing 0.0 \
  --mixup-alpha 0.0 \
  --kin-noise-std 0.02 \
  --weighted-sampler \
  --ipca-components 64 \
  --late-fusion --attn-pool \
  --swa-start-epoch 8
```

#### 4. Render the per-act overlays (about 2 minutes)

See Path A above.

## Dependencies

| Tool | Version | Role |
|---|---|---|
| Python | 3.13+ | runtime |
| [PyTorch](https://pytorch.org/) | 2.2+ | model + training |
| torchvision | 0.17+ | MobileNetV3 Small backbone for the CNN pre-cache |
| [mosaicolabs](https://pypi.org/project/mosaicolabs/) | 0.4.0+ | Python SDK for [Mosaico](https://mosaico.dev), the data platform that unifies the three datasets behind a single DataFrame interface |
| numpy | 2.0+ | tensor / array math |
| pandas | 2.0+ | DataFrame contract from the SDK |
| scikit-learn | 1.4+ | IncrementalPCA, ROC, F1, stratified split |
| Pillow | 10.0+ | JPEG decode in `feature_mapper` |
| opencv-python | 4.8+ | source video read + HUD overlay + VideoWriter |
| matplotlib | 3.8+ | training plots |
| seaborn | 0.13+ | confusion matrix |
| h5py | 3.10+ | Reassemble label backfill + video blob extract |
| psycopg2-binary | 2.9+ | catalog rebuild |
| pyarrow | 22.0+ | catalog rebuild + DROID parquet metadata |
| rich | 13.0+ | multiprocess ingest console |

Ingestion plugins for the three datasets live in [`mosaico-alchemy`](https://github.com/mosaico-labs/mosaico-alchemy) (Mosaico's manipulation plugin pack).

CPU only friendly: training takes about 6 minutes on a Ryzen 7 once the CNN cache is built; rendering the per-act overlays takes another 1 to 2 minutes.

## Reassemble Label Schema

Two column paths are supported by [`data/label_adapters.py:label_reassemble`](data/label_adapters.py), checked in order:

1. **`Boolean` ontology.** Column path: `/grasp_failure_label.boolean.data`. Polarity: `True = failure`. Populated either by the SDK plugin or by `scripts/backfill_reassemble_labels.py`.

2. **`SegmentInfo` ontology.** Column path: `/grasp_failure_label.segment_info.success`. Polarity inverted: `True = success`, so the failure label is computed as `1 - success`. The `is_terminal` flag distinguishes the start (False) and end (True) boundary of each segment, and `parent_action` carries the optional nesting for `low_level` sub segments.

The first path with a present column wins; the loader falls back to the second.

## Key Design Choices

- **15 feature canonical kin schema.** 3 ee_pos + 4 ee_quat + 1 gripper scalar (8 base) + 7 finite difference derivatives. The 8 base features are the lowest common denominator across the three datasets after `DataFrameExtractor` flattens the ontologies, projected by `data/feature_mapper.py:project`. The 7 derivatives are added by `add_derived_features` and are the only per window time varying signal for episode labeled datasets (DROID, Fractal RT-1).
- **Gripper binarization in `feature_mapper.py`.** The three sources record the gripper state on incompatible scales (Reassemble metres, DROID normalized, Fractal binary). Without semantic canonicalization the z scored value identifies the dataset rather than the gripper state. `_binarize_gripper` applies per dataset thresholds (`Reassemble 0.025`, `DROID 0.5`, `Fractal 0.5`) inside the canonical projection layer.
- **Spatial CNN features (not GAP).** Layer 6 of MobileNetV3 Small followed by `AdaptiveMaxPool2d(3)` produces a 360 dim spatial vector that retains geometric structure (where in the frame the gripper / object is), not just semantics.
- **IPCA reduces visual to 64 dims.** `IncrementalPCA` is fit on training set CNN features and applied to all splits. The visual stream feeding the BiLSTM is 64 dim, dimensionally balanced with the kin BiLSTM hidden size.
- **Late fusion with attention pool over time.** kin (15 -> BiLSTM 64) and visual (64 -> BiLSTM 64) pass through separate BiLSTMs. Within each stream a learned softmax attention pools across the 50 window timesteps (replaces the last timestep readout). Output streams are concatenated only at the linear head. Adapted to episode level supervision where any frame in the window is informative.
- **BCE with auto pos_weight, no label smoothing.** Calibrated probability output, with class separation (P fail mean - P succ mean) of 0.173 on the test split and P(failure) covering the full [0.027, 0.999] range.
- **Stratified per dataset 70 / 15 / 15 split.** Built with sklearn `train_test_split(stratify=label_per_seq)` per dataset, seed 42, with a positional fallback when a class has fewer than 2 samples (Reassemble n=48 has only 3 pure success sequences).
- **50 Hz SyncHold resampling.** Mosaico's `SyncTransformer` with forward fill policy, safe across booleans / floats / arrays, keeps quaternions valid (interpolation would break unit norm).
- **Sliding windows:** `seq_len = 50` (1 second at 50 Hz), `stride = 50` (no overlap), label = max over the window.
- **Pre cache + LSTM only training.** The visual encoder is frozen, its forward pass is computed once and reused across N training runs. Without this, training the head N times on CPU is unfeasible.

## License

Apache 2.0, see [`LICENSE`](LICENSE). Copyright (c) 2026 Edoardo Bambini.

## Citations and Tools Used

- Technical writeup: [`BLOG_POST.md`](BLOG_POST.md)
- **Mosaico**: open-source data platform for Physical AI ([mosaico.dev](https://mosaico.dev), [github.com/mosaico-labs/mosaico](https://github.com/mosaico-labs/mosaico)). Python SDK package `mosaicolabs`.
- **mosaico-alchemy**: manipulation plugin pack with ingestion plugins for Reassemble, DROID, and Fractal RT-1 ([github.com/mosaico-labs/mosaico-alchemy](https://github.com/mosaico-labs/mosaico-alchemy)).
- **Reassemble dataset**: HDF5 manipulation recordings.
- **DROID dataset**: large scale teleoperated manipulation (h5 / ROS bag layout).
- **Fractal RT-1 dataset**: RT-1 robotics transformer rollouts (RLDS / TFRecord format).
- **MobileNetV3 Small**: ImageNet weights via `torchvision`.
