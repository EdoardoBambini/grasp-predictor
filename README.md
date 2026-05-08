<div align="center">
  <picture>
    <a href="https://mosaico.dev"><img alt="Mosaico logo" src="https://mosaico.dev/github-hero.webp" width="600px"></a>
  </picture>
</div>
<br/>

<p align="center">
  <a href="https://discord.gg/mwQtFnsckE"><img src="https://shields.io/discord/1413199575442784266?color=%235865f2" alt="discord" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="python version"/></a>
  <a href="https://docs.mosaico.dev"><img src="https://img.shields.io/badge/-%20Documentation-%23bd38b4?style=flat&logo=readthedocs&logoColor=white&labelColor=gray" alt="documentation"></a>
</p>

<p align="center">
  <a href="https://github.com/mosaico-labs/mosaico/pkgs/container/mosaicod"><img src="https://img.shields.io/badge/mosaicod-v0.4.0-orange" /></a>
  <a href="https://pypi.org/project/mosaicolabs"><img src="https://img.shields.io/pypi/v/mosaicolabs?color=blue" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white" alt="pytorch" /></a>
</p>

# Grasp Integrity Predictor

Binary grasp failure classifier for robotic manipulation sequences, trained cross format on three heterogeneous datasets via [Mosaico](https://mosaico.dev), the data platform for Physical AI, and validated by overlaying its per-frame predictions on top of held-out recorded videos, with an active-control variant that aborts the playback at the first threshold crossing.

> Case study built on **Mosaico** ([mosaico.dev](https://mosaico.dev), [github.com/mosaico-labs/mosaico](https://github.com/mosaico-labs/mosaico)). Manipulation plugins live in [`mosaico-alchemy`](https://github.com/mosaico-labs/mosaico-alchemy). Full writeup in [`BLOG_POST.md`](BLOG_POST.md).

## Overview

A single LateFusionLSTM is trained jointly on three robotic manipulation datasets ingested through the Mosaico SDK: **Reassemble** (HDF5), **DROID** (h5 / ROS bag like) and **Fractal RT-1** (TFRecord, RLDS). The model fuses 15 canonical kinematic features with a precomputed visual stream (MobileNetV3 Small, layer 6 + `AdaptiveMaxPool2d(3)`, reduced from 360 to 64 components by IncrementalPCA) and emits a calibrated failure probability per 50 frame window (1 second at 50 Hz).

Same code path for all three formats: `MosaicoClient -> QuerySequence -> SequenceHandler -> DataFrameExtractor -> SyncTransformer(50 Hz, SyncHold) -> feature_mapper.project -> add_derived_features -> label_adapters`. The Mosaico SDK exposes the catalog as a uniform DataFrame source over Apache Arrow, so application code never reads `.h5`, `.tfrecord` or ROS bag files directly.

The deliverable is a four act sequence (two passive monitoring, two active control) rendered directly on top of the recorded source videos. In passive acts the classifier's per-frame `P(failure)` is overlaid as a HUD timeline while the recording plays uninterrupted; in active acts the classifier is granted authority to halt the playback at the first `P >= 0.806` crossing, freezing the source frame at the moment of intervention with an `ABORT INITIATED BY MOSAICO ML` overlay. All four sequences come from the held-out 15 percent test split.

## Final Results

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
    concat_acts.py                          Stitches per-act MP4s into the final deliverable
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
      acts_v9_2parts_reassemble_test.json   demo configuration consumed by video_overlay_demo.py
      *.png                                 ROC, loss curves, confusion matrix
    video_overlay_demo/
      module_c_deliverable.mp4              final deliverable (1280x720 30 fps, 64.5 s)
    test_split.json                         stratified per-dataset 70/15/15 split (seed 42)
  docker/
    compose-training.yml                    Postgres + MinIO + mosaicod stack
  run_training_cached.py                    Training entry point
  run_ingestion_parallel.py                 Parallel Reassemble ingest entry point
  BLOG_POST.md                              Case study writeup
  README.md                                 This file
  pyproject.toml
  requirements.txt
```

## Requirements

- Python 3.13 (see [`pyproject.toml`](pyproject.toml)).
- A running `mosaicod` daemon reachable on `127.0.0.1:6726` with the three datasets ingested.
- The `mosaicolabs` Python package and the `mosaico-alchemy` ingestion plugins. The Reassemble plugin emits the `/grasp_failure_label` topic via either the `Boolean` ontology or the `SegmentInfo` ontology (see [Reassemble label schema](#reassemble-label-schema) below).
- CPU only friendly: training takes about 6 minutes on a Ryzen 7 once the CNN cache is built; rendering the deliverable takes another 1 to 2 minutes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Editable install of the Mosaico SDK (point to your local clone)
pip install -e <path-to-mosaico-sdk-py>
```

The Mosaico daemon (Postgres + MinIO + `mosaicod`) is provisioned via Docker Compose:

```bash
docker compose -f docker/compose-training.yml up -d
```

## Reproducing the Case Study

Two entry points depending on whether the starting point is the released model or raw data.

### Path A: Load the released model and render the deliverable (about 2 minutes)

The repository ships the trained model, the test split definition, and the final deliverable so the result can be verified without re running the full pipeline:

```
results/multimodal_indist_v9_sharp/
  checkpoints/best_model.pt                  LateFusionLSTM trained weights (108547 params)
  scaler.npz                                 kin / cnn z score + IPCA 64 components
  results.json                               hyperparameters + per dataset test metrics
  acts_v9_2parts_reassemble_test.json        demo configuration
  *.png                                      ROC, loss curves, confusion matrix
results/test_split.json                      stratified per-dataset 70/15/15 split (seed 42)
results/video_overlay_demo/
  module_c_deliverable.mp4                   final deliverable (1280x720 30 fps, 64.5 s)
```

To re render the deliverable from the released model (overlays the per-frame `P(failure)` on top of the source video for each act, then concatenates the four acts with title cards):

```bash
# 1. Render the four per-act MP4s into results/video_overlay_demo/act_*.mp4
python -m scripts.video_overlay_demo \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json \
  --model-dir   results/multimodal_indist_v9_sharp \
  --threshold   0.806 \
  --out-dir     results/video_overlay_demo

# 2. Stitch them into the final 64.5 s deliverable with intro / title / outro cards
python -m scripts.concat_acts \
  --in-dir      results/video_overlay_demo \
  --acts-config results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json \
  --out         results/video_overlay_demo/module_c_deliverable.mp4
```

Acts are configured via the JSON at `results/multimodal_indist_v9_sharp/acts_v9_2parts_reassemble_test.json`: each entry binds a held-out test sequence to an `offset` (in 50 Hz cache frames) and a `duration` (seconds), with an optional `camera` field for Reassemble acts (`"hand"` for the wrist camera, `"hama1"` / `"hama2"` for the third-person side and front views available in the source HDF5). Failure acts at index >= 2 trigger the active-control branch and freeze the source frame at the first `P >= threshold` crossing.

### Path B: Full reproduction from raw datasets (about 3 hours total)

#### 1. Ingest the three datasets into Mosaico

```bash
# Reassemble (149 .h5 files), parallel ingestion
python run_ingestion_parallel.py \
  --workers 4 --dataset-root "D:/datasets/reassemble/data"

# DROID (h5 / ROS bag like)
python scripts/ingest_droid_parallel.py --data-root "D:/datasets/droid/data"

# Fractal RT-1 (TFRecord, RLDS)
python scripts/ingest_generic.py \
  --data-root "D:/datasets/fractal20220817_data/0.1.0"
```

Sanity check the catalog before continuing:

```bash
python scripts/check_label_topic.py
python scripts/check_catalog_images.py
```

If `check_label_topic.py` reports the Reassemble label topic missing, populate it from the source HDF5 `segments_info` field:

```bash
python scripts/backfill_reassemble_labels.py --h5-root "D:/datasets/reassemble/data"
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

#### 4. Render the deliverable (about 2 minutes)

See Path A above. The deliverable plays four held-out test sequences end to end: two passive monitoring acts (the classifier observes, the recording plays uninterrupted) and two active control acts (the classifier has authority to halt the playback at the first `P >= 0.806` crossing). The two failure acts are two distinct 12 s windows of the same Reassemble Insert-Ethernet recording, capturing two failure points along the same trajectory: the passive failure act is centered on the alignment failure downstream of the grasp, the active failure act is centered on the upstream grasp event itself, where the abort authority freezes the source frame with the object plainly held in the gripper.

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
- **50 Hz SyncHold resampling.** `SyncTransformer` with forward fill policy, safe across booleans / floats / arrays, keeps quaternions valid (interpolation would break unit norm).
- **Sliding windows:** `seq_len = 50` (1 second at 50 Hz), `stride = 50` (no overlap), label = max over the window.
- **Pre cache + LSTM only training.** The visual encoder is frozen, its forward pass is computed once and reused across N training runs. Without this, training the head N times on CPU is unfeasible.

## Citations

- Case study writeup: [`BLOG_POST.md`](BLOG_POST.md)
- Mosaico, the data platform for Physical AI: [mosaico.dev](https://mosaico.dev), [github.com/mosaico-labs/mosaico](https://github.com/mosaico-labs/mosaico)
- Manipulation plugins: [`mosaico-alchemy`](https://github.com/mosaico-labs/mosaico-alchemy)
- Mosaico SDK: `mosaicolabs` Python package
- Reassemble dataset (HDF5 manipulation recordings)
- DROID dataset (h5 / ROS bag layout, large scale teleoperated manipulation)
- Fractal RT-1 dataset (RLDS / TFRecord, RT-1 robotics transformer rollouts)
- MobileNetV3 Small ImageNet weights via `torchvision`
