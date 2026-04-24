# Grasp Integrity Predictor

Binary classifier for robotic grasp failure, trained cross-dataset on four
manipulation sources (Reassemble, DROID, MML, Fractal RT1). The model is a
stacked bidirectional LSTM; all four datasets are pulled through the Mosaico
SDK as a single data layer.

Everything runs on Mosaico: no custom parser, no direct reads from .h5,
parquet, ROS bag or TFRecord. Each source goes through the same
`DataFrameExtractor + SyncTransformer` chain, gets projected onto a
canonical 8-feature schema (end-effector pose + quaternion + gripper state),
is enriched with 7 finite-difference derivatives, and finally labeled by a
per-dataset adapter.

## Requirements

- Python 3.13 (see `pyproject.toml`; the ingestion and simulation scripts
  pin `py -3.13`).
- A running `mosaicod` daemon reachable at `127.0.0.1:6726` with the four
  datasets already ingested (see "Ingestion" below).
- The `mosaicolabs` SDK. This project uses a local development fork
  (`mosaico-marco/mosaico-sdk-py`) that carries the manipulation pack with
  the plugins for Reassemble, DROID, MML and Fractal RT1 — these plugins
  are not upstream yet. All pipeline APIs (`MosaicoClient`,
  `DataFrameExtractor`, `SyncTransformer`, `QuerySequence`,
  `SequenceHandler`) are the standard ones, unpatched. The Reassemble
  plugin carries a local patch (inline emission of `/grasp_failure_label`
  from `segments_info`, trimmed topic list) agreed with the team.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# or: .venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Install the SDK fork in editable mode (it carries the manipulation pack
# and the patched Reassemble plugin — see Requirements above).
pip install -e ../../mosaico-marco/mosaico-sdk-py
```

## Pre-trained model

The final cross-dataset checkpoint lives in:

```
results/crossdataset_run/checkpoints/best_model.pt
```

Alongside it, `results/crossdataset_run/` contains the final training
artifacts:

- `results.json`: full config, per-split window counts, final test metrics
  and training history
- `loss_curves.png`, `val_accuracy.png`, `confusion_matrix.png`, `roc_curve.png`

## Running the pipeline

### 1. Ingestion (once per dataset)

With `mosaicod` already running, ingest each dataset through its
manipulation-pack plugin in the SDK fork. Reassemble is the only one that
needs a dedicated entry point in this repo (the plugin is patched to emit
the label topic inline):

```bash
python scripts/ingest_reassemble.py
```

DROID, MML and Fractal RT1 are ingested through their own plugins in the
SDK fork (`mosaicolabs.packs.manipulation.datasets.{droid,mml,fractal_rt1}`);
point them at the dataset roots as documented in the SDK.

Sanity check that the Reassemble label topic is present:

```bash
python scripts/check_label_topic.py
```

`scripts/backfill_reassemble_labels.py` is the legacy path used before the
Reassemble plugin patch landed (writes `/grasp_failure_label` in a second
session). With the patched plugin the topic is emitted during ingest, so
the backfill is only needed for catalogs populated by older plugin
versions.

### 2. Training

```bash
# Full cross-dataset training (all 4 datasets, default 40 epochs).
# The final released run was done with --epochs 30 (early stopped at epoch 9).
python run_training_crossdataset.py --epochs 30

# Smoke test on a small subset
python run_training_crossdataset.py --cap 5 --epochs 3

# Single-dataset training
python run_training_crossdataset.py --datasets reassemble --cap 50 --epochs 40
```

Output goes to `results/crossdataset_run/` (override with `--output`).

### 3. Evaluation

Evaluation runs automatically at the end of training: the script loads the
best checkpoint and reports accuracy, AUC, F1 and the confusion matrix on
the held-out test split. The numbers are written to `results.json` and
visualized in the PNG plots.

To re-evaluate the saved model without retraining, load
`results/crossdataset_run/checkpoints/best_model.pt` into a
`GraspIntegrityLSTM` with the same hyperparameters used in the run (see the
`config` block in `results.json`) and run the `evaluate()` helper from
`run_training_crossdataset.py`.

### 4. Simulation (Module C)

Closed-loop MuJoCo replay driven by Mosaico: the script pulls a Reassemble
sequence through the SDK, initializes the Franka (Menagerie) EE to the
first recorded pose, drives the arm in IK along the recorded trajectory
and runs the LSTM on the same normalized 15-feature tensor used at
training time. P(fail) is compared tick-by-tick against
`/grasp_failure_label`.

One-off setup (dumps the training-time StandardScaler so inference sees
the same input distribution):

```bash
python scripts/dump_scaler.py
```

Run:

```bash
# replay (default): arm follows the recorded Reassemble trajectory via IK
python run_simulation.py

# demo: arm does a scripted pick-and-place on the scene objects,
# LSTM runs in parallel on the real Mosaico feature tensor (used for the video)
python run_simulation.py --mode demo

# specific sequence
python run_simulation.py --sequence reassemble_xxx
```

Both modes share the Mosaico-driven state init (first recorded EE pose via
IK spin) and feed the LSTM with the same normalized 15-feature window.
They differ only in the IK target source: Mosaico trajectory in `replay`
(brief-strict), scripted waypoints in `demo` (visually coherent with the
lab scene, used for the demonstration video).

A MuJoCo viewer opens; the HUD shows LSTM P(fail), the ground-truth label
at the current tick, running accuracy and inference latency. Final
per-timestep classification metrics and full prediction/label traces are
written to `outputs/sim_replay_stats.json`.

## Folder structure

```
grasp_integrity_predictor/
  data/
    cross_dataset_ingestor.py    Mosaico-native multi-dataset iterator
    feature_mapper.py            Per-dataset projection to the canonical 8 features
    label_adapters.py            Per-dataset success/failure labeling
    tensor_builder.py            Sliding-window tensor builder
  models/
    lstm.py                      GraspIntegrityLSTM (tsai LSTMPlus + vanilla fallback)
    dataset.py                   PyTorch Dataset wrapper + temporal split helper
    trainer.py                   Training loop with early stopping and focal/BCE loss
  scripts/
    ingest_reassemble.py             Reassemble ingest via patched ReassemblePlugin + ManipulationRunner
    backfill_reassemble_labels.py    Legacy: backfill /grasp_failure_label (pre-patch flow)
    check_label_topic.py             Sanity check the label topic is present
    rebuild_catalog.py               Rebuild Postgres catalog from MinIO blobs
    wipe_reassemble.py               SDK-only deletion of reassemble_* sequences
    dump_scaler.py                   Dump the training StandardScaler to results/crossdataset_run/scaler.npz
  simulation/                    MuJoCo closed-loop simulation and assets
  results/
    crossdataset_run/            Final training artifacts + best checkpoint
  docker/
    compose-training.yml         Postgres + MinIO + mosaicod stack
  run_training_crossdataset.py   Training entry point
  run_simulation.py              Mosaico-driven MuJoCo replay + real-time LSTM inference (Module C)
  run_ingestion_parallel.py      Parallel Reassemble ingest (multiprocess workers, faster alternative to scripts/ingest_reassemble.py)
  requirements.txt
  pyproject.toml
```

## Key design choices

- **Canonical 8-feature schema**: end-effector xyz, quaternion xyzw, gripper
  state. This is the true lowest common denominator across Reassemble,
  DROID, MML and Fractal RT1 after `DataFrameExtractor` normalization. Any
  richer feature (joint torques, tactile) is present in only one or two
  datasets and would break the cross-dataset bridge.
- **7 finite-difference derivatives** added on top (linear velocity, quaternion
  rate, gripper velocity): they are the only per-sample time-varying signal
  available for the datasets whose ground-truth label is episode-level (DROID,
  Fractal). Without them, the LSTM would collapse to a constant predictor on
  those sequences.
- **50Hz SyncHold resampling**: SyncTransformer with forward-fill policy.
  Required by the brief, safe across booleans / floats / arrays, and keeps
  quaternions valid (interpolation would not).
- **Sliding windows**: sequence length 50 (= 1 s at 50Hz), stride 10 (80%
  overlap), label = max over the window (conservative: one failed sample
  taints the window).
- **Stratified per-dataset split**: each dataset is cut 70/15/15 and the
  splits are concatenated, so every split sees all four datasets. A global
  temporal split would put all of Reassemble into train and all of Fractal
  into test.
- **BCEWithLogitsLoss with automatic pos_weight**: the positive class
  (failure) is a minority; the weight is computed from the training label
  counts (`n_neg / n_pos`). Focal loss is available in the trainer as an
  alternative but not used in the final run.
