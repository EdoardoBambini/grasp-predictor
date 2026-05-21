# Cross-Format Grasp Failure Classifier on Mosaico

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Mosaico >= 0.4.0](https://img.shields.io/badge/Mosaico-%3E%3D0.4.0-1f77b4.svg)](https://github.com/mosaico-labs/mosaico) [![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A binary grasp-failure classifier trained jointly on two heterogeneous robotic
manipulation datasets, DROID (stored as parquet / RLDS) and Fractal RT-1 (stored as
TFRecord / RLDS), through the [Mosaico](https://github.com/mosaico-labs/mosaico) data
platform. Both datasets are ingested once and read through a single uniform catalog, so
the same query and extraction code selects, labels and streams sequences regardless of
the underlying storage format. The classifier is the vehicle, not the goal: the project
demonstrates Mosaico's cross-format query and extraction layer for training-set
discovery, labelling and curation.

The released model is `LateFusionLSTMMIL`, a bag-level Multiple Instance Learning
classifier with a late-fusion LSTM encoder and gated attention pooling (237635
parameters, seed 7). Each frame is represented by a 16-D kinematic vector and a 360-D
cached CNN descriptor, with a Universal Sentence Encoder instruction embedding for
Fractal RT-1. Full design and analysis are in [`TECHNICAL_WRITEUP.md`](TECHNICAL_WRITEUP.md).

## Results

Metrics from `results/multimodal_indist_v10l_seed7/results.json`. Validation and test
partitions sit at each dataset's natural failure prior, so the per-dataset breakdown is
the production-representative view.

| Dataset      | AUC (seed 7) | Class sep (seed 7) | Accuracy (seed 7) | AUC (3-seed)  | Class sep (3-seed) |
|--------------|--------------|--------------------|-------------------|---------------|--------------------|
| DROID        | 0.91         | 0.65               | 0.91              | 0.94 +/- 0.03 | 0.71 +/- 0.04      |
| Fractal RT-1 | 0.51         | 0.00               | 0.39              | 0.50 +/- 0.01 | 0.00 +/- 0.00      |

Class separation is the gap between the mean predicted failure probability on failure
versus success sequences. On DROID the mean is 0.80 on failures and 0.15 on successes;
on Fractal RT-1 it is 0.51 on both. The asymmetry is a property of the labels and is
analysed in the writeup.

## Installation

```bash
pip install mosaicolabs
pip install -r requirements.txt
```

Python 3.13 or newer. Training runs on CPU.

## Quick start

The pipeline assumes DROID and Fractal RT-1 are already ingested into a local Mosaico
instance through the [mosaico-alchemy](https://github.com/mosaico-labs/mosaico-alchemy)
plugins, for example:

```bash
mosaico_alchemy manipulation --datasets /path/to/droid /path/to/fractal20220817_data \
  --host localhost --port 6726 --write-mode sync
```

Then run the pipeline end to end:

```bash
# 1. Precompute the frozen CNN feature cache (long step; cache is not shipped)
python scripts/precompute_cnn_features.py

# 2. Build per-dataset success/failure pools with Mosaico queries
python scripts/build_balanced_pools_via_mosaico.py --datasets droid,fractal_rt1

# 3. Train the released configuration (or run a fast smoke check)
bash scripts/train.sh
python run_training_cached.py --smoke

# 4. Regenerate the figures
python scripts/make_figures.py
```

Per-seed metrics, hyperparameters and pools are persisted under `results/`; the released
checkpoint is seed 7, with seeds 42 and 123 alongside it for the three-seed aggregate.

## Demo

`scripts/video_overlay_demo.py` overlays the model's per-window failure probability on
the exterior camera of DROID episodes held out from training, validation and test, and
`scripts/concat_acts.py` stitches the clips into a single timeline. The shipped
`results/video_overlay_demo/acts_v10l_droid_test.json` selects four held-out episodes
(two that fail, two that succeed). With the source DROID videos on disk:

```bash
python scripts/video_overlay_demo.py --droid-parquet-root /path/to/droid/data
python scripts/concat_acts.py
```

The overlaid probability is the model's running prediction over the windows observed so
far, drawn as an online monitoring timeline.

## Repository layout

```
data/
  cross_dataset_ingestor.py   Mosaico query, extract, sync, project, label
  feature_mapper.py           canonical 15-feature projection + gripper binarization
  label_adapters.py           per-dataset success/failure labels
models/
  mil_attention.py            LateFusionLSTMMIL (released model)
  pcgrad.py                   PCGrad gradient surgery
  cached_bag_dataset.py       bag-level dataset over the .npz cache
  cached_lstm.py, cached_dataset.py, trainer.py   earlier window-level path
scripts/
  build_balanced_pools_via_mosaico.py   query-based curation
  precompute_cnn_features.py            one-shot CNN feature cache
  train.sh                              released training command
  make_figures.py                       regenerate the figures
  video_overlay_demo.py                 per-window overlay on held-out DROID
  concat_acts.py                        stitch overlay clips
run_training_cached.py        training entry point
results/                      released model, metrics, pools, figures
docs/figures/                 figures used in the technical writeup
```

## Dependencies

Python 3.13+, `mosaicolabs>=0.4.0` for the data platform, PyTorch and torchvision (the frozen
MobileNetV3 Small backbone), plus numpy, pandas, scikit-learn, Pillow, opencv-python,
matplotlib and seaborn. Ingestion utilities add h5py, psycopg2-binary, pyarrow and
rich. See [`requirements.txt`](requirements.txt). On a Ryzen 7, a single seed trains in
roughly 15 to 20 minutes once the CNN cache exists.

## License

Apache 2.0, see [`LICENSE`](LICENSE).
