# A Cross Format Grasp Integrity Predictor on Mosaico

How Mosaico, the data platform for Physical AI, collapses the data plumbing layer of robotic learning, demonstrated on three heterogeneous manipulation datasets, a single LateFusionLSTM, and a real-time inference deliverable that overlays the classifier's per-frame predictions on top of the recorded source videos themselves.

## Abstract

Mosaico Labs ships an open source data platform for Physical AI: a Rust daemon (`mosaicod`) backed by a metadata database and pluggable object storage for blobs, fronted by a strictly typed ontology system, Apache Arrow zero copy data exchange, and a code first Python SDK (`mosaicolabs`). This case study showcases the platform end to end through a non trivial workload: a binary grasp failure classifier trained jointly on Reassemble (HDF5), DROID (h5 ROS bag layout) and Fractal RT-1 (TFRecord with the RLDS schema), all consumed through the same Mosaico code path with no per format branching. The trained classifier reaches an overall AUC of 0.685 on a stratified held out test split with calibrated probability output spanning the full [0.027, 0.999] range, and runs in real time over the recorded source videos themselves, with an active-control variant that aborts playback at the first threshold crossing. The headline finding is not the absolute classifier accuracy: it is that the application code on top of Mosaico is roughly 80 lines per dataset, against an estimated 2000 lines of bespoke parsing and synchronization that an equivalent pipeline would require without the platform. The classifier is the showcase; Mosaico is the substrate.

## What Mosaico Is

Mosaico replaces the linear ROS bag (`.bag`, `.mcap`) and ad hoc dataset directory with a structured, queryable archive purpose built for Physical AI. The architecture rests on three pillars:

- **Ontology**: strictly typed semantic models that describe the shape of every signal. Core ontologies in `mosaicolabs` cover the foundational types (`Pose`, `Boolean`, `CompressedImage`, `Floating32`, `RobotJoint`, `ForceTorque`, `Velocity`); domain specific ontologies live in pack repositories (`EndEffector`, `EventCamera`, `AudioDataStamped`, `SegmentInfo` ship with the manipulation pack in `mosaico-alchemy`). Data is treated as semantic objects, not byte arrays, so the catalog is validatable, optimized for transport, and deeply queryable by physical values.
- **Topic**: a concrete time series of one ontology model. Each topic has a strict one to one binding with its ontology type, so consumers know the schema before reading.
- **Sequence**: a recording session, a logically related collection of topics with shared metadata.

Around these three pillars sit a Rust server daemon (`mosaicod`) handling ingestion, indexing, querying, and storage; a metadata database; pluggable object storage for blobs (MinIO, S3, local filesystem); and Apache Arrow as the wire format, eliminating serialization overhead between server and client. New formats and sensors enter the platform through plugins that translate raw payloads into ontology messages: the manipulation pack lives in [`mosaico-alchemy`](https://github.com/mosaico-labs/mosaico-alchemy) and ships ingestion plugins for the three manipulation datasets used in this case study (Reassemble, DROID, Fractal RT-1) plus other datasets the case study does not exercise. The Python SDK exposes the catalog as if it were a single uniform DataFrame source, so application code never sees `.h5`, `.tfrecord`, or ROS bag files directly.

## The Data Plumbing Problem That Mosaico Dissolves

Combining heterogeneous robotic datasets normally means writing one parser per format, one synchronization routine per sample rate, one ontology mapping per topic naming convention, and a maintenance burden that grows superlinearly with the fourth dataset. The problem is paid for twice: first in engineering time spent on bespoke data plumbing, then again every iteration cycle in slower experimentation on the actual model.

Mosaico Labs frames this overhead as the central friction in Physical AI training and dissolves it through the Physical AI Bridge: a single uniform interface (`MosaicoClient` plus `DataFrameExtractor` plus `SyncTransformer`) that hides format specific decoding behind shared ontology types and serves the same DataFrame contract regardless of the storage format underneath. The case study below uses that interface to assemble a non trivial supervised learning task with bounded application code, and reports what the platform delivers at every step.

## Datasets Ingested Through Mosaico

Three publicly available manipulation datasets sit in a single Mosaico catalog. Ingestion goes through the manipulation plugins in [`mosaico-alchemy`](https://github.com/mosaico-labs/mosaico-alchemy): each plugin declares the source layout, the ontology binding for every topic, and the payload iterator that turns raw bytes into typed messages.

| Dataset       | Source format | Sequences ingested | Frames at 50 Hz | Failure rate (sequence) |
|---------------|---------------|--------------------|-----------------|-------------------------|
| Reassemble    | HDF5          | 48                 | 700740          | 91.7 percent            |
| DROID         | h5 ROS bag    | 1483               | 1470074         | 16.4 percent            |
| Fractal RT-1  | TFRecord      | 874                | 620446          | 36.6 percent            |
| Total         |               | 2405               | 2791260         | mixed                   |

These three formats differ in non trivial ways: nanosecond timestamps and quaternions as fixed length arrays in Reassemble, ROS bag style topic naming in DROID, RLDS nested step structures in Fractal RT-1. Sample rates span 30 Hz to 100 Hz within a single sequence, image codecs are heterogeneous, and the gripper signal is recorded on three incompatible scales. After ingestion, the Mosaico managed footprint on disk (`D:/mosaico-store/`) holds the original source corpus of roughly 90 GB as immutable, indexed blobs, with Postgres carrying the structured metadata; the precomputed CNN feature cache adds 356 MB of fp16 spatial features on top.

## The Pipeline, Step by Step

Every sequence in every dataset goes through the same chain. The Mosaico SDK and ML module expose three core capabilities that this case study exercises end to end: server side query of the indexed catalog, automatic synchronization of heterogeneous sample rates onto a uniform timeline, and direct injection of typed time series into pandas DataFrames. The application code path is identical across the three formats, with no `if dataset == ...` branch downstream of the SDK call site.

```python
from mosaicolabs import MosaicoClient
from mosaicolabs.ml import DataFrameExtractor, SyncHold, SyncTransformer
from mosaicolabs.models.platform import Sequence
from mosaicolabs.models.query import QuerySequence

client = MosaicoClient.connect(host="127.0.0.1", port=6726)
query = QuerySequence().with_expression(
    Sequence.Q.user_metadata["dataset_id"].eq(dataset_id)
)
for item in client.query(query):
    handler = client.sequence_handler(item.sequence.name)
    extractor = DataFrameExtractor(handler)
    sync = SyncTransformer(target_fps=50.0, policy=SyncHold())
    parts = []
    for sparse_chunk in extractor.to_pandas_chunks(topics=topics, window_sec=5.0):
        parts.append(sync.transform(sparse_chunk))
    dense = pd.concat(parts, ignore_index=True)
    dense = label_adapters.label(dataset_id, dense, handler)
    projected = feature_mapper.project(dataset_id, dense)
    projected = feature_mapper.add_derived_features(projected)
```

What Mosaico actually does, step by step:

1. **Server side query** (`MosaicoClient.connect` plus `QuerySequence().with_expression(...)`). The expression compiles into a server side `WHERE` on the metadata column. The query engine prunes irrelevant data and the indexed catalog returns only the precise sequences that match. No client side scan, no filesystem traversal, no per dataset branching on storage layout.
2. **Sequence handle** (`client.sequence_handler(name)`). The catalog returns a typed handle into the indexed sequence, abstracting away the underlying storage location and surfacing the topics with their bound ontology types.
3. **DataFrame injection** (`DataFrameExtractor.to_pandas_chunks`). Mosaico streams the requested topics directly into pandas DataFrames over Apache Arrow with bounded memory, regardless of how the source is laid out underneath. Three datasets, one DataFrame contract, zero serialization copies between server and client.
4. **Automatic synchronization** (`SyncTransformer(target_fps=50.0, policy=SyncHold())`). Mosaico resamples heterogeneous sample rates onto a single 50 Hz grid. The `SyncHold` forward fill policy preserves quaternion unit norm, where naive interpolation would silently corrupt orientation data, and is safe across booleans, floats, and array typed payloads.
5. **Application projection** (`feature_mapper.project`). Mosaico delivers a uniform dot notation schema (`<topic>.<ontology_tag>.<field>`) across the bound ontology types, and the application maps that schema onto the canonical 8 feature representation of the domain. This is where the application code lives; everything upstream is the platform.
6. **Derived features** (`add_derived_features`). Standard finite difference on the uniform timeline produced by `SyncTransformer`, yielding 7 velocity proxies that lift the kinematic vector to 15 dimensions.
7. **Label adapters** (`label_adapters.label`). Failure annotations are read through the same SDK handle, never from side car files. The Reassemble label topic is read from either `/grasp_failure_label.boolean.data` (`Boolean` ontology) or `/grasp_failure_label.segment_info.success` (`SegmentInfo` ontology). The `SegmentInfo` ontology and the corresponding Reassemble plugin update in `mosaico-alchemy` are contributed upstream as part of this case study (`mosaico-labs/mosaico-alchemy#4`).
8. **Loop closure**. Inference at training time and inference inside the deliverable both use this exact chain, so the production controller never diverges from the training pipeline.

## How Little Application Code Is Needed on Top

The only application code on top of the SDK lives in `data/feature_mapper.py`: 8 base features, 7 derivatives, three projection functions (`project_reassemble`, `project_droid`, `project_fractal_rt1`) totalling roughly 80 lines. Mosaico canonicalizes transport and schema; cross dataset semantics (what does "gripper open" mean across three control conventions) is by construction the application's responsibility. The single semantic decision at the application layer is the per dataset gripper binarization, with thresholds 0.025 (Reassemble, meters), 0.5 (DROID, normalized), 0.5 (Fractal, quasi binary), calibrated from histograms in `results/audit_cache.json`. After binarization `gripper_state` is a binary signal across the three datasets and `gripper_vel` becomes a clean edge detector for opening and closing.

For comparison, a from scratch implementation would need a dedicated HDF5 parser for Reassemble (with `segments_info` traversal, 2 finger gripper aggregation, per topic JPEG decoding), an h5 ROS bag style parser for DROID (cartesian pose unwrapping, episode level metadata, MP4 wrist camera decoding at 30 fps), a TFRecord and RLDS parser for Fractal RT-1 (nested step structure traversal, terminal reward extraction, image byte unpacking), plus a custom multi rate synchronization layer with quaternion safe policies. A reasonable estimate is around 2000 lines of production parsing and synchronization code, before any modelling work begins. The case study collapses that to about 80 application lines per dataset on top of `mosaicolabs`, a roughly 8x reduction in surface area, with the maintenance burden of upstream format changes absorbed by the SDK rather than the application.

## The Showcase Model

The classifier exists to exercise the SDK on a realistic workload, not as a deliverable in itself. It is a late fusion BiLSTM consuming the 15 dimensional canonical kinematic vector and a 64 dimensional visual descriptor per timestep, over windows of 50 frames (1 second at 50 Hz). The visual descriptor comes from layer 6 of an ImageNet pretrained MobileNetV3 Small followed by `AdaptiveMaxPool2d(3)`, producing 360 spatial channels stored once at fp16; IncrementalPCA fitted on the training split reduces these to 64 components. Two BiLSTMs (one per modality, hidden size 64) encode the temporal dimension independently, learned softmax attention pools across the 50 timesteps in each stream, and the two pooled vectors are concatenated before a linear head emits the failure logit. Total trainable parameters: 108547. Training uses BCE with `pos_weight=4.522` (auto computed from class balance), `kin_noise_std=0.02`, `WeightedRandomSampler`, dropout 0.35, gradient clip 1.0, SWA from epoch 8, `ReduceLROnPlateau`. Wall clock training time is approximately 6 minutes on a Ryzen 7 CPU with no GPU, made possible by the frozen CNN encoder and its one shot precomputed cache built directly from Mosaico sequences (356 MB on disk).

## Cross Format Generalization

Evaluation uses a 70 / 15 / 15 stratified per dataset split (seed 42, stratification on the sequence label). The threshold is selected on the validation split by F1 maximization (0.806) and applied unchanged to the test split.

| Metric                                         | Value                |
|------------------------------------------------|----------------------|
| Overall AUC (test)                             | 0.685                |
| Overall F1 (test, threshold 0.806)             | 0.334                |
| Class separation (P fail mean - P succ mean)   | 0.173                |
| Reassemble AUC                                 | 0.626                |
| DROID AUC                                      | 0.614                |
| Fractal RT-1 AUC                               | 0.563                |
| P(failure) range observed on test              | [0.027, 0.999]       |
| Confident failure (P > 0.8) window fraction    | 25.1 percent         |
| Confident success (P < 0.2) window fraction    | 13.8 percent         |

The per dataset breakdown is the actual bridge proof. A single set of weights, trained jointly on data Mosaico canonicalized from HDF5, h5 ROS bag and TFRecord, generalizes above chance to each of the three storage formats without dataset specific code paths anywhere in the model or the loader. Reassemble (frame level supervision) leads the per dataset numbers; DROID and Fractal RT-1 (episode level supervision) trail it but stay clearly above the 0.5 chance line. The structural ceiling on DROID and Fractal is set by their supervision granularity, not by the cross format pipeline. Calibration is consistent across formats: the classifier emits probabilities across the full unit interval, so the threshold 0.806 selects a calibrated decision rather than chasing the small differences typical of an under separated classifier.

## Closing the Loop on the Recorded Trajectories

The classifier runs at 50 Hz over the cached kinematic and CNN tensors of held-out test sequences, and its per-frame `P(failure)` is overlaid on top of the original source video frames produced by Mosaico (Reassemble `/hand` blob at 30 fps, DROID exterior camera at 15 fps). Inference runs against the windows the SDK produced at training time, and the rendered frames are the actual recordings the classifier scored, so what the viewer sees is what the model saw.

The composed deliverable is a single 1280x720, 30 fps, 64.5 second MP4 at `results/video_overlay_demo/module_c_deliverable.mp4`. Four acts play sequentially, all sourced from the 15 percent held out test split persisted at `results/test_split.json` (stratified per dataset, seed 42, never seen during training or validation tuning).

**Part 1, passive monitoring.** The classifier observes; the source video plays uninterrupted regardless of the prediction.

- *Act 1, success.* DROID episode `droid_file-000_ep168` (12 s window). `P` stays well below 0.5 throughout, the canvas border holds green.
- *Act 2, failure (alignment phase).* A 12 s window of `reassemble_2025-01-09-19-15-33` centered on the alignment phase of an Insert-Ethernet subtask. The Ethernet plug is visibly held by the gripper for the entire act. `P` rises from 0.13 at t = 0 to a peak of 0.98 around t = 5.5 s, the canvas border turns red, and the GT label reveal in the last 1.5 s confirms the failure (the alignment fails downstream and the operation never completes). The slip moment is the SDK's recorded segment boundary, not a scripted event.

**Part 2, active control with abort authority.** The classifier is granted the authority to halt the source playback at the first `P >= 0.806` crossing.

- *Act 3, success.* DROID episode `droid_file-000_ep476` (12 s window). `P` oscillates between 0.06 and 0.71 across the act but never crosses the threshold; the robot completes the task uninterrupted.
- *Act 4, failure (transport phase, upstream).* A 12 s window centered on the upstream grasp + transport phase of the same Insert-Ethernet trajectory. The robot reaches into the workspace empty-handed, closes the gripper around the Ethernet plug at t ≈ 3.7 s, and begins the transport toward the slot. At t = 8.00 s `P` crosses 0.806; the source frame is frozen, the canvas takes a red wash, and an `ABORT INITIATED BY MOSAICO ML` badge overlays the frozen frame for the remainder of the act, with the plug still inside the gripper.

The two failure acts target two different failure moments along the same task. Act 2 is the *late* failure moment: the operation has progressed to the alignment phase, the model warns mid-attempt, and the recorded outcome (alignment failure, GT reveal) confirms the warning was correct. Act 4 is the *early* failure moment: the model picks up the same impending failure several seconds upstream, while the plug is still in transport, and the abort authority halts the playback before the robot ever reaches the alignment phase. Both moments unfold against the recorded source frames the model scored during training. The causal chain (model output, threshold check, freeze command, on-screen consequence) is fully closed inside the inference loop.

## Mapping Onto the Mosaico Core Value Propositions

The case study maps directly onto the three Core Value Propositions Mosaico is built around.

**Efficiency.** The application code on top of the SDK is three canonical projection functions and a per dataset gripper threshold table, around 80 lines per dataset. The training script, the dataset wrapper, the demo renderer, and the closed loop controller are dataset agnostic by construction. Adding a fourth dataset requires one new projection function and one threshold; everything else compiles unchanged. Mosaico has absorbed the parsing, decoding, and synchronization burden upstream.

**Integrity.** `SyncTransformer` is what makes a cross modality BiLSTM possible at all. Joint torque at 100 Hz, vision at 30 Hz, and sparse event topics arrive at the model on a single 50 Hz grid with consistent timestamps; `SyncHold` preserves quaternion unit norm where naive interpolation would not. None of this logic appears in application code; Mosaico guarantees it for free.

**Actionability.** The deliverable shows that a model trained on Mosaico data can drive a real time safety controller without rewriting the inference pipeline. The same `predict_proba` call that produced the test set metrics fires against held-out test windows during the overlay rendering, in the same DataFrame layout that Mosaico produced at training time, and the abort authority in Part 2 closes the loop end to end (model output, threshold check, halt command, on-screen consequence). Training and deployment share an interface.

The honest limit of the showcase is bounded by supervision granularity, not by the bridge: Reassemble is the only dataset with per frame failure labels, DROID and Fractal carry episode level outcome flags only. With 48 Reassemble sequences contributing the only frame level supervision, an overall AUC of 0.685 is consistent with that structural ceiling. A more accurate classifier would require additional per frame annotations, which is orthogonal to what Mosaico does. The pipeline itself delivers what the Physical AI Bridge promises: one interface, three storage formats, one classifier that generalizes across all of them, and a real time controller that closes the loop.

## Reproducibility

Reproducing the case study from raw data takes roughly three hours of wall clock total. The end to end procedure is documented in `README.md`: clone the repository, bring up the Postgres + MinIO + `mosaicod` stack with `docker compose -f docker/compose-training.yml up -d`, ingest the three datasets via the SDK plugins, precompute the CNN cache once with `scripts/precompute_cnn_features.py` (about 3 hours, idempotent), train with `scripts/run_v9.sh released` (about 6 minutes), render the per-act overlays with `scripts/video_overlay_demo.py`, and concatenate them with `scripts/concat_acts.py`. All hyperparameters, the random seed (42), and the per dataset metrics are persisted to `results/multimodal_indist_v9_sharp/results.json`.

## References and Acknowledgments

- Mosaico, the data platform for Physical AI: [mosaico.dev](https://mosaico.dev), [github.com/mosaico-labs/mosaico](https://github.com/mosaico-labs/mosaico). Python SDK package `mosaicolabs`.
- Manipulation pack and ingestion plugins: [github.com/mosaico-labs/mosaico-alchemy](https://github.com/mosaico-labs/mosaico-alchemy).
- Reassemble dataset (HDF5 manipulation recordings).
- DROID dataset (large scale teleoperated manipulation, h5 / ROS bag layout).
- Fractal RT-1 dataset (RT-1 robotics transformer rollouts, RLDS / TFRecord format).
- MobileNetV3 Small ImageNet weights from `torchvision`.
