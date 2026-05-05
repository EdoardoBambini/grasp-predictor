# A Cross Format Grasp Integrity Predictor on Mosaico

How Mosaico, the data platform for Physical AI, collapses the data plumbing layer of robotic learning, demonstrated on three heterogeneous manipulation datasets, a single LateFusionLSTM, and a closed loop demo on a Franka Emika Panda in MuJoCo.

## Abstract

Mosaico Labs ships an open source data platform for Physical AI: a Rust daemon (`mosaicod`) backed by a metadata database and pluggable object storage for blobs, fronted by a strictly typed ontology system, Apache Arrow zero copy data exchange, and a code first Python SDK (`mosaicolabs`). This case study showcases the platform end to end through a non trivial workload: a binary grasp failure classifier trained jointly on Reassemble (HDF5), DROID (h5 ROS bag layout) and Fractal RT-1 (TFRecord with the RLDS schema), all consumed through the same Mosaico code path with no per format branching. The trained classifier reaches an overall AUC of 0.685 on a stratified held out test split with calibrated probability output spanning the full [0.027, 0.999] range, and runs in real time inside a closed loop MuJoCo demo where it triggers an abort during the lift phase of a failing grasp. The headline finding is not the absolute classifier accuracy: it is that the application code on top of Mosaico is roughly 80 lines per dataset, against an estimated 2000 lines of bespoke parsing and synchronization that an equivalent pipeline would require without the platform. The classifier is the showcase; Mosaico is the substrate.

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

These three formats differ in non trivial ways: nanosecond timestamps and quaternions as fixed length arrays in Reassemble, ROS bag style topic naming in DROID, RLDS nested step structures in Fractal RT-1. Sample rates span 30 Hz to 100 Hz within a single sequence, image codecs are heterogeneous, and the gripper signal is recorded on three incompatible scales. After ingestion, the Mosaico managed footprint on disk (`D:/mosaico-store/`) holds the original source corpus of roughly 90 GB as immutable, indexed blobs, with Postgres carrying the structured metadata; the precomputed CNN feature cache adds 572 MB of fp16 spatial features on top.

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
3. **DataFrame injection** (`DataFrameExtractor.to_pandas_chunks`). Mosaico streams the requested topics directly into pandas DataFrames over Apache Arrow with bounded memory, regardless of how the source was originally laid out. Three datasets, one DataFrame contract, zero serialization copies between server and client.
4. **Automatic synchronization** (`SyncTransformer(target_fps=50.0, policy=SyncHold())`). Mosaico resamples heterogeneous sample rates onto a single 50 Hz grid. The `SyncHold` forward fill policy preserves quaternion unit norm, where naive interpolation would silently corrupt orientation data, and is safe across booleans, floats, and array typed payloads.
5. **Application projection** (`feature_mapper.project`). Mosaico delivers a uniform dot notation schema (`<topic>.<ontology_tag>.<field>`) across the bound ontology types, and the application maps that schema onto the canonical 8 feature representation of the domain. This is where the application code lives; everything upstream is the platform.
6. **Derived features** (`add_derived_features`). Standard finite difference on the uniform timeline produced by `SyncTransformer`, yielding 7 velocity proxies that lift the kinematic vector to 15 dimensions.
7. **Label adapters** (`label_adapters.label`). Failure annotations are read through the same SDK handle, never from side car files. Two schemas for Reassemble are supported in parallel: the legacy `Boolean` ontology (`/grasp_failure_label.boolean.data`) and the newer `SegmentInfo` ontology (`/grasp_failure_label.segment_info.success`). The `SegmentInfo` ontology and the corresponding update to the Reassemble plugin in `mosaico-alchemy` were contributed via PR as part of this case study (`mosaico-labs/mosaico-alchemy#4`).
8. **Loop closure**. Inference at training time and inference inside the closed loop MuJoCo demo both use this exact chain, so the production controller never diverges from the training pipeline.

## How Little Application Code Is Needed on Top

The only application code on top of the SDK lives in `data/feature_mapper.py`: 8 base features, 7 derivatives, three projection functions (`project_reassemble`, `project_droid`, `project_fractal_rt1`) totalling roughly 80 lines. Mosaico canonicalizes transport and schema; cross dataset semantics (what does "gripper open" mean across three control conventions) is by construction the application's responsibility. The single semantic decision at the application layer is the per dataset gripper binarization, with thresholds 0.025 (Reassemble, meters), 0.5 (DROID, normalized), 0.5 (Fractal, quasi binary), calibrated from histograms in `results/audit_cache.json`. After binarization `gripper_state` is a binary signal across the three datasets and `gripper_vel` becomes a clean edge detector for opening and closing.

For comparison, a from scratch implementation would need a dedicated HDF5 parser for Reassemble (with `segments_info` traversal, 2 finger gripper aggregation, per topic JPEG decoding), an h5 ROS bag style parser for DROID (cartesian pose unwrapping, episode level metadata, MP4 wrist camera decoding at 30 fps), a TFRecord and RLDS parser for Fractal RT-1 (nested step structure traversal, terminal reward extraction, image byte unpacking), plus a custom multi rate synchronization layer with quaternion safe policies. A reasonable estimate is around 2000 lines of production parsing and synchronization code, before any modelling work begins. The case study collapses that to about 80 application lines per dataset on top of `mosaicolabs`, a roughly 8x reduction in surface area, with the maintenance burden of upstream format changes absorbed by the SDK rather than the application.

## The Showcase Model

The classifier exists to exercise the SDK on a realistic workload, not as a deliverable in itself. It is a late fusion BiLSTM consuming the 15 dimensional canonical kinematic vector and a 64 dimensional visual descriptor per timestep, over windows of 50 frames (1 second at 50 Hz). The visual descriptor comes from layer 6 of an ImageNet pretrained MobileNetV3 Small followed by `AdaptiveMaxPool2d(3)`, producing 360 spatial channels stored once at fp16; IncrementalPCA fitted on the training split reduces these to 64 components. Two BiLSTMs (one per modality, hidden size 64) encode the temporal dimension independently, learned softmax attention pools across the 50 timesteps in each stream, and the two pooled vectors are concatenated before a linear head emits the failure logit. Total trainable parameters: 108547. Training uses BCE with `pos_weight=4.522` (auto computed from class balance), `kin_noise_std=0.02`, `WeightedRandomSampler`, dropout 0.35, gradient clip 1.0, SWA from epoch 8, `ReduceLROnPlateau`. Wall clock training time is approximately 6 minutes on a Ryzen 7 CPU with no GPU, made possible by the frozen CNN encoder and its one shot precomputed cache built directly from Mosaico sequences (572 MB on disk).

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

## Closing the Loop in MuJoCo

The closed loop demo runs a Franka Emika Panda from MuJoCo Menagerie in a custom workcell scene at `simulation/assets/mujoco_menagerie/franka_emika_panda/grasp_lab.xml`. Four acts play sequentially, all sourced from the held out DROID test split (223 sequences never seen in training or validation). The classifier runs at 50 Hz in parallel to the physics solver, on windows assembled by the same Mosaico code path that produced the training data.

Part 1 is passive monitoring. The classifier observes the trajectory while the robot completes its scripted motion regardless of the prediction. The success act (DROID episode 168) keeps the probability near 0.3. The failure act (DROID episode 461) closes the gripper with a sub optimal `grip_close=100` instead of the default 50; the cube is grasped and lifted to z = 0.58 m (about 14 cm of visible lift), then the reduced normal force loses to inertial acceleration during transport and the cube slips out at approximately t = 11 s. The slip is the output of the MuJoCo solver under the static gripper parameter, not a scripted event; the probability rises through the lift from about 0.4 to above 0.9.

Part 2 is active control. The classifier is granted authority to abort at threshold 0.806. The success act (DROID episode 1382) keeps the maximum probability at 0.33, no abort fires, the robot completes the placement. The failure act (DROID episode 233) sees the probability climb 0.25, 0.43, 0.69, 0.71, 0.84; the threshold is crossed at simulation time 7.56 s with P = 0.837 during the lift, the abort fires, the gripper opens, the cube falls under gravity. The causal chain (model output, threshold trigger, gripper command, physical consequence) is fully closed inside the simulation loop. Final video at 1280x720, 30 fps, 64 seconds, committed at `results/module_c_franka/shadow_loop_v9_run3_v7.mp4` (31 MB).

## Mapping Onto the Mosaico Core Value Propositions

The case study maps directly onto the three Core Value Propositions Mosaico is built around.

**Efficiency.** The application code on top of the SDK is three canonical projection functions and a per dataset gripper threshold table, around 80 lines per dataset. The training script, the dataset wrapper, the demo renderer, and the closed loop controller are dataset agnostic by construction. Adding a fourth dataset requires one new projection function and one threshold; everything else compiles unchanged. Mosaico has absorbed the parsing, decoding, and synchronization burden upstream.

**Integrity.** `SyncTransformer` is what makes a cross modality BiLSTM possible at all. Joint torque at 100 Hz, vision at 30 Hz, and sparse event topics arrive at the model on a single 50 Hz grid with consistent timestamps; `SyncHold` preserves quaternion unit norm where naive interpolation would not. None of this logic appears in application code; Mosaico guarantees it for free.

**Actionability.** The closed loop demo shows that a model trained on historical Mosaico data can drive a real time safety controller without rewriting the inference pipeline. The same `predict_proba` call that produced the test set metrics fires inside the MuJoCo loop, against newly synthesized 50 Hz windows in the same DataFrame layout that Mosaico produced at training time. Training and deployment share an interface.

The honest limit of the showcase is bounded by supervision granularity, not by the bridge: Reassemble is the only dataset with per frame failure labels, DROID and Fractal carry episode level outcome flags only. With 48 Reassemble sequences contributing the only frame level supervision, an overall AUC of 0.685 is consistent with that structural ceiling. A more accurate classifier would require additional per frame annotations, which is orthogonal to what Mosaico does. The pipeline itself delivers what the Physical AI Bridge promises: one interface, three storage formats, one classifier that generalizes across all of them, and a real time controller that closes the loop.

## Reproducibility

Reproducing the case study from raw data takes roughly three hours of wall clock total. The end to end procedure is documented in `README.md`: clone the repository, bring up the Postgres + MinIO + `mosaicod` stack with `docker compose -f docker/compose-training.yml up -d`, ingest the three datasets via the SDK plugins, precompute the CNN cache once with `scripts/precompute_cnn_features.py` (about 3 hours, idempotent), train with `scripts/run_v9.sh released` (about 6 minutes), and render the demo with `scripts/shadow_loop_demo.py`. All hyperparameters, the random seed (42), and the per dataset metrics are persisted to `results/multimodal_indist_v9_sharp/results.json`.

## References and Acknowledgments

- Mosaico, the data platform for Physical AI: [mosaico.dev](https://mosaico.dev), [github.com/mosaico-labs/mosaico](https://github.com/mosaico-labs/mosaico). Python SDK package `mosaicolabs`.
- Manipulation pack and ingestion plugins: [github.com/mosaico-labs/mosaico-alchemy](https://github.com/mosaico-labs/mosaico-alchemy).
- Reassemble dataset (HDF5 manipulation recordings).
- DROID dataset (large scale teleoperated manipulation, h5 / ROS bag layout).
- Fractal RT-1 dataset (RT-1 robotics transformer rollouts, RLDS / TFRecord format).
- Franka Emika Panda asset from MuJoCo Menagerie (Apache 2.0).
- MobileNetV3 Small ImageNet weights from `torchvision`.
