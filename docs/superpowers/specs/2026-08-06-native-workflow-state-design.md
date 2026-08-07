# CLI-First Run Workflow and Native Capture-State Design

**Date:** 2026-08-06

**Status:** Implemented and verified

**Project:** Adaptive Edge Perception (working title)

## 1. Outcome

The product is a CLI-first research tool with an optional native GUI. A researcher can supply an existing video, materialize a public video, or optionally capture a camera source; then configure a run, execute it, inspect its results, and compare runs without opening a window. The native application presents the same workflow visually:

```text
Source
  -> Regions of interest
  -> Run configuration
  -> Run
  -> Metrics and artifacts
```

This is the causal order of the work, not a wizard or a progress stepper. In the GUI, every part remains visible in one Qt Widgets window. The GUI is a thin native client over the same capture service, configuration, worker, result projection, and artifact contracts used by the CLI.

The immediate correction is narrow: a successful camera capture must look successful, its durable destination must be controllable from CLI or GUI, and each disabled primary action must be explained by persistent state or a readiness reason.

## 2. Industry-Language Basis

The vocabulary follows recurring concepts in established ML, CV, and RL tools:

- MLflow and Weights & Biases use a **run** for one execution and organize its **configuration**, **metrics**, and **artifacts**. A project or experiment is a parent collection, which this single-run utility does not yet implement. ([MLflow Tracking](https://mlflow.org/docs/latest/tracking/), [W&B Runs](https://docs.wandb.ai/models/runs))
- ClearML similarly separates execution configuration from execution output and gives a task one explicit lifecycle. ([ClearML Tasks](https://clear.ml/docs/latest/docs/fundamentals/task/))
- FiftyOne uses **sample**, **frame**, **label**, **detection**, and **view** for dataset inspection. Our current input is one referenced video rather than a managed dataset, so **source** is more precise than dataset or sample. ([FiftyOne User Guide](https://docs.voxel51.com/user_guide/))
- DeepStream describes an inference pipeline in the order source/capture, inference, optional tracking, and output. We preserve those boundaries without pretending the current GUI is a pipeline editor. ([DeepStream Flow API](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_intro_to_flow_api.html))
- Gymnasium and RLlib reserve **environment**, **observation**, **action**, **reward**, **episode**, **policy**, **train**, **evaluate**, and **checkpoint** for an agent-environment learning loop. The current deterministic CV run is not labeled with those terms. ([Gymnasium Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/), [RLlib Environments](https://docs.ray.io/en/latest/rllib/rllib-env.html))

### Decisions

- Use **Run**, not Execution, Task, Trial, Job, or Experiment, for one local invocation.
- Use **Run configuration**, not Experiment configuration, because there is no parent experiment entity yet.
- Use **Metrics** only for measured numeric values with units.
- Use **Artifacts** only for persisted files produced or attached to a run.
- Use **Source** for the validated input-video reference. Camera captures additionally carry checksum-backed acquisition provenance.
- Use **Region of interest (ROI)** for an operator-defined area of the source frame.
- Use **Detection** only for detector output; an ROI is not a detection.
- Use **Preview** for the live camera image; reserve View for a future query-derived dataset view.
- Keep **Detector** and **Compute device** as explicit run parameters.

## 3. CLI-First Product Contract

The GUI must never be the sole path to a durable operation. Visual preview and drag-to-draw ROIs are conveniences; camera capture, coordinate-based ROIs, run execution, result inspection, and comparison remain headless operations.

The runner consumes a validated local video reference, not a provider page or mutable network stream. A YouTube or similar public video works after it is materialized to a local file by an external downloader. Provider-specific download APIs, credentials, and licensing policy remain outside this checkpoint; keeping that boundary preserves offline execution and reproducible local inputs.

### 3.1 Command surface

The checkpoint exposes or adds these commands:

```text
edge-perception camera list
edge-perception camera capture --device ID --duration SECONDS \
  [--output VIDEO] [--width PX] [--height PX] [--fps FPS] [--strict]

edge-perception run INPUT --output DIRECTORY [--detector ID] [--crop ID:X,Y,W,H ...]
edge-perception run --config RUN_CONFIG
edge-perception inspect RUN_DIRECTORY
edge-perception compare LEFT_RUN RIGHT_RUN
edge-perception gui [--run RUN_DIRECTORY]
```

`camera capture` is bounded by `--duration` in this checkpoint so it is deterministic, scriptable, and safe in a headless process. It uses the same format selection, strict validation, private staging, hashing, and atomic publication as the GUI. It prints the finalized path, checksum, capture request, applied camera format, and recorded format.

`inspect` renders the same canonical run projection as the native Run results widget in a terminal-readable form. The canonical JSON and JSONL artifacts remain the machine-readable interface; a future `--json` stdout mode is not required for this slice.

`run_checkpoint` remains the shared execution authority. `edge-perception run` invokes it in-process. The native GUI publishes a Run configuration and launches the existing internal JSONL worker, which invokes the same function while retaining its progress and private cancel-file protocol. No GUI-only runner exists.

For a direct CLI run, the first `Ctrl+C` sets an in-process cancellation event passed to `run_checkpoint` as `cancel_requested`; it does not raise `KeyboardInterrupt` into the runner. Once observed, the runner finalizes canonical `cancelled` artifacts and the CLI exits with status 130. GUI-worker cancellation behavior remains unchanged. Process exit codes and partial-artifact guarantees are documented and tested.

### 3.2 Shared boundaries

| Operation | CLI | Native GUI | Shared contract |
| --- | --- | --- | --- |
| Camera discovery | `camera list` | Device selector | Camera device and format records |
| Camera capture | `camera capture` | Camera acquisition controls | `CaptureRequest`, capture service, `CaptureResult` |
| ROI definition | repeatable `--crop` or Run configuration | visual ROI editor | `Region` in source-frame coordinates |
| Run configuration | flags or `--config` | Run configuration form | `RunConfig` |
| Run execution | `run` | Run / Cancel run | worker protocol and canonical artifacts |
| Result inspection | `inspect` | Run results | canonical run projection |
| Semantic comparison | `compare` | CLI only in this checkpoint | comparison report |

GUI parity is one-way: every durable GUI operation has a CLI or shared-library path, while the CLI may expose research operations that the minimal GUI does not. The GUI cannot invent a second schema, naming system, default, detector-loading path, or output format.

Camera contracts, format selection, validation, and the QtMultimedia capture backend move outside `edge_perception.gui`. One shared backend owns camera discovery, recording, validation, hashing, and atomic publication; this checkpoint does not introduce a second recorder implementation.

The GUI supplies a visual video output. The CLI drives the same asynchronous backend with a `QCoreApplication`/`QEventLoop` and `QTimer`, without importing QtWidgets or creating a window. Camera and GUI command handlers import PySide6 lazily.

The optional `camera` and `gui` extras each install the same compatible PySide6 dependency. The base package remains sufficient for existing-video runs, inspection, and comparison, and ordinary CLI startup imports neither optional surface. Missing optional support produces a concise install hint rather than breaking unrelated commands at import time.

## 4. Native Information Architecture

The left side remains the visual preview and ROI editor. The right side is ordered into five compact sections.

### 4.1 Source

The researcher chooses a source acquisition mode:

- **Video file** selects a local source reference and verifies that a preview frame can be decoded.
- **Camera** acquires a new source through preview, recording, strict media validation, hashing, and atomic publication.

The section always exposes the active source path and available measured media metadata. Camera-only controls appear under **Camera acquisition**. Source mode remains **Camera** after a capture is finalized. The mode selects acquisition controls; changing it alone does not replace the active source.

### 4.2 Regions of interest

The researcher adds, selects, edits, and removes ROIs over the source frame. The section reports the exact ROI count and source-frame coordinates. It does not call regions labels, predictions, detections, or views.

### 4.3 Run configuration

The researcher selects the detector adapter, compute device, confidence threshold, frame limit, warm-up iterations, annotation interval, and output directory. Before the first launch, Run configuration and CLI command are blank. After successful configuration publication, their resolved values remain visible and read-only.

### 4.4 Run

One **Run** action starts the isolated worker. **Cancel run** is available only while a run is active. A factual status line reports readiness, progress, terminal state, or an exact validation failure.

### 4.5 Run results

Results appear only after canonical run artifacts load. Their information order is:

1. **Overview** — status, run directory, error when present;
2. **Metrics** — frames, inference count, latency quantiles, memory, and units;
3. **Run configuration** — detector revision, compute device, confidence threshold, and ROIs;
4. **Source provenance** — source path, dimensions, frame rate, and camera-capture metadata when present;
5. **Annotations** — annotated frames produced as run artifacts.

This remains one dense widget. These names are conceptual information groups, not a requirement to add tabs, an artifact browser, or result fields that are not already present in the canonical projection. Browsing manifests, configuration files, and JSONL files is outside this checkpoint.

## 5. State Model

The design does not use `SOURCE -> REGIONS -> EXECUTION -> RESULTS` as a stage bar. Source readiness, ROI count, configuration validity, and run status are simultaneous facts; only a run has a lifecycle.

The visible states are separated by owner:

| Owner | Visible states | Meaning |
| --- | --- | --- |
| Source | `No source`, `Ready` | Whether a validated source reference is active. |
| Camera acquisition | `Idle`, `Previewing`, `Recording`, `Finalizing`, `Finalized`, `Failed` | State of the optional camera operation. |
| Run readiness | `Not ready: <reason>`, `Ready` | Whether the current controls can produce a valid fresh run configuration. |
| Run | `Not started`, `Running`, `Cancelling`, `Completed`, `Failed`, `Cancelled` | Lifecycle of the current or most recently loaded run. |

The canonical artifact values remain `complete`, `failed`, and `cancelled`. The GUI maps `complete` to the conventional human-facing state **Completed** without changing the stored contract.

There is no separate Results lifecycle. Results are either absent or loaded from a canonical terminal run.

Example status text:

```text
Source status: Ready
Acquisition status: Finalized
Run readiness: Ready
Run status: Not started

Run status: Running — frame 143 / 500
Run status: Completed — C:\...\run
```

When the Run action is disabled, the status reports the first actionable cause, for example `Not ready: select a source` or `Not ready: choose an empty output directory`.

## 6. Exact Interface Language

| Current surface | Proposed label |
| --- | --- |
| `Source mode` | `Source mode` |
| `Video` | `Source path` |
| `Dimensions` | `Frame size` |
| `Camera capture` | `Camera acquisition` |
| `Camera` | `Device` within Camera acquisition |
| `FPS` | `Frame rate` |
| `Strict` | `Require specified frame size and rate` |
| `Selected` | `Applied camera format` |
| `Requested` | `Capture request` |
| `Actual` | `Recorded format` |
| new camera path field | `Capture destination` |
| `Regions` | `Regions of interest (ROIs)` |
| `New` | `Add ROI` |
| `Delete` | `Remove ROI` |
| `Device` in run parameters | `Compute device` |
| `Threshold` | `Confidence threshold` |
| `Warm-up runs` | `Warm-up iterations` |
| `Annotate every` | `Annotation interval (frames)` |
| `Output` | `Output directory` |
| `Experiment config` | `Run configuration` |
| `CLI` | `CLI command` |
| `Run` action | `Run` |
| `Cancel` action | `Cancel run` |
| `Results` | `Run results` |

Camera actions are **Start preview**, **Start recording**, and **Stop recording**. The explicit verbs avoid confusing capture stop with run cancellation.

The exact-format checkbox has the tooltip: `Reject the capture if any specified width, height, or frame-rate value is not met.` It does not claim an exact codec, container, or pixel format.

Status-bar messages remain concise and factual:

```text
Preview started: EMEET SmartCam Nova 4K
Recording started: C:\...\capture.mp4
Capture finalized: C:\...\capture.mp4
Run started: C:\...\run
Run completed: C:\...\run
Run failed: <exact error>
Run cancelled: C:\...\run
```

The application does not address the researcher conversationally or explain a state transition with tutorial copy.

## 7. User Flow

### 7.1 Headless reference workflow

```text
# Discover the stable camera ID and supported frame sizes/rates.
edge-perception camera list

# Acquire a deterministic, checksum-backed source without opening a window.
edge-perception camera capture \
  --device CAMERA_ID \
  --output fixtures/private/reference.mp4 \
  --duration 25 \
  --width 3840 --height 2160 --fps 30 --strict

# Execute the same run contract used by the GUI.
edge-perception run fixtures/private/reference.mp4 \
  --output runs/reference \
  --detector dfine-nano-coco \
  --crop left:0,0,1920,2160 \
  --crop right:1920,0,1920,2160

# Inspect and compare canonical run records.
edge-perception inspect runs/reference
edge-perception compare runs/gpu runs/cpu
```

The same workflow can use an externally produced video or a checked-in Run configuration. Camera acquisition is optional; no GUI is required at any point.

### 7.2 Native GUI with an existing video

1. Select **Video file** under Source mode.
2. Choose **Browse…** and select a local video.
3. The application validates and previews the file, then reports Source status **Ready** with the absolute path and measured frame metadata.
4. Define any required ROIs.
5. Review the Run configuration and resolve any visible readiness error, including choosing an empty Output directory.
6. Select **Run**.
7. Monitor Run status, then inspect Metrics, Run configuration, Source provenance, and Annotations loaded from the run directory.

### 7.3 Native GUI camera acquisition

1. Select **Camera** under Source mode.
2. Select the device and optional width, height, frame-rate, exact-format, and capture-destination constraints.
3. Select **Start preview**. After preview startup succeeds, the previous source and its ROIs are invalidated, Source status becomes **No source**, and Acquisition status becomes **Previewing**.
4. Select **Start recording**, then **Stop recording**.
5. The application reports **Finalizing** while it probes, validates, hashes, and publishes the recording.
6. On success, Source mode remains **Camera**, Acquisition status becomes **Finalized**, Source status becomes **Ready**, and the finalized absolute path plus recorded metadata remain visible.
7. Define ROIs, review the Run configuration, select **Run**, and inspect the canonical results exactly as for an existing video.

Starting another preview is an explicit source-replacement action. The finalized source remains runnable until a new preview starts successfully or another video validates successfully.

## 8. Control Behavior

Controls are enabled individually from the states above; an entire group does not become an unexplained disabled block.

- **Browse…** for a video is available in Video-file mode when no run or recording is active.
- Changing Source mode alone changes the available acquisition controls and preserves the active source.
- Camera device and constraints are editable when Camera is selected and acquisition is Idle, Finalized, or Failed.
- **Start preview** is enabled when a camera is available and no recording, finalization, or run is active.
- **Start recording** is enabled only while previewing.
- **Stop recording** is enabled only while recording.
- Source, ROI, detector, configuration, and capture-destination mutation is locked while a run is active.
- **Run** is enabled only when run readiness is Ready.
- **Cancel run** is enabled only while a run is active.
- Selecting **Cancel run** immediately changes Run status to **Cancelling** and disables the action until the worker reaches a terminal state.
- Starting a run hides previously loaded results so they cannot appear to describe the active run.
- Completed and cancelled runs load their canonical results. A failed run leaves results hidden and preserves its error in Run status.
- Terminal run cleanup restores the controls appropriate to the still-visible source and acquisition states.

## 9. Capture Destination

Camera acquisition adds:

```text
Capture destination  [ Automatic                              ] [ Browse… ]
```

Behavior:

1. Blank in the GUI or omitted `--output` in the CLI passes `None` to the shared capture backend, which creates the same private, unique, app-local destination.
2. Browse opens a native save-file dialog and writes the selected path into the field.
3. A typed, selected, or CLI-supplied path is resolved to an absolute path before recording and passed to the shared capture backend.
4. Existing destinations are rejected before mutation; captures are never overwritten.
5. Parent-directory creation, private same-filesystem staging, validation, hashing, and atomic publication remain owned by the shared capture service; the native controller only adapts its lifecycle to Qt signals.
6. The GUI keeps the finalized absolute path visible, and the CLI always prints it. The path and SHA-256 remain part of `CaptureResult`, the Run configuration, and the run manifest.
7. The actual container and codec reported by PyAV remain authoritative. The UI does not infer successful encoding from a filename suffix.

The field is a production input, not a demo control. It supports durable local data acquisition, reproducible fixture creation, and ignored private reference sources while retaining an automatic default.

## 10. Root Cause Addressed

The current capture path is technically correct but visually ambiguous. When recording finalization succeeds, the GUI silently changes Source mode from Camera to Video file, loads the finalized frame, and disables the entire camera group. A responsive, successful state therefore looks like a frozen application.

The design removes that silent mode change, exposes acquisition and source state separately, and keeps the next valid actions visible.

## 11. Data and Reproducibility Boundary

```text
CLI or native GUI
  -> camera request or existing file
  -> validated source reference
     + checksum-backed capture provenance when acquired by camera
  -> source-frame ROIs
  -> immutable run configuration
  -> isolated detector worker
  -> canonical metrics and run artifacts
  -> read-only run results
```

The GUI owns transient presentation state only. The CLI owns no alternate domain model. `CaptureResult`, `RunConfig`, the worker protocol, the canonical run projection, and canonical run artifacts remain the reproducibility boundary. Detector adapters remain plug-and-play; no detector model is packaged with the tool.

## 12. Failure Behavior

- No camera: Acquisition status **Idle**, no device selected, and Video file remains usable.
- Preview failure before Previewing: Acquisition status **Failed**; the prior source and ROIs remain active.
- Recording failure after a preview started: Acquisition status **Failed** and Source status **No source**; no incomplete path is presented as a source.
- Finalization or strict-validation failure: Acquisition status **Failed**, Source status remains **No source**, and the exact controller error is shown.
- Existing capture destination: recording does not start and the existing file is untouched.
- Capture published with cleanup diagnostics: the finalized source remains usable and the cleanup diagnostic is shown without relabeling the capture as failed.
- Active run: source, ROI, detector, run-configuration, and destination mutation remains locked until terminal cleanup.
- Run failure: Run status becomes **Failed** and the exact error remains visible. This checkpoint does not automatically load artifacts from a failed worker; a canonical failed run remains inspectable when explicitly loaded by run directory.

## 13. Testing

The default automated suite will prove:

1. ordinary CLI startup and `run`, `inspect`, and `compare` remain camera-, GUI-, model-, and network-lazy;
2. `camera list` and `camera capture` import multimedia support lazily and never construct a Widgets application or window;
3. CLI and GUI capture paths use the same format selection, strict validation, publication, and `CaptureResult` contract;
4. `camera capture --duration` finalizes deterministically, rejects overwrite, and reports provenance through fake devices in the default suite;
5. direct CLI cancellation reaches the canonical `cancelled` terminal contract;
6. `inspect` and Run results consume the same canonical run projection;
7. a GUI-published Run configuration executes unchanged through `edge-perception run --config`;
8. source mode selects the correct file or camera controls without replacing an active source by itself;
9. run readiness exposes a factual reason whenever Run is disabled;
10. destination blank passes `None`, while an explicit path is resolved and passed unchanged to the capture service;
11. Browse cancellation does not mutate either source or capture destination;
12. capture finalization leaves Source mode on Camera;
13. finalized acquisition and source states are **Finalized** and **Ready**;
14. finalized path, requested metadata, recorded metadata, and checksum-backed `CaptureResult` remain attached to the Run configuration;
15. Start preview is available after finalization, while Start recording and Stop recording are not;
16. a new preview and an active run produce the exact control locks defined above;
17. run states cover readiness, progress, cancelling, completion, failure, cancellation, and terminal-run loading;
18. existing result fields use the agreed run, metric, provenance, and artifact terminology; and
19. no browser, server, database, model load, or physical camera is required by the default suite.

The required end-to-end acceptance uses a local materialized public or synthetic video and does not require a physical camera. When the EMEET is available, an optional Windows hardware pass verifies CLI and GUI capture, automatic and explicit destinations, equivalent capture provenance, strict behavior, and the observed acquisition/source/run states; hardware availability does not gate the checkpoint.

## 14. Acceptance Criteria

The slice is complete when:

- a researcher can complete camera discovery, timed capture, ROI configuration, detector selection, run execution, result inspection, and run comparison from the CLI without opening the GUI;
- the complete run/inspect/compare proof works with an existing local video and requires no physical camera;
- every durable native-GUI operation resolves to the same shared contract and canonical artifacts as its CLI counterpart;
- installing and using the base run/inspect/compare path does not require the optional native multimedia or GUI dependency;
- within one event-loop turn after `recordingFinished`, the finalized path and recorded metadata are visible, Source is Ready, Acquisition is Finalized, Start preview is enabled, recording actions are disabled, and run readiness is recomputed;
- no automatic source-mode change occurs after finalization;
- a disabled Run action has a textual readiness reason, and camera-button availability follows directly from the persistent Acquisition status;
- an explicit private reference path can be selected before recording;
- the automatic capture destination remains the default;
- the finalized capture can immediately feed the existing ROI, run-configuration, worker, CLI, and results workflow;
- existing run results use status, metric, configuration/provenance, and artifact language without adding a dashboard or browser; and
- offline tests, Ruff, mypy, package build, and wheel-content checks pass.

## 15. Scope Guardrail

This design clarifies and exposes the production workflow already present. It does not add a provider-specific video downloader, wizard, browser UI, database, experiment tracker, queue, GUI comparison dashboard, detector accuracy evaluation, live inference, tracking, policy training, or reinforcement-learning controls.

Future RL work should introduce a separate, explicit environment contract with observation and action spaces, reset/step behavior, reward, episode boundaries, seeding, policy training, evaluation, and checkpoints. Those terms are intentionally absent from this checkpoint until those objects exist.
