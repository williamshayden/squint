# CLI-First Research UI Design

**Date:** 2026-08-06

**Status:** Draft for user review

**Project:** Adaptive Edge Perception (working title)

## 1. Outcome

Adaptive Edge Perception will remain a headless, configuration-driven research tool with an optional local graphical interface for work that is inherently visual.

The command line, typed Python API, versioned configuration, and run artifacts are the product contract. The GUI is a thin local adapter over that contract. It must not introduce a second experiment engine, a demo-only workflow, or results that cannot be reproduced without the GUI.

In simple terms: a researcher can run everything from a terminal, but can open a small old-school utility when they need to see a camera, draw a crop, or inspect detections.

## 2. Intended Users

The initial users are computer-vision, edge-ML, and reinforcement-learning researchers who need to:

- run deterministic experiments locally or in automation;
- use constrained CPU/GPU hardware honestly;
- preserve the exact source, model, configuration, environment, and measurements;
- define spatial regions visually without hand-calculating pixels;
- inspect whether detections and coordinate mappings are plausible;
- consume machine-readable records in their own analysis or training code; and
- extend detector and, later, strategy interfaces without modifying the UI runtime.

The first checkpoint does not claim to provide an RL environment or learned policy. It preserves boundaries that future Gymnasium environments and strategies can use.

## 3. Goals

1. Keep the CLI and typed Python API first-class and fully usable without GUI dependencies.
2. Add a minimal local GUI for camera/file preview, recording, region definition, run launch, and artifact inspection.
3. Use one versioned experiment configuration across the CLI, Python API, and GUI.
4. Save every recording as an ordinary immutable video input before inference begins.
5. Record requested and actual capture properties, including width, height, and FPS.
6. Make every inference run launched from the GUI reproducible through a displayed CLI command.
7. Preserve JSON/JSONL artifacts as the source of truth and generate an offline visual report from them.
8. Keep the UI dependency set optional and avoid a Node.js build toolchain.
9. Support a production-quality local workflow on Windows and Linux first.

## 4. Non-Goals

The initial UI will not include:

- accounts, authentication systems, cloud storage, or remote collaboration;
- a multi-user server or network deployment;
- a database or a second results store;
- a general dashboard, experiment tracker, or notebook replacement;
- live object detection, live tracking, or live policy decisions;
- a video editor or labeling suite;
- parallel GPU job scheduling;
- a plugin marketplace;
- RL controls before strategy and environment contracts exist in the core package;
- bundled detector weights; or
- a native desktop application bundle.

## 5. Product Interfaces

### 5.1 CLI

The canonical interfaces are:

```text
edge-perception run --config experiment.json
edge-perception compare RUN_A RUN_B
edge-perception report RUN_DIR [--open]
edge-perception ui [--port PORT] [--no-open]
```

Existing explicit `run` flags remain available for one-off experiments. A config file is the preferred reproducible workflow. If both are supplied, explicitly provided flags override config values and the resolved configuration is written to the run manifest.

`edge-perception ui` binds to `127.0.0.1` by default, starts the local interface, prints its URL, and opens the default browser unless `--no-open` is supplied.

### 5.2 Python API

The typed Python application layer accepts the same validated experiment configuration and exposes run, comparison, report, capability-discovery, and progress/cancellation boundaries. Neither CLI parsing nor HTTP concepts appear in this layer.

### 5.3 Optional GUI Package

The core package remains importable and testable without GUI dependencies. The local UI is installed as an optional extra:

```text
pip install adaptive-edge-perception[ui]
```

Invoking `edge-perception ui` without the extra prints the exact installation command and exits cleanly.

The optional runtime uses a small local ASGI server and bundled vanilla HTML, CSS, and JavaScript. It has no frontend compilation step and loads no remote assets.

## 6. Configuration Contracts

### 6.1 Experiment Configuration

`ExperimentConfig` is a versioned, JSON-native document containing:

- source video path;
- optional capture-provenance record for UI-acquired sources;
- output directory;
- detector ID and detector-owned settings;
- execution device;
- score threshold;
- zero or more named source-frame regions;
- warm-up count;
- maximum frame count;
- annotation interval; and
- schema version.

The resolved configuration is validated before model loading and written into the run manifest. When the source was captured by the UI, the manifest copies the requested and actual `CaptureResult` fields so a later CLI run retains acquisition provenance. Region coordinates are integer source-frame pixels, independent of preview scaling.

### 6.2 Capture Request

Camera acquisition is a separate step from inference. `CaptureRequest` contains:

- browser-provided device ID and human-readable label;
- optional requested width;
- optional requested height;
- optional requested FPS;
- strict-mode boolean; and
- audio-enabled boolean, defaulting to `false`.

Width, height, and FPS are independent optional parameters. No particular resolution or frame rate is required by the product.

The UI offers `Auto`, common presets, and custom values. In normal mode the browser treats supplied values as preferences. In strict mode the browser requests exact constraints for each supplied field and the application rejects the capture before recording if any supplied width, height, or FPS differs from the negotiated value.

`CaptureResult` records:

- requested width, height, and FPS;
- actual negotiated width, height, and FPS;
- device label;
- container and codec;
- duration;
- audio presence;
- file size;
- final path; and
- SHA-256 digest.

The reference Eyes and Stopwatch experiment requests and strictly verifies `3840 × 2160 @ 30 FPS`. That is a property of the reference experiment, not a product requirement. Other experiments may use any decodable resolution, FPS, aspect ratio, prerecorded file, or synthetic video.

## 7. Architecture

The system has six bounded components:

1. **Core contracts and application services** validate configurations, run experiments, compare runs, expose capabilities, report progress, and support graceful cancellation.
2. **Detector adapters** own model loading and prediction while returning backend-neutral records. The UI consumes detector descriptors from the core and does not hard-code model logic.
3. **CLI adapter** maps arguments and config files into application calls and renders concise terminal output.
4. **Local UI server** validates local requests, manages one active worker process, serves bundled assets, and exposes artifacts. It does not perform inference itself.
5. **Browser client** owns camera permission, raw preview, capture constraints, region drawing, compact run controls, and artifact presentation.
6. **Report renderer** converts a completed run directory into an offline `report.html` that contains configuration and measurements inline and references diagnostic images within the run directory.

The local UI launches inference in an isolated worker process using the resolved configuration. The worker calls the same application service as the CLI. This isolates CUDA/model memory from the UI server and makes CLI/GUI parity testable.

Only one active experiment is permitted initially. The server rejects a second launch with a clear conflict response rather than allowing accidental GPU contention.

## 8. Camera and Video Flow

1. The browser requests camera permission only after an explicit user action.
2. The user selects a camera and optional width, height, and FPS.
3. The browser displays the actual negotiated mode before recording.
4. The raw camera preview contains no model inference.
5. Recording starts and stops explicitly; duration is not hard-coded.
6. Audio is omitted unless the user explicitly enables it.
7. `MediaRecorder` emits ordered chunks to the local server so long or high-resolution captures are not retained as one browser-memory blob.
8. The server writes chunks to a private temporary file, validates ordering and size limits, and atomically publishes the final video after successful finalization.
9. PyAV validates that the captured file is decodable and records its actual stream metadata and checksum.
10. The completed capture becomes an ordinary source video for `ExperimentConfig`.

Closing the browser during recording stops its chunk stream. The server marks the capture abandoned after a bounded missing-heartbeat interval and removes only the incomplete temporary capture. Completed captures are never silently overwritten.

## 9. Visual Region Flow

The browser draws the source frame into a scaled canvas. A user may drag rectangles, assign unique region IDs, resize or remove them, and see their pixel coordinates.

Canvas coordinates are mapped once into integer source-frame coordinates using the displayed source dimensions. The server revalidates bounds, dimensions, and unique IDs. Regions are serialized into `ExperimentConfig` and displayed as an exact CLI/config representation before launch.

The UI never stores only viewport-relative or normalized coordinates because the existing runner contract and output schema use source-frame pixels.

## 10. Run and Progress Flow

1. The GUI resolves and validates an experiment configuration before importing or loading a detector.
2. It displays the config and equivalent CLI command.
3. The local server reserves a new empty output directory and starts one worker process.
4. The worker loads the selected detector and executes the chronological runner.
5. Structured progress reports current phase, completed frames, inference count, elapsed time, and terminal status.
6. Closing the browser does not stop the worker. Reopening the same local UI reconnects to the active server-side run.
7. A cancel action requests a graceful stop after the current frame. The runner publishes a `cancelled` summary containing all completed-frame artifacts.
8. Closing the UI server itself terminates its worker only after requesting graceful cancellation and waiting a bounded interval.

The run directory remains canonical. The server keeps only transient active-job state and does not create a database.

## 11. Report and Visual Style

The GUI is a compact, old-school scientific utility rather than a dashboard. It uses one page with:

- camera and video-file source tabs;
- a large preview canvas;
- visible actual source dimensions and FPS;
- simple named-region overlays;
- a compact configuration panel;
- explicit run/progress state; and
- a completed-run section.

The completed-run view provides:

- annotated diagnostic frame navigation;
- region overlays and detection labels;
- frame and inference p50/p95/p99 latency;
- peak RSS, Torch VRAM, and available NVML measurements;
- source, detector, device, threshold, regions, and dependency provenance;
- completion/failure/cancellation status; and
- copyable config path and equivalent CLI command.

There are no animations, decorative charts, hidden state, or responsive mobile requirements. Dense information, native controls, readable monospace values, and obvious status indicators take priority.

`edge-perception report` generates the same offline report from an existing run without starting the GUI. Machine-readable artifacts remain primary; `report.html` is a view over them.

## 12. Extensibility Boundary

Detector IDs and descriptors come from the detector-neutral core interface. The first proven adapter remains the externally downloaded, pinned D-FINE-N model. The GUI does not package model weights.

Future strategy and Gymnasium interfaces will follow the same pattern: typed descriptors and versioned config owned by the core, headless execution first, and optional GUI rendering second. This design does not invent strategy controls before those contracts are specified.

Unknown configuration fields, detector IDs, or future strategy IDs fail clearly rather than being silently ignored.

## 13. Failure Behavior

- **Camera permission denied:** explain the denial and keep file selection usable.
- **No camera available:** retain the file-input workflow.
- **Requested mode unavailable:** normal mode shows and accepts actual values; strict mode fails before recording.
- **Capture interruption:** delete only the incomplete temporary file.
- **Unsupported capture codec/container:** preserve the source for diagnosis, reject inference, and report the supported decode requirement.
- **Invalid config or region:** reject before model loading and identify the exact field.
- **Missing UI extra:** print the installation command without a traceback.
- **Model download or import failure:** publish an actionable failed status without claiming inference occurred.
- **Decode, CUDA OOM, or inference failure:** preserve the failed summary and valid completed-frame artifacts.
- **Output collision:** reject a non-empty directory without mutation.
- **Browser disconnect:** continue an active run.
- **Server restart:** completed run directories remain viewable; an in-progress worker is not claimed resumable.

## 14. Local Security Boundary

- Bind to `127.0.0.1` by default.
- Generate a random launch token required for mutating API requests.
- Validate request origin and content type.
- Apply capture and upload size limits.
- Resolve and validate paths before writes.
- Never overwrite completed captures or non-empty run directories.
- Store captures in a private, Git-ignored local directory by default and never upload them externally.
- Load no remote scripts, fonts, analytics, or other assets.
- Do not expose a non-loopback bind option in the initial release; remote deployment is unsupported.

## 15. Testing Strategy

### 15.1 Default Offline Suite

The default suite remains model-free, camera-free, and network-free. It covers:

- experiment and capture config validation;
- requested-versus-actual capture metadata;
- strict and normal mode behavior;
- preview-to-source coordinate scaling and clipping;
- unique region IDs and invalid rectangles;
- CLI/config resolution and override precedence;
- CLI/GUI application-service parity;
- local API validation, conflict, failure, and security paths;
- worker progress, cancellation, failure finalization, and browser reconnection state;
- deterministic report generation from fixture artifacts;
- optional-extra isolation; and
- package contents with no captured media, run artifacts, or model weights.

### 15.2 Browser Tests

Python Playwright is a development-only dependency. Chromium runs with a fake camera stream and pre-granted permission to verify:

- camera/file source switching;
- actual mode display;
- recording and stop controls;
- region drawing and coordinate conversion;
- generated config and CLI representation;
- run launch and progress rendering; and
- completed and failed report states.

No Node.js build is required.

### 15.3 Opt-In Hardware Tests

Physical camera, real model, CUDA, and long-running tests remain opt-in and identify the actual hardware used. Initial claimed support requires:

- Windows reference-laptop validation with the EMEET SmartCam Nova 4K;
- Windows CPU and NVIDIA CUDA inference;
- Linux file-input, CPU, and applicable CUDA validation; and
- explicit documentation that macOS is not yet a validated hardware target.

## 16. First User Acceptance Test

1. Install the package with the UI extra.
2. Run `edge-perception ui`.
3. Grant camera permission and select the EMEET.
4. Select `Auto` or independently request width, height, and FPS.
5. Verify the displayed actual width, height, and FPS.
6. Record a short no-audio clip while moving one ordinary object into or out of frame.
7. Draw and name one crop region.
8. Select CUDA and run a bounded D-FINE experiment.
9. Open an annotated frame and inspect detection placement, latency, and VRAM.
10. Copy the generated config/CLI command and run it into a second output directory.
11. Compare both run directories with the existing comparison command.

The reference Eyes and Stopwatch evidence run additionally uses its strict 4K/30 profile, two bounded CUDA passes, and one bounded CPU pass.

## 17. Success Criteria

The UI slice is complete when:

1. The core package and existing CLI work without UI dependencies.
2. `edge-perception ui` starts a local, token-protected interface from the optional extra.
3. A camera or file can become a validated immutable video source.
4. Width, height, and FPS are independently configurable and requested/actual values are recorded.
5. A user can create valid named source-frame regions visually.
6. The GUI and CLI resolve equivalent configs into the same application service and artifact schema.
7. A bounded CUDA run can complete while the browser displays honest progress.
8. Closing and reopening the browser does not cancel the active run.
9. Failed and cancelled runs preserve honest terminal summaries and valid partial artifacts.
10. An offline report presents annotated frames, measurements, and provenance from only the run directory.
11. The documented first user acceptance test succeeds on the reference Windows laptop.
12. Offline tests, lint, typing, package build, and clean-wheel checks pass.

## 18. Scope Guardrail

This specification adds a production-oriented visual adapter and configuration workflow to the first engineering checkpoint. It does not change the scientific claim: the checkpoint proves deterministic high-resolution detector execution, coordinate mapping, telemetry, repeatability, and usable research interfaces on constrained hardware. It does not yet prove policy quality, RL generalization, tracker performance, or detector accuracy.
