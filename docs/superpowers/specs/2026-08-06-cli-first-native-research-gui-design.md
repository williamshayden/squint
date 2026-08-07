# CLI-First Native Research GUI Design

**Date:** 2026-08-06

**Status:** Approved for implementation planning

**Evidence correction (2026-08-07):** The implemented checkpoint is gated by the generated 200×100 local-video workflow and non-model suite. The strict-4K, two-pass CUDA, and real-model CPU items in the original acceptance design are unproved, are superseded as checkpoint completion criteria, and remain planned opt-in hardware validation.

**Project:** Adaptive Edge Perception (working title)

## 1. Outcome

Adaptive Edge Perception will remain a headless, configuration-driven research tool with an optional native desktop interface for work that is inherently visual.

The command line, typed Python API, versioned configuration, and run artifacts are the product contract. The GUI is a thin local Qt adapter over that contract. It must not introduce a browser, webview, local HTTP server, second experiment engine, demo-only workflow, or results that cannot be reproduced without the GUI.

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
2. Add a minimal native Qt Widgets GUI for camera/file preview, recording, region definition, run launch, and artifact inspection.
3. Use one versioned experiment configuration across the CLI, Python API, and GUI.
4. Save every recording as an ordinary immutable video input before inference begins.
5. Record requested and actual capture properties, including width, height, and FPS.
6. Make every inference run launched from the GUI reproducible through a displayed CLI command.
7. Preserve JSON/JSONL artifacts as the source of truth and render them in a native completed-run view.
8. Keep the GUI dependency set optional and avoid browser, webview, HTTP-server, and Node.js dependencies.
9. Support a production-quality local workflow on Windows and Linux first.

## 4. Non-Goals

The initial UI will not include:

- accounts, authentication systems, cloud storage, or remote collaboration;
- a browser-based or embedded-webview interface;
- a multi-user server or network deployment;
- any listening network socket;
- a database or a second results store;
- a general dashboard, experiment tracker, or notebook replacement;
- live object detection, live tracking, or live policy decisions;
- a video editor or labeling suite;
- audio capture;
- parallel GPU job scheduling;
- a plugin marketplace;
- RL controls before strategy and environment contracts exist in the core package;
- bundled detector weights; or
- a standalone executable bundle or operating-system installer.

## 5. Product Interfaces

### 5.1 CLI

The canonical interfaces are:

```text
edge-perception run --config experiment.json
edge-perception compare RUN_A RUN_B
edge-perception gui [--run RUN_DIR]
```

Existing explicit `run` flags remain available for one-off experiments. A config file is the preferred reproducible workflow. If both are supplied, explicitly provided flags override config values and the resolved configuration is written to the run manifest.

`edge-perception gui` launches one native desktop window. Supplying `--run` opens an existing run directory directly in the read-only completed-run view.

### 5.2 Python API

The typed Python application layer accepts the same validated experiment configuration and exposes run, comparison, completed-run projection, capability-discovery, and progress/cancellation boundaries. Neither CLI parsing nor GUI concepts appear in this layer.

### 5.3 Optional Native GUI Package

The core package remains importable without GUI dependencies. From a clean repository checkout, install the native GUI with an installed `uv` or into an activated local virtual environment with pip:

```text
uv sync --extra gui
python -m pip install -e ".[gui]"
```

The complete non-model test suite includes GUI tests and therefore starts with `uv sync --extra gui`, followed by `uv run pytest -m "not model"`. Invoking `edge-perception gui` without the extra prints checkout-safe installation commands and exits cleanly.

The optional runtime uses PySide6 with Qt Widgets and Qt Multimedia. PySide6 remains isolated in the `gui` extra because its wheels include Qt binaries. The interface uses platform-native Qt widget styles with only restrained status-color customization; it contains no HTML, JavaScript, QML, webview, or frontend compilation step.

## 6. Configuration Contracts

### 6.1 Experiment Configuration

`ExperimentConfig` is a versioned, JSON-native document containing:

- source video path;
- optional capture-provenance record for GUI-acquired sources;
- output directory;
- detector ID and detector-owned settings;
- execution device;
- score threshold;
- zero or more named source-frame regions;
- warm-up count;
- maximum frame count;
- annotation interval; and
- schema version.

The resolved configuration is validated before model loading and written into the run manifest. When the source was captured by the GUI, the manifest copies the requested and actual `CaptureResult` fields so a later CLI run retains acquisition provenance. Region coordinates are integer source-frame pixels, independent of preview scaling.

Annotations are optional diagnostic artifacts. An annotation interval of zero disables PNG generation without disabling canonical detections, telemetry, inspection, or comparison.

### 6.2 Capture Request

Camera acquisition is a separate step from inference. `CaptureRequest` contains:

- Qt-provided camera-device ID and human-readable description;
- optional requested width;
- optional requested height;
- optional requested FPS;
- strict-mode boolean.

Width, height, and FPS are independent optional parameters. No particular resolution or frame rate is required by the product.

The GUI offers `Auto`, camera-reported formats, common presets, and custom values. The capture adapter selects the closest supported `QCameraFormat` in normal mode. In strict mode it accepts only a camera-reported format whose supplied width and height match exactly and whose requested FPS falls within the format's reported range; every supplied field is validated independently before recording. Post-recording strict validation requires exact decoded width and height plus average FPS within the documented tolerance `max(0.1 FPS, requested FPS × 0.005)`.

`CaptureResult` records:

- requested width, height, and FPS;
- actual negotiated width, height, and FPS;
- camera-device ID and description;
- selected Qt camera format;
- container and codec;
- duration;
- audio presence, which must be `false` for GUI captures;
- file size;
- final path; and
- SHA-256 digest.

The original reference design proposed strict `3840 × 2160 @ 30 FPS` validation, but the implemented checkpoint did not prove it. That profile is now planned opt-in hardware validation, not a product requirement or completion gate. Other experiments may use any decodable resolution, FPS, aspect ratio, prerecorded file, or synthetic video.

## 7. Architecture

The system has six bounded components:

1. **Core contracts and application services** validate configurations, run experiments, compare runs, expose capabilities, report progress, and support graceful cancellation.
2. **Detector adapters** own model loading and prediction while returning backend-neutral records. The GUI consumes detector descriptors from the core and does not hard-code model logic.
3. **CLI adapter** maps arguments and config files into application calls and renders concise terminal output.
4. **Qt GUI controller** owns one `QMainWindow`, validates user input, launches one worker with `QProcess`, and presents explicit state. It does not perform inference itself.
5. **Qt capture adapter** uses `QMediaDevices`, `QCamera`, `QCameraFormat`, `QMediaCaptureSession`, `QGraphicsVideoItem`, and `QMediaRecorder` for device discovery, raw preview, format selection, recording, and region overlays.
6. **Native run viewer** reads a completed run directory and renders annotations, measurements, and provenance with Qt Widgets. It never creates a second results format.

The GUI launches inference in an isolated worker process using the resolved configuration. `QProcess` passes an argument list directly to an internal worker entry point, never through a shell. The worker calls the same application service as the CLI and emits structured progress records. This isolates CUDA/model memory from the GUI and makes CLI/GUI parity testable.

Only one active experiment is permitted initially. The GUI disables a second launch while its worker is active rather than allowing accidental GPU contention.

## 8. Camera and Video Flow

1. The GUI asks the operating system for camera access only after the user explicitly starts preview.
2. `QMediaDevices.videoInputs()` supplies available cameras and each device's reported `QCameraFormat` values.
3. The user selects a camera plus optional width, height, and FPS constraints.
4. The capture adapter selects a supported format, applies it to `QCamera`, and displays the selected resolution, pixel format, and frame-rate range before recording.
5. `QMediaCaptureSession` routes the raw camera preview to a `QGraphicsVideoItem`; no model inference runs in preview.
6. Recording starts and stops explicitly; duration is not hard-coded.
7. `QMediaRecorder` writes video with no audio input.
8. Recording targets a private temporary path. Recorder errors leave a failed diagnostic and remove only the incomplete temporary file.
9. After recording stops, PyAV verifies decodability and measures the actual stream width, height, average FPS, codec, duration, file size, and checksum.
10. Strict-mode capture succeeds only when decoded width and height match exactly and decoded average FPS is within the documented tolerance. Normal mode records any difference honestly.
11. The validated file is atomically published and becomes an ordinary source video for `ExperimentConfig`.

Closing the GUI during recording first prompts the user to stop and discard the incomplete recording. Completed captures are never silently overwritten.

## 9. Visual Region Flow

The GUI places the video item and rectangle overlays in one `QGraphicsScene`. A user may drag rectangles, assign unique region IDs, resize or remove them, and see their pixel coordinates.

Scene coordinates are mapped once into integer source-frame coordinates using the video item's displayed content bounds and source dimensions, including letterbox offsets. The core application revalidates bounds, dimensions, and unique IDs. Regions are serialized into `ExperimentConfig` and displayed as an exact CLI/config representation before launch.

The GUI never stores only widget-relative or normalized coordinates because the existing runner contract and output schema use source-frame pixels.

## 10. Run and Progress Flow

1. The GUI resolves and validates an experiment configuration before importing or loading a detector.
2. It displays the config and equivalent CLI command.
3. The GUI reserves a new empty output directory and starts one worker through `QProcess`.
4. The worker loads the selected detector and executes the chronological runner.
5. The worker writes newline-delimited JSON progress records to standard output. `QProcess` parses those records and reports current phase, completed frames, inference count, elapsed time, and terminal status without blocking the GUI event loop.
6. A cancel action atomically publishes a cancellation request that the worker checks between frames. The runner publishes a `cancelled` summary containing all completed-frame artifacts.
7. Closing the GUI during an active run offers only two explicit choices: keep the window open, or cancel the run and exit after bounded cleanup. There is no tray process or hidden background mode.

The run directory remains canonical. The GUI keeps only transient in-process view state and does not create a database.

## 11. Report and Visual Style

The GUI is a compact, old-school scientific utility rather than a dashboard. It uses one `QMainWindow` with:

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

There are no animations, decorative charts, custom window chrome, hidden state, or responsive-layout requirements. Qt's platform-native widget style, dense information, readable monospace values, and obvious status indicators take priority. `edge-perception gui --run RUN_DIR` opens the same window directly in completed-run mode. Machine-readable artifacts remain primary; the native viewer is only a view over them.

## 12. Extensibility Boundary

Detector IDs and descriptors come from the detector-neutral core interface. The first proven adapter remains the externally downloaded, pinned D-FINE-N model. The GUI does not package model weights.

Future strategy and Gymnasium interfaces will follow the same pattern: typed descriptors and versioned config owned by the core, headless execution first, and optional GUI rendering second. This design does not invent strategy controls before those contracts are specified.

Unknown configuration fields, detector IDs, or future strategy IDs fail clearly rather than being silently ignored.

## 13. Failure Behavior

- **Camera permission denied:** explain the denial and keep file selection usable.
- **No camera available:** retain the file-input workflow.
- **Requested mode unavailable:** normal mode shows and accepts actual values; strict mode fails before recording.
- **Capture interruption:** delete only the incomplete temporary file.
- **Qt recorder error:** show the native recorder error, preserve diagnostic metadata, and remove only the incomplete temporary file.
- **Unsupported capture codec/container:** preserve the source for diagnosis, reject inference, and report the supported decode requirement.
- **Invalid config or region:** reject before model loading and identify the exact field.
- **Missing GUI extra:** print the installation command without a traceback.
- **Model download or import failure:** publish an actionable failed status without claiming inference occurred.
- **Decode, CUDA OOM, or inference failure:** preserve the failed summary and valid completed-frame artifacts.
- **Worker crash:** report the `QProcess` exit status, preserve stderr for diagnosis, and never label the run complete.
- **Output collision:** reject a non-empty directory without mutation.
- **Window-close request during capture or inference:** require an explicit discard/cancel confirmation; do not hide work in a tray or background process.

## 14. Local Process and Filesystem Safety

- Open no listening socket and start no HTTP server.
- Pass worker arguments as a `QProcess` argument list without shell evaluation.
- Resolve and validate paths before writes.
- Never overwrite completed captures or non-empty run directories.
- Store captures in a private, Git-ignored local directory by default and never upload them externally.
- Load no remote assets, analytics, or network services.
- Keep model downloads in the existing explicit detector-loading path; starting the GUI alone performs no network access.

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
- GUI controller state and one-worker conflict behavior;
- `QProcess` progress parsing, cancellation, crashes, and failure finalization;
- deterministic native run-view models built from fixture artifacts;
- filesystem and no-shell process safety;
- optional-extra isolation; and
- package contents with no captured media, run artifacts, or model weights.

### 15.2 Native GUI Tests

`pytest-qt` is a development-only dependency. Tests run Qt with its offscreen platform and inject fake capture and worker adapters to verify:

- camera/file source switching;
- camera-format selection with independent width, height, and FPS constraints;
- actual mode display;
- recording and stop controls;
- region drawing and coordinate conversion;
- generated config and CLI representation;
- run launch and progress rendering; and
- completed, failed, and cancelled run views.

The default suite does not require a physical camera, model, CUDA device, browser, display server, or Node.js runtime. Qt Multimedia integration with a real camera remains opt-in because virtual-camera behavior differs by operating system.

### 15.3 Opt-In Hardware Tests

Physical camera, real model, CUDA, CPU-model, and long-running tests remain opt-in and identify the actual hardware used. Future hardware-support claims are planned to require:

- Windows reference-laptop validation with the EMEET SmartCam Nova 4K;
- Windows CPU and NVIDIA CUDA inference;
- Linux file-input, CPU, and applicable CUDA validation; and
- explicit documentation that macOS is not yet a validated hardware target.

## 16. First User Acceptance Test

1. Install the package with the GUI extra.
2. Run `edge-perception gui` and confirm that a native desktop window opens without a browser or listening socket.
3. Grant camera permission and select the EMEET.
4. Select `Auto` or independently request width, height, and FPS.
5. Verify the displayed actual width, height, and FPS.
6. Record a short no-audio clip while moving one ordinary object into or out of frame.
7. Draw and name one crop region.
8. Select CUDA and run a bounded D-FINE experiment.
9. Open an annotated frame and inspect detection placement, latency, and VRAM.
10. Copy the generated config/CLI command and run it into a second output directory.
11. Compare both run directories with the existing comparison command.

Superseded acceptance note: the implemented evidence did not run the strict 4K/30 profile, two bounded CUDA passes, or a bounded real-model CPU pass. Those remain planned opt-in experiments and cannot be cited as current checkpoint evidence.

## 17. Success Criteria

The GUI slice is complete when:

1. The core package and existing CLI work without GUI dependencies.
2. `edge-perception gui` starts one native Qt Widgets window from the optional extra without opening a browser or listening socket.
3. A camera or file can become a validated immutable video source.
4. Width, height, and FPS are independently configurable and requested/actual values are recorded.
5. A user can create valid named source-frame regions visually.
6. The GUI and CLI resolve equivalent configs into the same application service and artifact schema.
7. The generated-video, model-replaced acceptance can complete while the native window displays honest progress from the isolated worker; CUDA GUI execution remains opt-in evidence.
8. A close request during capture or inference requires an explicit discard/cancel decision and leaves no hidden process.
9. Failed and cancelled runs preserve honest terminal summaries and valid partial artifacts.
10. The native completed-run view presents annotated frames, measurements, and provenance from only the run directory.
11. The source-agnostic acceptance and default GUI suite succeed on the reference Windows laptop; the physical-camera/real-model user acceptance remains planned and opt-in.
12. Offline tests, lint, typing, package build, and clean-wheel checks pass.

## 18. Scope Guardrail

This specification adds a production-oriented visual adapter and configuration workflow to the first engineering checkpoint. The verified claim is narrower: a deterministic generated local video exercises the production decode, coordinate, runner, artifact, inspection, comparison, and GUI contracts with the external model-loading boundary replaced, plus separately labeled optional hardware smoke evidence. It does not prove high-resolution performance, real-model CPU behavior, two-pass reproducibility, policy quality, RL generalization, tracker performance, or detector accuracy.
