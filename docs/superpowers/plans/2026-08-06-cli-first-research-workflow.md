# CLI-First Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Adaptive Edge Perception a CLI-first, source-agnostic research tool with shared headless camera acquisition, complete run/inspect/compare commands, an explicit native workflow state model, and an offline end-to-end proof that requires no physical camera or model download.

**Architecture:** `run_checkpoint`, `RunConfig`, `CaptureResult`, canonical run artifacts, and `RunViewData` remain the product boundary. Move the existing QtCore/QtMultimedia capture backend out of the GUI package, add a non-Widgets timed-capture adapter and terminal inspection renderer, and keep all optional imports lazy. The native window remains a thin stateful adapter over those same contracts and the existing isolated JSONL worker.

**Tech Stack:** Python 3.12, argparse, PyAV 16, NumPy 2, Pillow 11/12, psutil 7, optional PySide6 6.x QtCore/QtMultimedia/QtWidgets, pytest 9, pytest-qt, Ruff, mypy, Hatchling, uv.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-06-native-workflow-state-design.md` at commit `77ed6fa`, amended on the same branch to mark implementation approval and use detector ID `dfine-nano-coco`.
- The product is CLI-first. Every durable GUI operation resolves to the same capture, configuration, runner, projection, and artifact contracts available headlessly.
- A validated local video is the runner input. Public-provider pages are materialized externally; do not add a provider downloader, credentials, network fixture, or mutable URL stream.
- A physical camera, model download, CUDA, NVML, network, browser, display server, and public video are never required by the default suite or completion gate.
- `run_checkpoint` remains the sole execution authority. Direct CLI calls are in-process; the GUI retains the existing isolated JSONL worker and cancel-file protocol.
- Preserve configuration schema version `0.1.0`, `RunConfig.capture.path == RunConfig.input_path`, capture SHA-256 provenance, canonical artifact names, and terminal values `complete`, `failed`, and `cancelled`.
- Existing-file sources are validated references, not physically immutable files. Camera captures are additionally validated, checksum-backed, privately staged, and atomically published without overwrite.
- Camera width, height, and frame rate remain independent optional constraints. Strict mode validates only specified frame size/rate, not codec, container, or pixel format.
- Camera discovery and capture use one QtMultimedia backend. The CLI drives it with `QCoreApplication`/`QEventLoop`/`QTimer`; it never imports QtWidgets or creates a window.
- Base `run`, `inspect`, and `compare` imports require neither PySide6 nor detector runtime imports. Optional `camera` and `gui` handlers import PySide6 lazily and print exact installation hints when absent.
- Detector IDs come from the lazy registry. Preserve `dfine-nano-coco` as the default; no weights, videos, captures, run directories, or Qt binaries enter Git or the project wheel.
- The native UI uses Source, Camera acquisition, Regions of interest (ROIs), Run configuration, Run, Metrics, Source provenance, and Annotations. Do not expose RL vocabulary before an RL environment exists.
- Run readiness and run lifecycle are distinct. Display lifecycle states `Not started`, `Running`, `Cancelling`, `Completed`, `Failed`, and `Cancelled`; canonical `complete` maps to presentation `Completed` only.
- The GUI opens no socket and adds no browser, QML, webview, database, dashboard, queue, provider downloader, tracking, live inference, policy, or RL controls.
- Preserve Windows and Linux compatibility, unrelated user work, and the existing default detector/config/artifact behavior.
- Every production change follows red-green-refactor. A task is not complete until its focused tests pass and its commit receives both spec-compliance and code-quality approval.

---

## File and Responsibility Map

- `src/edge_perception/capture.py` — the single shared QtCore/QtMultimedia discovery, selection, recording, validation, hashing, staging, and publication backend, moved from `gui/capture.py` without behavior drift.
- `src/edge_perception/camera_cli.py` — non-Widgets `QCoreApplication`/nested-event-loop adapter for listing cameras and timed capture.
- `src/edge_perception/inspection.py` — Qt-free terminal rendering of one already-validated `RunViewData` projection.
- `src/edge_perception/cli.py` — nested camera commands, detector override, graceful direct-run cancellation, inspect command, and concise error/exit semantics.
- `src/edge_perception/gui/workflow.py` — explicit presentation enums for source, acquisition, and run lifecycle state.
- `src/edge_perception/gui/main_window.py` — native state projection, readiness reasons, source/capture transitions, destination input, run lifecycle, and exact interface language.
- `src/edge_perception/gui/results.py` — dense grouping of existing projection fields into Overview, Metrics, Run configuration, Source provenance, and Annotations.
- `src/edge_perception/gui/region_view.py` — existing source-pixel preview/ROI surface; no second coordinate system.
- `src/edge_perception/gui/run_controller.py` — unchanged JSONL worker/cancel-file authority except for tests proving its existing terminal signal ordering.
- `pyproject.toml`, `uv.lock` — matching optional `camera` and `gui` PySide6 extras while retaining a Qt-free base.
- `tests/test_capture.py` — shared backend characterization and import-boundary coverage.
- `tests/test_camera_cli.py` — timed non-Widgets orchestration with injected Qt/controller doubles.
- `tests/test_cli.py` — parser, lazy import, detector, cancellation, inspect, and exit-code behavior.
- `tests/test_inspection.py` — literal terminal rendering from `RunViewData`.
- `tests/test_gui_app.py` — window state, language, source replacement, destination, cancellation, and close behavior.
- `tests/test_results.py` — grouped canonical projection and camera provenance rendering.
- `tests/test_cli_workflow_acceptance.py` — generated-video run → inspect → rerun → compare proof with the real runner and fake detector.
- `README.md` — install matrix, CLI-first local-video workflow, optional GUI/camera paths, public-video materialization boundary, and artifact contract.
- `docs/checkpoints/eyes-and-stopwatch.md` — reproducible offline proof, emitted artifacts, verification results, and optional hardware/model evidence lanes.

---

### Task 0: Repair Close-After-Cancel Baseline Regression

**Files:**
- Modify: `tests/test_gui_app.py`
- Modify: `src/edge_perception/gui/main_window.py`

**Interfaces:**
- Consumes: existing `RunController.runFinished` then `RunController.processTerminated` signal order and `MainWindow._exit_after_run`.
- Produces: an exit-after-cancel path that never opens a results-error modal before the termination signal can close the window.

- [ ] **Step 1: Turn the hanging test into a deterministic failing regression**

In `test_close_while_inference_cancels_restarts_kill_timer_and_waits_for_finished`, replace modal behavior with a recording function before `process.finish()`:

```python
critical_messages: list[tuple[str, str]] = []
monkeypatch.setattr(
    QMessageBox,
    "critical",
    lambda _parent, title, message: critical_messages.append((title, message)),
)
```

After `process.finish()`, assert:

```python
assert critical_messages == []
assert window.resultsWidget.isHidden()
assert window.isVisible() is False
```

- [ ] **Step 2: Run the single test and verify RED without a modal hang**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py::test_close_while_inference_cancels_restarts_kill_timer_and_waits_for_finished -q
```

Expected: FAIL because the cancelled terminal event attempts to load absent fixture artifacts and calls `QMessageBox.critical`.

- [ ] **Step 3: Skip result inspection while an explicit cancel-and-exit is pending**

At the start of `_run_finished`, preserve the factual terminal message but return before `_load_completed_run_or_report` when `_exit_after_run` is true:

```python
phase = payload.get("phase")
if self._exit_after_run:
    self.statusBar().showMessage(f"Run {phase}: {Path(run_dir).resolve()}")
    return
```

Do not set `_allow_close` here; `processTerminated` remains the only close authority.

- [ ] **Step 4: Verify the regression and the previously green baseline**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py::test_close_while_inference_cancels_restarts_kill_timer_and_waits_for_finished -q
./.tools/uv.exe run pytest -m "not model" -q
```

Expected: the focused test passes; the full baseline reports 275 passed, 1 skipped, 1 deselected with no modal hang.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_gui_app.py src/edge_perception/gui/main_window.py
git commit -m "fix: close cleanly after cancelling a run"
```

---

### Task 1: Extract the Shared Capture Backend and Optional Boundary

**Files:**
- Create by move: `src/edge_perception/capture.py`
- Delete by move: `src/edge_perception/gui/capture.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `tests/test_capture.py`
- Modify: `tests/test_gui_app.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: every current class/function/signal in `edge_perception.gui.capture`.
- Produces: the same interfaces at `edge_perception.capture`, including `CameraFormatInfo`, `CameraDeviceInfo`, `RecordingProfile`, `select_camera_format`, `select_recording_profile`, `validate_capture_result`, and `QtCaptureController`; no QtWidgets import.

- [ ] **Step 1: Write the failing shared-import boundary test**

Add a subprocess test to `tests/test_capture.py`:

```python
def test_shared_capture_import_does_not_load_qtwidgets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import edge_perception.capture; "
                "print('PySide6.QtWidgets' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
```

The production change caught by this test is moving shared capture beneath a module that imports Widgets or leaving it trapped in the GUI package.

- [ ] **Step 2: Verify RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_capture.py::test_shared_capture_import_does_not_load_qtwidgets -q
```

Expected: FAIL because `edge_perception.capture` does not exist.

- [ ] **Step 3: Move one implementation and update imports**

Use `git mv` for history, then update imports:

```powershell
git mv src/edge_perception/gui/capture.py src/edge_perception/capture.py
```

The only accepted import form in production and tests is:

```python
from edge_perception.capture import (
    CameraDeviceInfo,
    CameraFormatInfo,
    QtCaptureController,
    select_camera_format,
)
```

Do not leave a second recorder or re-export wrapper in `edge_perception.gui`.

- [ ] **Step 4: Add matching optional extras and refresh the lock**

Set both extras to the exact same range:

```toml
[project.optional-dependencies]
camera = [
    "PySide6>=6.8,<7",
]
gui = [
    "PySide6>=6.8,<7",
]
```

Run:

```powershell
./.tools/uv.exe lock
```

- [ ] **Step 5: Verify unchanged backend behavior and import laziness**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_capture.py tests/test_gui_app.py -q
./.tools/uv.exe run python -c "import sys, edge_perception.cli; assert 'PySide6' not in sys.modules"
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
```

Expected: capture/GUI tests pass; base CLI import does not import PySide6; Ruff and mypy succeed.

- [ ] **Step 6: Commit**

```powershell
git add src/edge_perception/capture.py src/edge_perception/gui/main_window.py tests/test_capture.py tests/test_gui_app.py pyproject.toml uv.lock
git commit -m "refactor: share native capture backend"
```

---

### Task 2: Add Headless Camera Discovery and Timed Capture

**Files:**
- Create: `src/edge_perception/camera_cli.py`
- Modify: `src/edge_perception/cli.py`
- Create: `tests/test_camera_cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `CaptureRequest`, `CaptureResult`, `CameraDeviceInfo`, and `QtCaptureController` from the shared capture boundary.
- Produces:

```python
def list_cameras() -> tuple[CameraDeviceInfo, ...]: ...

def capture_camera(
    request: CaptureRequest,
    *,
    duration_seconds: float,
    output: Path | None,
    _runtime: _CameraRuntime | None = None,
) -> CaptureResult: ...
```

and CLI commands `edge-perception camera list` and `edge-perception camera capture`.

- [ ] **Step 1: Write failing non-Widgets orchestration tests**

Create injected fake controller/timer/loop tests in `tests/test_camera_cli.py`. The core success assertion is:

```python
result = capture_camera(
    request,
    duration_seconds=0.05,
    output=tmp_path / "capture.mp4",
    _runtime=fake_runtime,
)

assert fake_runtime.controller.preview_requests == [(request, fake_runtime.video_sink)]
assert fake_runtime.controller.record_paths == [(tmp_path / "capture.mp4").resolve()]
assert fake_runtime.timer.intervals == [50]
assert fake_runtime.controller.stop_recording_calls == 1
assert result == expected_capture_result
assert fake_runtime.widgets_application_created is False
```

Add separate tests for `output=None`, controller error, non-positive/non-finite duration, and loop cleanup. The test double must emit the real controller signals and return a complete `CaptureResult`; do not assert only that a mock was called.

- [ ] **Step 2: Verify RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_camera_cli.py -q
```

Expected: collection fails because `edge_perception.camera_cli` does not exist.

- [ ] **Step 3: Implement one asynchronous backend adapter**

Implement `camera_cli.py` with protocols/factories collected in a private runtime record for tests. Production behavior must:

```python
application = QCoreApplication.instance() or QCoreApplication(["edge-perception-camera"])
loop = QEventLoop()
video_sink = QVideoSink()
timer = QTimer()
timer.setSingleShot(True)
```

Connect `recordingStarted` to start `round(duration_seconds * 1000)` milliseconds, timer timeout to `stop_recording`, and both `recordingFinished` and `errorOccurred` to a terminal result/error plus `loop.quit`. Schedule preview/record startup after the loop is ready, release controller-owned preview on every non-published terminal path, and never import QtWidgets.

- [ ] **Step 4: Write failing CLI parser and rendering tests**

Add tests proving these exact accepted forms:

```text
edge-perception camera list
edge-perception camera capture --device camera-1 --duration 0.05
edge-perception camera capture --device camera-1 --duration 0.05 --output capture.mp4 --width 1920 --height 1080 --fps 30 --strict
```

Patch only the optional `camera_cli` boundary. Assert that list output includes stable device ID, description, and each reported format; capture output includes finalized absolute path, SHA-256, Capture request, Applied camera format, and Recorded format. Assert unknown device, zero/NaN duration, and existing output return 2 without publication. Assert a missing PySide import prints:

```text
error: camera support is unavailable; install adaptive-edge-perception[camera]
```

- [ ] **Step 5: Implement nested camera commands with lazy imports**

Add `camera` subparsers without importing camera modules at `_build_parser` time. Parse duration with a finite-positive helper and width/height as positive integers. Resolve an explicit output path; pass `None` when omitted. Create `CaptureRequest` from the selected device descriptor so the human-readable description is not accepted from untrusted CLI text.

- [ ] **Step 6: Verify**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_camera_cli.py tests/test_cli.py tests/test_capture.py -q
./.tools/uv.exe run edge-perception camera --help
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
```

Expected: focused tests pass; help lists `list` and `capture`; static checks succeed.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/camera_cli.py src/edge_perception/cli.py tests/test_camera_cli.py tests/test_cli.py
git commit -m "feat: capture camera sources from the CLI"
```

---

### Task 3: Complete Direct Run CLI Parity and Graceful Cancellation

**Files:**
- Modify: `src/edge_perception/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: `detector_descriptors`, `load_detector`, and `run_checkpoint(..., cancel_requested=Callable[[], bool])`.
- Produces: `--detector DETECTOR_ID` in explicit/config override modes and first-SIGINT cooperative cancellation with exit 130.

- [ ] **Step 1: Write failing detector-selection tests**

Add literal config assertions:

```python
exit_code = main(
    [
        "run",
        str(video_path),
        "--output",
        str(tmp_path / "run"),
        "--detector",
        "dfine-nano-coco",
    ]
)

assert exit_code == 0
assert loaded_detector_ids == ["dfine-nano-coco"]
assert runner_configs[0].detector_id == "dfine-nano-coco"
```

Add a config-mode test where `--detector` overrides the persisted ID, an omitted flag preserves it, and an unknown ID fails before the detector runtime module imports.

- [ ] **Step 2: Verify detector RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_cli.py -k detector -q
```

Expected: FAIL because the run parser rejects `--detector`.

- [ ] **Step 3: Implement the lazy registry-backed override**

Add the argument with suppressed default and registry descriptor IDs:

```python
run.add_argument(
    "--detector",
    choices=tuple(descriptor.detector_id for descriptor in detector_descriptors()),
    default=argparse.SUPPRESS,
)
```

Set `detector_id` in `_explicit_run_config` and `_apply_config_overrides` exactly like device/threshold overrides. Do not import a detector adapter while building choices.

- [ ] **Step 4: Write the failing first-SIGINT cancellation test**

Use a fake `run_checkpoint` that raises the real process signal after receiving its callback:

```python
def fake_run_checkpoint(config: RunConfig, detector: object, *, cancel_requested: object) -> dict[str, object]:
    assert callable(cancel_requested)
    signal.raise_signal(signal.SIGINT)
    assert cancel_requested()
    return {"status": "cancelled", "frames_processed": 0, "inference_count": 0}

previous_handler = signal.getsignal(signal.SIGINT)
exit_code = main(["run", str(video_path), "--output", str(tmp_path / "run")])

assert exit_code == 130
assert signal.getsignal(signal.SIGINT) == previous_handler
```

The production break this catches is allowing `KeyboardInterrupt` into `run_checkpoint`, which would serialize `failed` instead of `cancelled`.

- [ ] **Step 5: Verify cancellation RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_cli.py -k sigint -q
```

Expected: FAIL because `_run_command` does not pass `cancel_requested` and returns 0 for cancelled summaries.

- [ ] **Step 6: Implement scoped signal handling**

Use a `threading.Event`. Install the handler only around detector load/run and restore it in `finally`:

```python
cancel_event = threading.Event()
previous_handler = signal.getsignal(signal.SIGINT)

def request_cancel(signum: int, frame: FrameType | None) -> None:
    if cancel_event.is_set():
        signal.default_int_handler(signum, frame)
    cancel_event.set()

signal.signal(signal.SIGINT, request_cancel)
try:
    detector = load_detector(...)
    summary = run_checkpoint(config, detector, cancel_requested=cancel_event.is_set)
finally:
    signal.signal(signal.SIGINT, previous_handler)
```

Return 130 only when the canonical summary status is `cancelled`; preserve existing 0/1/2 behavior elsewhere.

- [ ] **Step 7: Verify**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_cli.py tests/test_runner.py tests/test_worker.py -q
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
```

- [ ] **Step 8: Commit**

```powershell
git add src/edge_perception/cli.py tests/test_cli.py tests/test_runner.py
git commit -m "feat: make direct CLI runs selectable and cancellable"
```

---

### Task 4: Add Canonical Terminal Inspection

**Files:**
- Create: `src/edge_perception/inspection.py`
- Modify: `src/edge_perception/cli.py`
- Create: `tests/test_inspection.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `RunViewData` and `load_run_view(Path)`.
- Produces:

```python
def render_run_inspection(view: RunViewData) -> str: ...
```

and `edge-perception inspect RUN_DIRECTORY`.

- [ ] **Step 1: Write the failing literal renderer test**

Construct one `RunViewData` directly and assert complete human output contains these literal groups/values:

```python
rendered = render_run_inspection(view)

assert "Run\n  status: Completed" in rendered
assert f"  directory: {view.run_dir}" in rendered
assert "Metrics\n  frames processed: 3" in rendered
assert "  frame latency p95: 1.500 ms" in rendered
assert "Run configuration\n  detector: tests/fake-detector" in rendered
assert "Source provenance\n  path:" in rendered
assert "Annotations\n  count: 1" in rendered
```

Add failed/cancelled cases, `N/A` metrics, capture request/recorded format/SHA-256, zero annotations, and an error line only when present. Expectations must be literals, not values formatted by production helpers.

- [ ] **Step 2: Verify renderer RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_inspection.py -q
```

Expected: collection fails because `edge_perception.inspection` does not exist.

- [ ] **Step 3: Implement a pure formatter**

Render sentence-case labels and explicit units. Map only presentation status:

```python
_STATUS_LABELS = {
    "complete": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
```

Do not read files, parse JSON, or import Qt in `inspection.py`.

- [ ] **Step 4: Write failing command integration tests**

Create real canonical run fixtures and assert `main(["inspect", str(run_dir)])` prints the renderer output and returns 0. Assert missing/malformed artifacts return 2 through the existing one-line `error:` path. The CLI handler must call `load_run_view` and pass its result to `render_run_inspection`; it must not parse artifacts itself.

- [ ] **Step 5: Add the inspect parser/handler and verify**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_inspection.py tests/test_cli.py tests/test_results.py -q
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
```

- [ ] **Step 6: Commit**

```powershell
git add src/edge_perception/inspection.py src/edge_perception/cli.py tests/test_inspection.py tests/test_cli.py
git commit -m "feat: inspect canonical runs from the terminal"
```

---

### Task 5: Make Native Workflow State and Language Explicit

**Files:**
- Create: `src/edge_perception/gui/workflow.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/gui/region_view.py`
- Modify: `tests/test_gui_app.py`
- Modify: `tests/test_region_view.py`

**Interfaces:**
- Consumes: existing capture signals, source path/capture provenance, ROI collection, `RunController.is_active`, and launch-time `resolved_config` validation.
- Produces presentation enums `SourceState`, `AcquisitionState`, `RunState`, persistent labels `source-status`, `acquisition-status`, `run-readiness-status`, `run-status`, and exact professional labels from the approved spec.

- [ ] **Step 1: Write failing state/language tests**

Add tests that find the four persistent labels, a `browse-source-button`, and a read-only `capture-sha256` value, then assert startup:

```python
assert source_status.text() == "Source status: No source"
assert acquisition_status.text() == "Acquisition status: Idle"
assert run_readiness.text() == "Run readiness: Not ready: select a source"
assert run_status.text() == "Run status: Not started"
assert browse_source.text() == "Browse…"
assert capture_sha256.text() == "—"
```

Assert the exact labels `Source path`, `Frame size`, `Camera acquisition`, `Device`, `Frame rate`, `Require specified frame size and rate`, `Applied camera format`, `Capture request`, `Recorded format`, `Regions of interest (ROIs)`, `Add ROI`, `Remove ROI`, `Compute device`, `Confidence threshold`, `Warm-up iterations`, `Annotation interval (frames)`, `Output directory`, `Run configuration`, `CLI command`, `Start recording`, `Stop recording`, and `Cancel run`. Assert the strict checkbox tooltip exactly matches the spec.

Add readiness precedence cases for missing source, active/finalizing capture, empty output, nonempty output, config-path collision, and Ready.

- [ ] **Step 2: Verify RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py -k "state or language or readiness" -q
```

Expected: FAIL because the persistent labels and professional copy do not exist.

- [ ] **Step 3: Add explicit presentation enums and one render path**

Create string enums:

```python
class SourceState(StrEnum):
    NO_SOURCE = "No source"
    READY = "Ready"

class AcquisitionState(StrEnum):
    IDLE = "Idle"
    PREVIEWING = "Previewing"
    RECORDING = "Recording"
    FINALIZING = "Finalizing"
    FINALIZED = "Finalized"
    FAILED = "Failed"

class RunState(StrEnum):
    NOT_STARTED = "Not started"
    RUNNING = "Running"
    CANCELLING = "Cancelling"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
```

Store acquisition/run display state in `MainWindow`, derive source state from `_source_path`, and update labels/control availability through one `_update_control_state`/state-render pass. Add a source Browse action beside Source path and route it through the same `_choose_video` method as File → Open Video. Show camera-capture SHA-256 in a selectable read-only value after finalization and `—` for ordinary files. Run readiness reports the first actionable validation reason; `resolved_config()` remains the launch authority.

- [ ] **Step 4: Write failing transactional source tests**

Prove: changing Source mode alone preserves source/ROIs; failed file decode preserves them; successful preview invalidates them only after `previewStarted`; preview startup failure restores the prior frame/source/capture/ROIs; Stop recording displays Finalizing; successful capture retains Camera mode and displays Source Ready/Acquisition Finalized; capture failure displays Failed and no incomplete source.

- [ ] **Step 5: Implement source/acquisition transitions**

Keep the existing controller backend authoritative. Snapshot the prior source path, capture provenance, and ordered ROIs before preparing preview; on synchronous startup failure reload that source and re-add its ROIs. Commit invalidation on `previewStarted`. In `_camera_recording_finished`, remove the automatic `setCurrentText("Video file")`, load the validated result, retain Camera mode, and set Finalized.

- [ ] **Step 6: Verify**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py tests/test_region_view.py -q
./.tools/uv.exe run ruff check src/edge_perception/gui tests/test_gui_app.py tests/test_region_view.py
./.tools/uv.exe run mypy src/edge_perception/gui
```

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/gui/workflow.py src/edge_perception/gui/main_window.py src/edge_perception/gui/region_view.py tests/test_gui_app.py tests/test_region_view.py
git commit -m "feat: expose native research workflow state"
```

---

### Task 6: Wire Capture Destination and Correct Run/Results Lifecycle

**Files:**
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/gui/results.py`
- Modify: `tests/test_gui_app.py`
- Modify: `tests/test_gui_acceptance.py`
- Modify: `tests/test_results.py`
- Modify: `tests/test_run_controller.py`

**Interfaces:**
- Consumes: shared `QtCaptureController.start_recording(final_path: Path | None)`, workflow states from Task 5, `RunController` terminal signals, and `ResultsWidget.load_run`.
- Produces: `capture-destination`, `browse-capture-destination-button`, truthful Cancelling/terminal state, stale-results hiding, and projection-only grouped Run results.

- [ ] **Step 1: Write failing destination tests**

Assert blank passes `None`, relative/typed and native-dialog paths resolve before forwarding, dialog cancellation preserves the field/source, an existing target is rejected by the backend without mutation, and the destination controls lock during preview/record/finalization/run.

The key real-behavior assertion is:

```python
destination.setText(str(tmp_path / "captures" / "reference.mp4"))
qtbot.mouseClick(start_recording, Qt.MouseButton.LeftButton)
assert controller.record_paths == [(tmp_path / "captures" / "reference.mp4").resolve()]
```

- [ ] **Step 2: Verify destination RED**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py -k destination -q
```

Expected: FAIL because destination widgets do not exist and recording always passes no argument.

- [ ] **Step 3: Implement destination as a thin input**

Add a line edit with placeholder `Automatic` and a native save dialog. Resolve only a nonblank value immediately before `start_recording`; do not create directories, infer codecs, overwrite, stage, hash, or publish in the window.

- [ ] **Step 4: Write failing run/result lifecycle tests**

Prove that starting a run hides stale results; Run state becomes Running; Cancel run immediately becomes Cancelling and disables itself; complete/cancelled terminal artifacts map to Completed/Cancelled and load after cleanup; failed/nonterminal exits map to Failed and keep results hidden; close-after-cancel remains modal-free.

- [ ] **Step 5: Implement lifecycle projection**

Set Running and hide results synchronously before `RunController.start`. Set Cancelling before `cancel()`. In `_run_finished`, map `phase` through canonical values and load only complete/cancelled artifacts unless an explicit exit-after-run is pending. `_run_failed` sets Failed and leaves results hidden. Controls unlock only from `processTerminated`.

- [ ] **Step 6: Write failing ResultsWidget grouping tests**

Assert object-named groups appear in this order: `results-overview`, `results-metrics`, `results-run-configuration`, `results-source-provenance`, `results-annotations`. Assert status renders `Completed`, not `complete`, and camera capture path/SHA-256/request/recorded format come from `RunViewData.capture`. Do not assert source-code text or add a manifest browser.

- [ ] **Step 7: Group existing projection values without a second parser**

Use `QGroupBox`/forms inside the existing dense details pane. Add only values already present in `RunViewData`; `load_run_view` remains the sole artifact parser.

- [ ] **Step 8: Verify**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_gui_app.py tests/test_gui_acceptance.py tests/test_results.py tests/test_run_controller.py tests/test_capture.py -q
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
```

- [ ] **Step 9: Commit**

```powershell
git add src/edge_perception/gui/main_window.py src/edge_perception/gui/results.py tests/test_gui_app.py tests/test_gui_acceptance.py tests/test_results.py tests/test_run_controller.py
git commit -m "feat: complete native capture and run workflow"
```

---

### Task 7: Prove the Source-Agnostic Workflow and Package It

**Files:**
- Create: `tests/test_cli_workflow_acceptance.py`
- Modify: `README.md`
- Create: `docs/checkpoints/eyes-and-stopwatch.md`
- Modify: `docs/superpowers/specs/2026-08-06-native-workflow-state-design.md`

**Interfaces:**
- Consumes: generated `video_path`, real CLI/config/runner/artifact/projection/compare code, and injected `FakeDetector` only at the external detector-loading seam.
- Produces: an offline local-video run → inspect → second run → compare proof, install/use documentation, and a checkpoint evidence record.

- [ ] **Step 1: Write the failing end-to-end CLI acceptance test**

Call real `main()` commands with only `cli.load_detector` replaced by a function returning `FakeDetector`. Execute:

```python
assert main([
    "run", str(video_path),
    "--output", str(run_a),
    "--detector", "dfine-nano-coco",
    "--device", "cpu",
    "--max-frames", "3",
    "--warmup-runs", "0",
    "--annotate-every", "1",
    "--crop", "left:0,0,100,100",
    "--crop", "right:100,0,100,100",
]) == 0
assert main(["inspect", str(run_a)]) == 0
assert main([
    "run", str(video_path),
    "--output", str(run_b),
    "--detector", "dfine-nano-coco",
    "--device", "cpu",
    "--max-frames", "3",
    "--warmup-runs", "0",
    "--annotate-every", "1",
    "--crop", "left:0,0,100,100",
    "--crop", "right:100,0,100,100",
]) == 0
assert main(["compare", str(run_a), str(run_b)]) == 0
```

Assert each run contains `manifest.json`, `summary.json`, `inferences.jsonl`, `detections.jsonl`, `hardware.jsonl`, and three annotated PNGs; inspect output reports Completed/3 frames/9 inferences; compare reports `equivalent=true` and zero mismatches.

- [ ] **Step 2: Verify acceptance RED before the final command surface exists**

Run:

```powershell
./.tools/uv.exe run pytest tests/test_cli_workflow_acceptance.py -q
```

Expected before Tasks 2–4: FAIL on missing detector/inspect command behavior. At Task 7 entry, rerun and require PASS without changing the acceptance oracle.

- [ ] **Step 3: Replace stale README scope with executable workflows**

Document exact installs:

```text
pip install adaptive-edge-perception
pip install adaptive-edge-perception[camera]
pip install adaptive-edge-perception[gui]
```

Document local-video run/inspect/compare, camera list/capture, GUI launch, optional external public-video materialization, canonical artifacts, exit 130 cancellation, no bundled models, Windows/Linux support, and macOS unvalidated status. State that camera/public-video/model/CUDA evidence is optional.

- [ ] **Step 4: Write the checkpoint report from verified evidence**

Create `docs/checkpoints/eyes-and-stopwatch.md` with: claim; generated 200×100/30 FPS/3-frame source recipe; exact commands; detector-neutral fake acceptance boundary; expected artifact tree; test/lint/type/build results; optional real D-FINE/CUDA and EMEET evidence sections labeled additive; limitations; and next scientific question. Do not claim detector accuracy or hardware performance from fake-detector results.

- [ ] **Step 5: Mark the implemented design and verify documentation commands**

Set the design status to `Implemented and verified` only after all acceptance commands below pass. Search README/checkpoint/spec for stale claims that live camera is excluded, physical camera is required, GUI is the only capture path, or detector ID is `dfine`.

- [ ] **Step 6: Run full verification and build**

Run fresh:

```powershell
./.tools/uv.exe run pytest -m "not model" -q
./.tools/uv.exe run ruff check src tests
./.tools/uv.exe run mypy src
./.tools/uv.exe build
./.tools/uv.exe run python -m zipfile -l dist/adaptive_edge_perception-0.1.0-py3-none-any.whl
```

Inspect the wheel listing and assert it contains project Python/package metadata only: no MP4, JSON/JSONL run artifact, PNG annotation, model weight, private capture, Qt binary, `.superpowers`, or test file.

- [ ] **Step 7: Run optional evidence without gating completion**

If a cached pinned D-FINE model and a local video are available, run one bounded CPU or CUDA command and record only observed output. If camera hardware is available without prompting, run `camera list` and a short explicit-destination capture; otherwise record `not run — optional hardware unavailable`. Neither lane may alter pass/fail status of the source-agnostic proof.

- [ ] **Step 8: Commit**

```powershell
git add tests/test_cli_workflow_acceptance.py README.md docs/checkpoints/eyes-and-stopwatch.md docs/superpowers/specs/2026-08-06-native-workflow-state-design.md
git commit -m "docs: publish CLI-first workflow proof"
```

---

## Plan Self-Review Checklist

- Every approved spec section maps to a task: shared CLI/capture boundary (Tasks 1–4), native state/language/destination/results (Tasks 5–6), source-agnostic proof/package/docs (Task 7), optional hardware lane (Task 7 only).
- The known pre-existing modal hang has an isolated red-green repair before feature work (Task 0).
- No task changes schema version, canonical terminal values, detector default, artifact names, capture publication guarantees, GUI worker topology, or source-coordinate ROI contracts.
- Every new production interface has a test that fails before implementation and asserts real behavior at the owned boundary.
- Camera CLI and GUI use one backend; inspect and GUI use one run projection; direct CLI and worker use one runner.
- Default verification is offline, model-free, camera-free, GPU-free, browser-free, provider-free, and repeatable from a generated video.
