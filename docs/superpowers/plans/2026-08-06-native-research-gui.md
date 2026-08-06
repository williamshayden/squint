# Native Research GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal native PySide6 research GUI over the existing detector-neutral CLI/core so a researcher can capture or choose video, define source-pixel regions, launch the canonical runner, and inspect artifacts without any browser or web runtime.

**Architecture:** Keep `RunConfig`, runner, detector adapters, and run directories authoritative. Add pure JSON configuration and progress contracts, then place two thin adapters over them: the existing CLI and one optional Qt Widgets application. The GUI uses Qt Multimedia for preview/recording, `QProcess` for isolated inference, and native widgets for completed-run inspection; it opens no socket and stores no second copy of results.

**Tech Stack:** Python 3.12, PyAV 16, NumPy 2, Pillow 11/12, PySide6 6.x (`gui` extra), Qt Widgets, Qt Multimedia, `pytest`, `pytest-qt`, Ruff, mypy, uv.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-06-cli-first-native-research-gui-design.md` at commit `35d6737`.
- The core package and `edge-perception run` / `compare` commands must import and run without PySide6 installed.
- `edge-perception gui` must launch one native `QMainWindow`; do not add HTML, JavaScript, QML, WebEngine, a webview, an HTTP server, or any listening socket.
- PySide6 belongs only in the optional `gui` extra. GUI tests may depend on `pytest-qt` through the development group.
- Use platform-native Qt widget styling. Permit only small status-color and monospace-value adjustments; do not add animations, charts, custom chrome, or a design-system layer.
- Camera width, height, and FPS are independent optional parameters. The product has no required resolution or frame rate.
- Strict capture requires exact decoded width/height and decoded average FPS within `max(0.1, requested_fps * 0.005)` FPS.
- GUI capture is video-only. Do not connect an audio input or add an audio control.
- A capture becomes an immutable, checksummed local video before inference. PyAV—not the selected Qt format alone—provides final stream evidence.
- The GUI may own one active inference worker. It must not queue or parallelize GPU work.
- Run directories and their JSON/JSONL/PNG artifacts remain the source of truth. Do not add a database or GUI-only result format.
- Detector weights remain external and pinned by the adapter; no model/video/run artifact may enter the wheel or Git history.
- Default tests remain offline, model-free, camera-free, CUDA-free, browser-free, and display-server-free.
- Initial claimed GUI support is Windows and Linux. macOS remains unvalidated and must be labeled accordingly.
- Preserve unrelated user changes. In particular, the existing uncommitted `tests/test_dfine_integration.py` device-selection correction belongs to the paused reference-checkpoint task and must not be overwritten.

---

## File and Responsibility Map

- `src/edge_perception/config.py` — Qt-free capture/run configuration records, JSON loading/writing, relative-path resolution, and CLI rendering.
- `src/edge_perception/detectors/registry.py` — lazy detector descriptors and loader dispatch; no model import during discovery.
- `src/edge_perception/progress.py` — JSON-native progress event and cancellation callback contracts.
- `src/edge_perception/worker.py` — isolated config-driven runner entry point used by `QProcess`.
- `src/edge_perception/run_view.py` — Qt-free projection of canonical run artifacts for native presentation.
- `src/edge_perception/gui/app.py` — optional PySide import boundary and `QApplication` lifecycle.
- `src/edge_perception/gui/main_window.py` — one compact window and explicit UI state transitions.
- `src/edge_perception/gui/region_view.py` — source-coordinate graphics scene, file preview, and named region editing.
- `src/edge_perception/gui/capture.py` — pure format-selection helpers plus the Qt Multimedia capture controller.
- `src/edge_perception/gui/run_controller.py` — one `QProcess`, progress parsing, cancellation-file ownership, and terminal-state signals.
- `src/edge_perception/gui/results.py` — native completed-run artifact view.
- `tests/test_config.py` — pure config and provenance contracts.
- `tests/test_detector_registry.py` — lazy descriptor and detector loading.
- `tests/test_progress.py`, `tests/test_worker.py` — progress, cancellation, and worker behavior.
- `tests/test_gui_app.py` — optional dependency and native window bootstrap.
- `tests/test_region_view.py` — source-coordinate mapping and file preview.
- `tests/test_capture.py` — pure capture selection/validation plus injected Qt-controller behavior.
- `tests/test_run_controller.py` — worker process state machine and cancellation.
- `tests/test_results.py` — canonical artifact projection and native viewer.
- `tests/test_gui_acceptance.py` — offscreen vertical slice with fake capture/detector worker.

## Execution Preflight: Preserve the Existing Model-Smoke Fix

Before Task 1, return `tests/test_dfine_integration.py` to the original checkpoint implementer or review it as an isolated pre-existing change. It must contain `_model_test_device()` and use `MODEL_TEST_DEVICE` for both detector loading and identity assertion. Verify:

```powershell
.\.tools\uv.exe run pytest tests/test_dfine_integration.py::test_model_test_device_respects_requested_cuda -q
$env:RUN_MODEL_TESTS = "1"
$env:MODEL_TEST_DEVICE = "cpu"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
$env:MODEL_TEST_DEVICE = "cuda"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
```

Expected: the focused regression passes; the opt-in CPU and actual CUDA smokes each pass. Commit only that file as `test: honor requested model smoke device`. Begin Task 1 from a clean worktree.

---

### Task 1: Versioned Run and Capture Configuration

**Files:**
- Create: `src/edge_perception/config.py`
- Modify: `src/edge_perception/runner.py`
- Modify: `src/edge_perception/outputs.py`
- Create: `tests/test_config.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: existing `Region` and the seven existing `RunConfig` fields.
- Produces: `CONFIG_SCHEMA_VERSION`, `CaptureRequest`, `CaptureResult`, extended `RunConfig`, `CaptureRequest.to_dict()`, `CaptureRequest.from_dict(payload)`, `CaptureResult.to_dict()`, `CaptureResult.from_dict(payload, *, base_dir: Path)`, `load_run_config(path: Path) -> RunConfig`, `write_run_config(path: Path, config: RunConfig) -> None`, and `render_run_cli(config_path: Path, output_override: Path | None = None) -> tuple[str, ...]`.
- Compatibility: `from edge_perception.runner import RunConfig` must continue working through a re-export.

- [ ] **Step 1: Write failing pure-contract tests**

Create tests proving that width, height, and FPS are independent, strict mode is JSON-native, relative input/output/capture paths resolve against the config file, unknown schema fields fail, non-finite FPS fails, and write/load round-trips exactly:

```python
from pathlib import Path

import pytest

from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    load_run_config,
    write_run_config,
)
from edge_perception.contracts import Region


def test_capture_request_treats_dimensions_and_fps_independently() -> None:
    assert CaptureRequest("camera-1", "EMEET", 1920, None, 15.0, False).requested_height is None
    assert CaptureRequest("camera-1", "EMEET", None, 1080, None, True).requested_width is None


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0.0, -1.0])
def test_capture_request_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="FPS"):
        CaptureRequest("camera-1", "EMEET", None, None, fps, False)


def test_run_config_file_round_trips_capture_provenance(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video")
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 30.0, True)
    capture = CaptureResult(
        request=request,
        selected_width=1920,
        selected_height=1080,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=1920,
        actual_height=1080,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=5.0,
        has_audio=False,
        file_size_bytes=5,
        path=video,
        sha256="a" * 64,
    )
    config = RunConfig(
        input_path=video,
        output_dir=tmp_path / "run",
        regions=(Region("roi", 1, 2, 30, 40),),
        threshold=0.3,
        max_frames=3,
        warmup_runs=1,
        annotate_every=1,
        detector_id="dfine-nano-coco",
        device="cuda",
        capture=capture,
    )
    path = tmp_path / "experiment.json"

    write_run_config(path, config)

    assert load_run_config(path) == config
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.tools\uv.exe run pytest tests/test_config.py -q
```

Expected: collection fails because `edge_perception.config` does not exist.

- [ ] **Step 3: Implement immutable Qt-free capture contracts**

Create the records with exact fields and validations:

```python
CONFIG_SCHEMA_VERSION = "0.1.0"
FPS_ABSOLUTE_TOLERANCE = 0.1
FPS_RELATIVE_TOLERANCE = 0.005


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    device_id: str
    device_description: str
    requested_width: int | None
    requested_height: int | None
    requested_fps: float | None
    strict: bool


@dataclass(frozen=True, slots=True)
class CaptureResult:
    request: CaptureRequest
    selected_width: int
    selected_height: int
    selected_min_fps: float
    selected_max_fps: float
    selected_pixel_format: str
    actual_width: int
    actual_height: int
    actual_fps: float
    container: str
    codec: str
    duration_seconds: float
    has_audio: bool
    file_size_bytes: int
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.has_audio:
            raise ValueError("GUI capture must not contain audio")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
```

Validate non-empty device strings; positive optional width/height; finite positive optional FPS; positive selected/actual dimensions; finite non-negative duration; positive file size; and selected/actual FPS values. `to_dict()` must emit only JSON-native values and paths as strings. `from_dict()` must reject missing, unknown, bool-as-number, non-finite, and mistyped values.

- [ ] **Step 4: Move and extend `RunConfig` without breaking imports**

Move `RunConfig` to `config.py`, preserve its first seven field positions, and add defaults afterward:

```python
@dataclass(frozen=True, slots=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    regions: tuple[Region, ...]
    threshold: float
    max_frames: int | None
    warmup_runs: int
    annotate_every: int
    detector_id: str = "dfine-nano-coco"
    device: str = "auto"
    capture: CaptureResult | None = None
```

Retain every existing validation and add: non-empty detector ID, device in `{"auto", "cpu", "cuda"}`, and capture path equal to `input_path` after resolution. In `runner.py`, import `RunConfig` from `config.py` so existing external imports remain valid.

- [ ] **Step 5: Implement deterministic config serialization and CLI rendering**

Serialize this exact document shape with `allow_nan=False`, sorted keys, sibling temporary file, and `os.replace`:

```json
{
  "schema_version": "0.1.0",
  "source": {"path": "C:/abs/input.mp4", "capture": null},
  "output": {"directory": "C:/abs/run"},
  "detector": {"id": "dfine-nano-coco", "device": "cuda", "threshold": 0.3},
  "regions": [{"region_id": "roi", "x": 1, "y": 2, "width": 30, "height": 40}],
  "execution": {"max_frames": 3, "warmup_runs": 1, "annotate_every": 1}
}
```

When loading, resolve relative `source.path`, `source.capture.path`, and `output.directory` against the config file's parent. `render_run_cli()` returns an argument tuple, never a shell string:

```python
("edge-perception", "run", "--config", str(config_path.resolve()))
```

Append `("--output", str(output_override.resolve()))` only when an override exists.

- [ ] **Step 6: Preserve capture provenance in the manifest**

Add `capture` under `source_video`:

```python
"source_video": {
    "path": str(config.input_path.resolve()),
    "sha256": _sha256_file(config.input_path),
    "frame_width": int(first_frame.image.shape[1]),
    "frame_height": int(first_frame.image.shape[0]),
    "capture": None if config.capture is None else config.capture.to_dict(),
},
```

Add `detector_id` and requested `device` to the existing `configuration`
mapping so every resolved CLI override is recorded, while the existing
`detector` mapping continues to report actual loaded model/device identity.
Add a runner test asserting that capture provenance survives unchanged, a
file-only run records `capture: null`, and detector/device overrides appear in
the resolved configuration.

- [ ] **Step 7: Run focused and existing contract suites**

Run:

```powershell
.\.tools\uv.exe run pytest tests/test_config.py tests/test_contracts.py tests/test_runner.py -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass; no PySide6 import occurs.

- [ ] **Step 8: Commit**

```powershell
git add src/edge_perception/config.py src/edge_perception/runner.py src/edge_perception/outputs.py tests/test_config.py tests/test_runner.py
git commit -m "feat: add versioned experiment configuration"
```

---

### Task 2: Lazy Detector Registry and Config-Driven CLI

**Files:**
- Create: `src/edge_perception/detectors/registry.py`
- Modify: `src/edge_perception/cli.py`
- Create: `tests/test_detector_registry.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `RunConfig`, `load_run_config()`, existing `DfineDetector.load(threshold, device)`.
- Produces: `DetectorDescriptor`, `detector_descriptors() -> tuple[DetectorDescriptor, ...]`, `load_detector(detector_id: str, *, threshold: float, device: str) -> Detector`, and config-aware `edge-perception run`.

- [ ] **Step 1: Write failing registry and CLI tests**

```python
def test_registry_discovery_does_not_import_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))

    descriptors = detector_descriptors()

    assert [item.detector_id for item in descriptors] == ["dfine-nano-coco"]
    assert imported == []


def test_run_config_loads_selected_detector_once(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "experiment.json"
    write_run_config(config_path, RunConfig(video_path, tmp_path / "run", (), 0.3, 1, 0, 0))
    calls: list[tuple[str, float, str]] = []
    monkeypatch.setattr(cli, "load_detector", lambda detector_id, *, threshold, device: calls.append((detector_id, threshold, device)) or object())
    monkeypatch.setattr(cli, "run_checkpoint", lambda _config, _detector: {"status": "complete", "frames_processed": 1, "inference_count": 1})

    assert cli.main(["run", "--config", str(config_path)]) == 0
    assert calls == [("dfine-nano-coco", 0.3, "auto")]
```

Cover the remaining CLI cases with exact test names and outcomes:

| Test | Required outcome |
|---|---|
| `test_run_accepts_config_without_positional_input` | exit `0`; loaded config reaches `run_checkpoint` unchanged |
| `test_explicit_run_flags_keep_existing_defaults` | existing positional invocation constructs the pre-change values |
| `test_output_flag_overrides_config_output_only` | only `output_dir` differs from the loaded config |
| `test_present_flags_override_config_and_omitted_flags_do_not` | threshold/device/max-frames override; omitted warm-up/annotation fields remain |
| `test_run_requires_input_or_config` | exit `2` with `error: run requires INPUT or --config` |
| `test_input_and_config_are_mutually_exclusive` | exit `2` with the specified conflict error |
| `test_unknown_detector_id_fails_before_model_import` | exit `2`; D-FINE module absent from `sys.modules` |
| `test_malformed_config_is_one_line_and_model_free` | one stderr line; no D-FINE import |

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.tools\uv.exe run pytest tests/test_detector_registry.py tests/test_cli.py -q
```

Expected: failures for missing registry and missing `--config` parsing.

- [ ] **Step 3: Implement lazy detector descriptors and loading**

```python
@dataclass(frozen=True, slots=True)
class DetectorDescriptor:
    detector_id: str
    display_name: str
    model_id: str
    revision: str


_DFINE = DetectorDescriptor(
    detector_id="dfine-nano-coco",
    display_name="D-FINE Nano (COCO)",
    model_id="ustc-community/dfine-nano-coco",
    revision="066438d3d8f0da137a37b38fdf3368fd4afceced",
)


def detector_descriptors() -> tuple[DetectorDescriptor, ...]:
    return (_DFINE,)


def load_detector(detector_id: str, *, threshold: float, device: str) -> Detector:
    if detector_id != _DFINE.detector_id:
        raise ValueError(f"unknown detector ID: {detector_id}")
    from edge_perception.detectors.dfine import DfineDetector
    return DfineDetector.load(threshold=threshold, device=device)
```

Discovery must import neither Torch nor Transformers nor the D-FINE module.

- [ ] **Step 4: Make `run` accept config or explicit input**

Change the positional input to `nargs="?"`; add `--config`; make `--output` optional at parse time. Use `argparse.SUPPRESS` for override-capable flag defaults so omitted flags cannot overwrite config values. Resolve one `RunConfig`, validate its output before detector loading, call `load_detector()` once, then call `run_checkpoint()`.

Explicit mode retains current defaults. Config mode starts from `load_run_config()` and applies only fields present on the namespace. Reject these exact invalid states:

```text
error: run requires INPUT or --config
error: --output is required without --config
error: INPUT cannot be combined with --config
```

- [ ] **Step 5: Run CLI and registry suites**

```powershell
.\.tools\uv.exe run pytest tests/test_detector_registry.py tests/test_cli.py tests/test_dfine.py -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass; ordinary `--help`, `compare`, and validation paths remain model/network-free.

- [ ] **Step 6: Commit**

```powershell
git add src/edge_perception/detectors/registry.py src/edge_perception/cli.py tests/test_detector_registry.py tests/test_cli.py
git commit -m "feat: run versioned detector configurations"
```

---

### Task 3: Structured Progress, Graceful Cancellation, and Worker Entry Point

**Files:**
- Create: `src/edge_perception/progress.py`
- Create: `src/edge_perception/worker.py`
- Modify: `src/edge_perception/runner.py`
- Create: `tests/test_progress.py`
- Create: `tests/test_worker.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RunConfig`, `load_run_config()`, `load_detector()`, `run_checkpoint()`.
- Produces: `ProgressEvent`, `ProgressCallback`, `CancelCheck`, `run_checkpoint(..., progress=None, cancel_requested=None)`, and `python -m edge_perception.worker --config PATH --cancel-file PATH`.

- [ ] **Step 1: Write failing progress and cancellation tests**

```python
def test_progress_event_is_finite_json_native() -> None:
    event = ProgressEvent("running", 2, 6, 123.5, None)
    assert event.to_dict() == {
        "phase": "running",
        "frames_processed": 2,
        "inference_count": 6,
        "elapsed_ms": 123.5,
        "error": None,
    }


def test_runner_cancels_between_completed_frames(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    events: list[ProgressEvent] = []
    config = RunConfig(video_path, tmp_path / "run", (), 0.3, None, 0, 0)

    summary = run_checkpoint(
        config,
        fake_detector,
        progress=events.append,
        cancel_requested=lambda: len(fake_detector.predict_batch_sizes) >= 1,
    )

    assert summary["status"] == "cancelled"
    assert summary["frames_processed"] == 1
    assert events[-1].phase == "cancelled"
```

Worker tests use these exact cases:

| Test | Required outcome |
|---|---|
| `test_worker_emits_one_finite_json_object_per_line` | every non-empty stdout line parses independently and contains no `NaN`/`Infinity` |
| `test_worker_honors_preexisting_cancel_file` | terminal phase and summary status are both `cancelled` |
| `test_worker_reports_detector_load_failure_once` | one `failed` event, one compact stderr line, exit `2` |

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
.\.tools\uv.exe run pytest tests/test_progress.py tests/test_worker.py tests/test_runner.py -q
```

Expected: missing progress module and unsupported runner keyword arguments.

- [ ] **Step 3: Implement the progress contract**

```python
ProgressPhase = Literal["validating", "warming_up", "running", "complete", "cancelled", "failed"]
ProgressCallback = Callable[["ProgressEvent"], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: ProgressPhase
    frames_processed: int
    inference_count: int
    elapsed_ms: float
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "frames_processed": self.frames_processed,
            "inference_count": self.inference_count,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }
```

Validate allowed phase, non-negative integer counts, finite non-negative elapsed time, and optional string error.

- [ ] **Step 4: Add progress and cancellation to the runner**

Extend only by keyword-only callbacks:

```python
def run_checkpoint(
    config: RunConfig,
    detector: Detector,
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCheck | None = None,
) -> dict[str, object]:
```

Emit `validating` before preview, `warming_up` before warm-up, `running` after every durable frame flush, and exactly one terminal event. Check cancellation after each durable completed frame and before requesting the next decode. Extend `_summary()` with an explicit status of `complete`, `cancelled`, or `failed`; cancellation is not an exception and retains valid completed-frame artifacts.

- [ ] **Step 5: Implement the isolated worker**

Provide an injection-friendly function plus module entry:

```python
def run_worker(
    config_path: Path,
    cancel_file: Path,
    *,
    detector_loader: Callable[..., Detector] = load_detector,
    stream: TextIO = sys.stdout,
) -> int:
    config = load_run_config(config_path)

    def emit(event: ProgressEvent) -> None:
        stream.write(json.dumps(event.to_dict(), allow_nan=False, sort_keys=True) + "\n")
        stream.flush()

    detector = detector_loader(
        config.detector_id,
        threshold=config.threshold,
        device=config.device,
    )
    summary = run_checkpoint(
        config,
        detector,
        progress=emit,
        cancel_requested=cancel_file.exists,
    )
    return 0 if summary["status"] in {"complete", "cancelled"} else 2
```

`main()` parses only `--config` and `--cancel-file`, catches controlled errors, emits one `failed` event when failure occurs before the runner lifecycle, prints a compact diagnostic to stderr, and returns `2`. The GUI will invoke it as `sys.executable -m edge_perception.worker`; do not add a public shell command.

- [ ] **Step 6: Run focused and full offline tests**

```powershell
.\.tools\uv.exe run pytest tests/test_progress.py tests/test_worker.py tests/test_runner.py -q
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass and failure summaries remain compatible with existing records.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/progress.py src/edge_perception/worker.py src/edge_perception/runner.py tests/test_progress.py tests/test_worker.py tests/test_runner.py
git commit -m "feat: expose cancellable checkpoint progress"
```

---
### Task 4: Optional PySide6 Extra and Native Window Bootstrap

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/edge_perception/gui/__init__.py`
- Create: `src/edge_perception/gui/app.py`
- Create: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/cli.py`
- Create: `tests/test_gui_app.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: detector descriptors and Qt-free config contracts.
- Produces: optional `gui` dependency extra, `launch_gui(run_dir: Path | None = None, *, argv: Sequence[str] = ()) -> int`, `MainWindow`, and `edge-perception gui [--run RUN_DIR]`.

- [ ] **Step 1: Write failing optional-boundary and native-window tests**

```python
def test_gui_command_lazily_launches_native_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | None] = []
    module = ModuleType("edge_perception.gui.app")
    module.launch_gui = lambda run_dir=None: calls.append(run_dir) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_perception.gui.app", module)

    assert cli.main(["gui", "--run", str(tmp_path)]) == 0
    assert calls == [tmp_path.resolve()]


def test_main_window_is_one_native_qmain_window(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window, QMainWindow)
    assert window.objectName() == "edge-perception-main-window"
    assert window.findChild(QGraphicsView, "source-view") is not None
    assert window.findChild(QPushButton, "run-button") is not None
```

Add `test_gui_command_reports_missing_optional_extra`: inject
`ImportError("No module named 'PySide6'")`, assert exit code `2`, assert stderr
equals `error: native GUI dependencies are unavailable; install adaptive-edge-perception[gui]\n`,
and assert no traceback text is present.

- [ ] **Step 2: Add the optional dependency and test dependency**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
gui = [
    "PySide6>=6.8,<7",
]

[dependency-groups]
dev = [
    "pytest",
    "pytest-cov",
    "pytest-qt>=4.4,<5",
    "ruff",
    "mypy",
]
```

Retain the existing `cpu` and `cu128` extras unchanged. Resolve the lock and environment:

```powershell
.\.tools\uv.exe lock
.\.tools\uv.exe sync --extra cu128 --extra gui --group dev
```

- [ ] **Step 3: Run tests and verify the missing GUI modules fail**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_gui_app.py tests/test_cli.py -q
```

Expected: imports fail because `edge_perception.gui.app` and `MainWindow` are absent.

- [ ] **Step 4: Implement the smallest native application lifecycle**

`app.py` imports PySide6; the parent `edge_perception.gui` package must stay empty of Qt imports:

```python
def launch_gui(run_dir: Path | None = None, *, argv: Sequence[str] = ()) -> int:
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(["edge-perception", *argv])
        app.setApplicationName("Edge Perception")
    window = MainWindow(run_dir=run_dir)
    window.show()
    if not owns_application:
        return 0
    return int(app.exec())
```

Keep shown windows in a module-level `_OPEN_WINDOWS` collection until their
`destroyed` signal fires. This prevents a window launched into an existing
`QApplication` (notably an embedding host or test process) from being collected
when `launch_gui()` returns.

`MainWindow` contains one horizontal `QSplitter`: a `QGraphicsView` named `source-view` on the left and a compact `QFormLayout` on the right with source mode, detector, device, threshold, output, run, and cancel controls. Add a bottom status bar and a collapsed completed-run widget placeholder. Do not add custom stylesheets beyond status colors and a monospace font for provenance values.

- [ ] **Step 5: Add a lazy CLI subcommand**

Add `gui` parsing with optional `--run`. Import `launch_gui` only inside `_gui_command()`. Resolve `--run`, require it to contain `manifest.json` and `summary.json`, and convert optional-import failure into the exact actionable error from Step 1.

- [ ] **Step 6: Run native bootstrap, CLI, packaging, and no-browser checks**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_gui_app.py tests/test_cli.py -q
.\.tools\uv.exe run python -c "from edge_perception import cli; assert 'PySide6' not in __import__('sys').modules; assert cli.main(['--help']) == 0"
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Inspect imports with `rg "QtWeb|WebEngine|http.server|starlette|fastapi|uvicorn|socket" src tests`; expected: no matches in GUI implementation.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml uv.lock src/edge_perception/gui src/edge_perception/cli.py tests/test_gui_app.py tests/test_cli.py
git commit -m "feat: launch optional native research GUI"
```

---

### Task 5: File Preview and Source-Pixel Region Editor

**Files:**
- Create: `src/edge_perception/gui/region_view.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/video.py`
- Create: `tests/test_region_view.py`
- Modify: `tests/test_video.py`
- Modify: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `Region`, PyAV chronological decoding, `RunConfig`.
- Produces: `first_video_frame(path: Path) -> DecodedFrame`, `RegionView`, `RegionView.set_rgb_frame(image: np.ndarray)`, `RegionView.add_region(region: Region)`, `RegionView.regions() -> tuple[Region, ...]`, and `MainWindow.load_video(path: Path)`.

- [ ] **Step 1: Write failing first-frame and region-coordinate tests**

```python
def test_region_view_keeps_scene_coordinates_in_source_pixels(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    view.set_rgb_frame(image)

    view.add_region(Region("upper-right", 960, 0, 960, 540))

    assert view.scene().sceneRect() == QRectF(0.0, 0.0, 1920.0, 1080.0)
    assert view.regions() == (Region("upper-right", 960, 0, 960, 540),)


def test_region_from_drag_clips_and_rounds_to_source_pixels(qtbot: QtBot) -> None:
    view = RegionView()
    qtbot.addWidget(view)
    view.set_rgb_frame(np.zeros((100, 200, 3), dtype=np.uint8))

    region = view.region_from_scene_rect("roi", QRectF(-2.4, 10.2, 44.8, 30.1))

    assert region == Region("roi", 0, 10, 43, 31)
```

Use exact named contracts for the rest of the editor behavior:

| Test | Required outcome |
|---|---|
| `test_region_ids_must_be_unique_and_not_reserved` | duplicate and `full-frame` both raise before scene mutation |
| `test_zero_size_drag_creates_no_region` | region tuple and scene item count stay unchanged |
| `test_moving_region_clamps_to_source_bounds` | emitted source-pixel `Region` remains fully in frame |
| `test_numeric_resize_updates_scene_and_contract` | spin-box values, rectangle, and emitted region agree exactly |
| `test_delete_removes_selected_region` | region and overlay disappear together |
| `test_resize_only_refits_view` | regions before/after viewport resize are equal |
| `test_rgb_frame_is_owned_by_qimage` | mutating the input NumPy array does not change displayed pixel data |
| `test_first_video_frame_closes_iterator` | injected generator's `finally` flag is true after one returned frame |

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_region_view.py tests/test_video.py -q
```

Expected: `RegionView` and `first_video_frame` are missing.

- [ ] **Step 3: Add a bounded first-frame API**

```python
def first_video_frame(path: Path) -> DecodedFrame:
    frames = iter_video(path)
    try:
        try:
            return next(frames)
        except StopIteration as error:
            raise ValueError(f"video contains no decoded frames: {path}") from error
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
```

- [ ] **Step 4: Implement a source-coordinate `QGraphicsView`**

`RegionView` owns a `QGraphicsScene` whose rectangle is always `(0, 0, source_width, source_height)`. Convert RGB arrays into an owning `QImage(...).copy()` and one `QGraphicsPixmapItem`. Use `fitInView(sceneRect(), Qt.KeepAspectRatio)` only for viewport scaling; never rescale region coordinates.

Region creation clamps the normalized drag rectangle to the scene, floors its left/top, ceils its right/bottom, then constructs `Region`. Region items store their ID with `setData(0, region_id)`. Keep insertion order in a private list. Moving or numeric editing clamps the item and emits `regionsChanged(tuple[Region, ...])`. Use numeric `QSpinBox` fields for exact x/y/width/height resizing rather than custom drag handles.

- [ ] **Step 5: Connect file selection and region controls in `MainWindow`**

`load_video(path)` calls `first_video_frame()`, places the image in `RegionView`, records source dimensions, clears stale regions, and displays path/dimensions. Add File / Open Video through `QFileDialog`; tests call `load_video()` directly. Add `New Region`, `Delete Region`, ID, x, y, width, and height controls. Disable run until a valid source and output are present.

- [ ] **Step 6: Run focused GUI and video suites**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_region_view.py tests/test_video.py tests/test_gui_app.py -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass with no model, camera, display server, or network.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/gui/region_view.py src/edge_perception/gui/main_window.py src/edge_perception/video.py tests/test_region_view.py tests/test_video.py tests/test_gui_app.py
git commit -m "feat: define source regions in native preview"
```

---

### Task 6: Qt Camera Format Selection and Video-Only Capture

**Files:**
- Create: `src/edge_perception/gui/capture.py`
- Modify: `src/edge_perception/gui/region_view.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/video.py`
- Create: `tests/test_capture.py`
- Modify: `tests/test_video.py`
- Modify: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: `CaptureRequest`, `CaptureResult`, `RegionView`, PyAV.
- Produces: `VideoMetadata`, `probe_video(path: Path) -> VideoMetadata`, `CameraFormatInfo`, `RecordingProfile`, `select_camera_format(formats, request) -> CameraFormatInfo`, `select_recording_profile(file_formats, video_codecs) -> RecordingProfile`, `validate_capture_result(request, metadata) -> None`, and `QtCaptureController` signals/methods.

- [ ] **Step 1: Write failing pure camera-selection and probe tests**

```python
def test_select_camera_format_treats_fps_independently() -> None:
    formats = (
        CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object()),
        CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),
    )
    request = CaptureRequest("camera-1", "EMEET", None, None, 60.0, False)

    selected = select_camera_format(formats, request)

    assert (selected.width, selected.height, selected.max_fps) == (1280, 720, 60.0)


def test_strict_capture_uses_documented_fps_tolerance(tmp_path: Path) -> None:
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 30.0, True)
    accepted = VideoMetadata(1920, 1080, 29.95, "mp4", "h264", 5.0, False, 100)
    rejected = replace(accepted, average_fps=29.7)

    validate_capture_result(request, accepted)
    with pytest.raises(ValueError, match="FPS"):
        validate_capture_result(request, rejected)
```

Use `test_probe_video_reads_generated_fixture`,
`test_probe_video_reports_injected_audio_stream`,
`test_probe_video_rejects_container_without_video`, and
`test_probe_video_rejects_malformed_input` for the probing contract. Use the existing generated MP4, an injected fake PyAV
container with an audio stream, a fake no-video container, and a malformed
file. This avoids depending on a host AAC encoder in the default suite. Add
`test_controller_enumerates_stable_device_descriptors`,
`test_preview_applies_selected_format`, `test_recording_state_is_explicit`,
`test_discard_removes_only_owned_temporary_file`,
`test_recorder_error_cleans_temporary_file`,
`test_success_atomically_publishes_capture`, and
`test_capture_result_uses_probed_metadata` for the injected controller. Each
name states the exact oracle; assert the selected paths, signal count, and full
record equality rather than widget text alone.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_capture.py tests/test_video.py -q
```

Expected: missing capture controller, selection helper, and probe API.

- [ ] **Step 3: Implement reusable video probing**

```python
@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    average_fps: float
    container: str
    codec: str
    duration_seconds: float
    has_audio: bool
    file_size_bytes: int
```

`probe_video()` opens with PyAV, requires one video stream, derives average FPS from `average_rate`, derives duration from stream/container timing without non-finite values, reports whether any audio stream exists, and closes the container on every path.

- [ ] **Step 4: Implement deterministic format selection and strict validation**

`CameraFormatInfo` contains width, height, min/max FPS, pixel-format name, and
an opaque Qt format object excluded from equality/repr:

```python
@dataclass(frozen=True, slots=True)
class CameraFormatInfo:
    width: int
    height: int
    min_fps: float
    max_fps: float
    pixel_format: str
    qt_format: object = field(compare=False, repr=False)
```

In normal mode rank formats lexicographically by: number of supplied constraints satisfied, absolute width difference, absolute height difference, distance from requested FPS range, then higher pixel count and max FPS. With no constraints, prefer the format with the largest pixel count and then max FPS. Strict mode filters exact supplied dimensions and a reported range containing requested FPS, failing with `requested camera mode is unavailable` when empty.

Post-record validation requires no audio. Strict width/height are exact. Strict FPS uses:

```python
tolerance = max(0.1, request.requested_fps * 0.005)
if abs(metadata.average_fps - request.requested_fps) > tolerance:
    raise ValueError("captured FPS differs from strict request")
```

- [ ] **Step 5: Implement the Qt Multimedia capture controller**

Create `QtCaptureController(QObject)` with signals `devicesChanged`, `previewStarted`, `previewStopped`, `recordingStarted`, `recordingFinished`, and `errorOccurred`. It owns exactly one `QCamera`, `QMediaCaptureSession`, and `QMediaRecorder` at a time.

Use `QMediaDevices.videoInputs()` and `QCameraDevice.videoFormats()` to build stable descriptors (`bytes(device.id()).hex()` for ID). Apply the selected opaque `QCameraFormat` through `QCamera.setCameraFormat()`. Route preview to the `QGraphicsVideoItem` supplied by `RegionView`. Connect no `QAudioInput`.

Choose recorder output deterministically from Qt's encode capabilities: prefer
MPEG-4, then Matroska, then the first enum-sorted supported file format; prefer
H.264, then H.265, VP9, AV1, then the first enum-sorted supported video codec.
Represent that decision with a pure `RecordingProfile` record and cover the
preference/fallback order with fake enum values. If Qt reports no encodable
file format or video codec, fail before recording with
`no supported video recording profile`.

Record to a sibling private temporary path. On `QMediaRecorder.RecorderState.StoppedState`, resolve `actualLocation()`, call `probe_video()`, validate, hash, construct `CaptureResult`, and publish with `os.replace`. On recorder error or explicit discard, stop owned Qt objects and remove only the temporary file.

- [ ] **Step 6: Integrate camera mode into the one window**

Populate camera, width, height, and FPS controls from controller descriptors. `Auto` maps to `None` independently for each numeric field. Start Preview displays selected mode; Record/Stop transitions are explicit. When recording completes, switch the source to the finalized file, call `first_video_frame()` for a stable region-editing image, and show requested plus actual capture metadata.

- [ ] **Step 7: Run fake-controller, GUI, and offline suites**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_capture.py tests/test_video.py tests/test_gui_app.py -q
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass without opening a physical camera.

- [ ] **Step 8: Commit**

```powershell
git add src/edge_perception/gui/capture.py src/edge_perception/gui/region_view.py src/edge_perception/gui/main_window.py src/edge_perception/video.py tests/test_capture.py tests/test_video.py tests/test_gui_app.py
git commit -m "feat: capture parameterized native camera video"
```

---

### Task 7: Native Run Controller and Canonical Worker Execution

**Files:**
- Create: `src/edge_perception/gui/run_controller.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/config.py`
- Create: `tests/test_run_controller.py`
- Modify: `tests/test_gui_app.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `write_run_config()`, `render_run_cli()`, `ProgressEvent`, and `python -m edge_perception.worker`.
- Produces: `RunController.start(config: RunConfig)`, `RunController.cancel()`, signals `progressChanged`, `runFinished`, `runFailed`, and GUI config/run/cancel behavior.

- [ ] **Step 1: Write failing process-state and CLI-parity tests**

```python
def test_run_controller_starts_worker_without_shell(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)

    controller.start(config)

    assert fake_process.program == sys.executable
    assert fake_process.arguments == [
        "-m",
        "edge_perception.worker",
        "--config",
        str((config.output_dir.parent / f"{config.output_dir.name}.experiment.json").resolve()),
        "--cancel-file",
        str((config.output_dir.parent / f".{config.output_dir.name}.cancel").resolve()),
    ]
    assert fake_process.started_once


def test_run_controller_rejects_second_active_run(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    controller.start(make_config(tmp_path, output="run-a"))

    with pytest.raises(RuntimeError, match="already active"):
        controller.start(make_config(tmp_path, output="run-b"))
```

Implement the remaining state-machine tests with these exact names and
oracles:

| Test | Stimulus | Required assertion |
|---|---|---|
| `test_partial_stdout_is_buffered_until_newline` | emit half a JSON record, then its remainder plus newline | no signal before newline; one equal `ProgressEvent` afterward |
| `test_one_stdout_chunk_can_contain_multiple_events` | emit two newline-terminated records in one chunk | two ordered `progressChanged` emissions |
| `test_non_json_stdout_fails_the_run` | emit `not-json\n` | one `runFailed("worker emitted malformed progress")` |
| `test_stderr_is_bounded_and_included_in_failure` | emit more than 16 KiB on stderr, then crash | failure contains the final 16 KiB and not the discarded prefix |
| `test_cancel_atomically_publishes_owned_file` | call `cancel()` during an active run | the exact cancel path exists, is empty, and no sibling temporary remains |
| `test_zero_exit_without_terminal_event_is_failure` | emit `running`, then normal exit | no `runFinished`; one missing-terminal failure |
| `test_crash_preserves_run_and_experiment_config` | abnormal exit after output creation | run directory and config path both still exist |
| `test_terminal_cleanup_removes_only_cancel_file` | cancel, emit `cancelled`, then normal exit | cancel path is gone; config and run artifacts remain |
| `test_persisted_gui_config_loads_as_last_config` | complete a run | `load_run_config(controller.config_path) == controller.last_config` |

For the malformed-progress row, declare the exact error text as a module
constant and assert that constant in the test so implementation and test do
not drift.

Each fake process records program, argument list, start count, kill count, and
signal emissions. Assert exact paths and messages; do not use sleeps or a real
subprocess in this suite.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_run_controller.py tests/test_config.py tests/test_gui_app.py -q
```

Expected: `RunController` and GUI-run integration are absent.

- [ ] **Step 3: Implement one injected `QProcess` state machine**

```python
class RunController(QObject):
    progressChanged = Signal(object)
    runFinished = Signal(Path, dict)
    runFailed = Signal(str)

    def __init__(self, process: QProcess | None = None) -> None:
        super().__init__()
        self._process = QProcess(self) if process is None else process
        self._active = False
        self._stdout_buffer = ""
```

Expose read-only `config_path: Path | None` and `last_config: RunConfig | None`
properties. `start()` validates an absent/empty output and an absent persistent
`<output-name>.experiment.json`, writes that config atomically, removes a stale
private `.<output-name>.cancel` file, sets `QProcess` program and argument list,
and starts once. The experiment config is user-facing reproducibility input and
must survive completion; only the cancellation control file is temporary.
Never call `startCommand()` or pass a shell string. Parse UTF-8 stdout incrementally by newline into `ProgressEvent`; retain bounded stderr for failure messages. Require a terminal event plus normal exit before `runFinished`. Any crash, malformed event, or exit without a terminal event emits `runFailed` and preserves the run directory and experiment config.

`cancel()` atomically writes a zero-byte cancellation file only while active.
Terminal cleanup removes only that owned cancellation file and never removes
the experiment config or run artifacts.

- [ ] **Step 4: Build a validated config from native controls**

Add `MainWindow.resolved_config() -> RunConfig`. It requires a finalized video source and new empty output directory, reads detector/device/threshold/execution controls, reads ordered regions from `RegionView`, and carries current `CaptureResult`. Display the exact JSON config path and a copyable argument rendering from `render_run_cli()`.

Run button calls `RunController.start()`, disables source/config mutation, enables Cancel, and renders structured progress. Terminal completion restores controls and loads the output in the results area. Failure displays one `QMessageBox` plus status text; do not infer success from process exit alone.

- [ ] **Step 5: Implement explicit close behavior**

Override `closeEvent()`. During recording, offer `Keep Window Open` or `Stop and Discard`. During inference, offer `Keep Window Open` or `Cancel Run and Exit`; the latter ignores the first close, calls `cancel()`, and closes only after terminal cleanup. Arm an injected/restartable single-shot timer for 5,000 ms; its production callback calls `QProcess.kill()` and closes after the process-finished signal. With no active work, accept immediately. Tests inject the dialog decision, timer, and process.

- [ ] **Step 6: Run focused GUI/process suites**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_run_controller.py tests/test_config.py tests/test_gui_app.py -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass; the fake worker proves no shell and one active process.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/gui/run_controller.py src/edge_perception/gui/main_window.py src/edge_perception/config.py tests/test_run_controller.py tests/test_gui_app.py tests/test_config.py
git commit -m "feat: launch canonical runs from native GUI"
```

---

### Task 8: Canonical Run Projection and Native Results Viewer

**Files:**
- Create: `src/edge_perception/run_view.py`
- Create: `src/edge_perception/gui/results.py`
- Modify: `src/edge_perception/gui/main_window.py`
- Modify: `src/edge_perception/gui/app.py`
- Create: `tests/test_results.py`
- Modify: `tests/test_gui_app.py`

**Interfaces:**
- Consumes: canonical `manifest.json`, `summary.json`, and `annotated/*.png`.
- Produces: `RunViewData`, `load_run_view(run_dir: Path) -> RunViewData`, `ResultsWidget.load_run(run_dir: Path)`, and `edge-perception gui --run RUN_DIR` completed mode.

- [ ] **Step 1: Write failing artifact-projection tests**

```python
def test_load_run_view_uses_only_canonical_artifacts(tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path)

    view = load_run_view(run_dir)

    assert view.status == "complete"
    assert view.frames_processed == 3
    assert view.detector_model_id == "tests/fake-detector"
    assert view.frame_p50_ms == pytest.approx(10.0)
    assert [path.name for path in view.annotation_paths] == ["000000.png", "000002.png"]


def test_results_widget_loads_annotation_without_mutating_run(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    run_dir = write_completed_run_fixture(tmp_path)
    before = snapshot_tree(run_dir)
    widget = ResultsWidget()
    qtbot.addWidget(widget)

    widget.load_run(run_dir)

    assert widget.statusLabel.text() == "complete"
    assert widget.annotationList.count() == 2
    assert snapshot_tree(run_dir) == before
```

Use exact named contracts for remaining result behavior:

| Test | Required outcome |
|---|---|
| `test_load_run_view_accepts_failed_and_cancelled_status` | status/error fields are projected exactly |
| `test_load_run_view_rejects_missing_malformed_or_wrong_schema_artifacts` | each fixture raises one field-specific `ValueError` |
| `test_load_run_view_allows_no_annotations` | annotation tuple is empty without failure |
| `test_load_run_view_rejects_non_finite_metrics` | `NaN` and infinities are rejected before widget loading |
| `test_annotation_paths_are_filename_sorted_and_contained` | ordering is deterministic; traversal/symlink escape is rejected |
| `test_results_widget_formats_unavailable_telemetry_as_na` | RSS/VRAM/NVML labels show `N/A` |
| `test_capture_provenance_is_typed` | projected capture equals the expected `CaptureResult` |
| `test_gui_run_mode_is_camera_model_and_network_lazy` | injected camera/model/network sentinels have zero calls |

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_results.py tests/test_gui_app.py -q
```

Expected: missing `run_view` and `ResultsWidget`.

- [ ] **Step 3: Implement a strict Qt-free run projection**

```python
@dataclass(frozen=True, slots=True)
class RunViewData:
    run_dir: Path
    status: Literal["complete", "failed", "cancelled"]
    frames_processed: int
    inference_count: int
    annotated_frame_count: int
    frame_p50_ms: float | None
    frame_p95_ms: float | None
    frame_p99_ms: float | None
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    detector_model_id: str
    detector_revision: str
    device: str
    threshold: float
    source_path: Path
    source_width: int
    source_height: int
    capture: CaptureResult | None
    regions: tuple[Region, ...]
    annotation_paths: tuple[Path, ...]
    error: str | None
```

`load_run_view()` requires both JSON files, accepted schema version, matching run ID, finite metrics, valid status, and PNG annotations contained under `run_dir/annotated`. It parses capture provenance through `CaptureResult.from_dict()` rather than retaining an untyped nested mapping. It performs no writes and imports no Qt.

- [ ] **Step 4: Implement the old-school native results widget**

Use a horizontal `QSplitter`: annotation filename list on the left, `QGraphicsView` image on the center, and read-only `QFormLayout`/`QTableWidget` values on the right. Show status, error, frames, inferences, p50/p95/p99, peak RSS/VRAM, detector/revision/device/threshold, source dimensions/FPS when captured, and ordered regions. Use `N/A` for unavailable metrics. No plots, animations, thumbnails, or generated result files.

- [ ] **Step 5: Integrate completed mode**

`MainWindow(run_dir=...)` validates and loads `ResultsWidget` immediately while leaving source/run controls available for a new experiment. A newly completed run loads through the same `load_run_view()` path. `gui --run` must not enumerate cameras or import the detector/model runtime until the user explicitly switches to camera or starts a new run.

- [ ] **Step 6: Run results, GUI, and default offline suites**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_results.py tests/test_gui_app.py -q
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all pass; run fixtures remain byte-for-byte unchanged.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/run_view.py src/edge_perception/gui/results.py src/edge_perception/gui/main_window.py src/edge_perception/gui/app.py tests/test_results.py tests/test_gui_app.py
git commit -m "feat: inspect canonical runs in native GUI"
```

---

### Task 9: Offscreen Vertical Slice, Physical Acceptance, and Checkpoint Publication

**Files:**
- Create: `tests/test_gui_acceptance.py`
- Modify: `README.md`
- Create: `docs/checkpoints/eyes-and-stopwatch.md`
- Modify as evidence requires: `tests/test_capture.py`
- Local ignored input: `fixtures/private/checkpoint-01.mp4`
- Local ignored outputs: `docs/checkpoints/artifacts/*`

**Interfaces:**
- Consumes: completed config/registry/worker/GUI/capture/results stack and reference hardware.
- Produces: one automated fake vertical slice, one user-runnable native acceptance path, real CPU/CUDA evidence, and checkpoint documentation.

- [ ] **Step 1: Write the offscreen vertical-slice test**

Inject a fake capture controller that reports two formats and finalizes the existing three-frame video, plus a fake process that emits validating/running/complete events and points at a canonical fixture run:

```python
def test_native_gui_vertical_slice_is_config_reproducible(
    qtbot: QtBot,
    tmp_path: Path,
    video_path: Path,
) -> None:
    capture = FakeCaptureController(video_path)
    process = FakeProcess(completed_run_factory=write_completed_run_fixture)
    window = MainWindow(capture_controller=capture, process=process)
    qtbot.addWidget(window)

    window.select_camera("camera-1")
    window.set_capture_request(width=1280, height=None, fps=15.0, strict=False)
    window.record_short_fixture()
    window.regionView.add_region(Region("roi", 10, 10, 40, 30))
    window.set_output_dir(tmp_path / "run")
    qtbot.mouseClick(window.runButton, Qt.LeftButton)

    assert window.resultsWidget.statusLabel.text() == "complete"
    config_path = window.runController.config_path
    assert config_path is not None
    config = load_run_config(config_path)
    assert config.regions == (Region("roi", 10, 10, 40, 30),)
    assert config.capture is not None
    assert window.runController.last_config == config
    assert render_run_cli(config_path)[0:3] == ("edge-perception", "run", "--config")
```

The test must use real Qt signals and widgets but no physical camera, model, GPU, browser, server, or network.

- [ ] **Step 2: Run the complete automated quality gate**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest tests/test_gui_acceptance.py -q
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
.\.tools\uv.exe build
.\.tools\uv.exe run python -m zipfile -l dist\adaptive_edge_perception-0.1.0-py3-none-any.whl
```

Expected: all checks pass; wheel contains Python/Qt resource code but no Qt binaries, model weights, videos, run artifacts, or private captures.

- [ ] **Step 3: Launch the native GUI for the user's first acceptance test**

```powershell
.\.tools\uv.exe sync --extra cu128 --extra gui --group dev
.\.tools\uv.exe run edge-perception gui
```

The user verifies: one native window opens; no browser opens; no listening socket appears; EMEET is selectable; Auto and independent width/height/FPS controls work; actual selected format is visible; a short video-only recording finalizes; one named region can be created; a bounded CUDA run completes; an annotated frame and latency/VRAM values display; and the saved config reruns from CLI.

- [ ] **Step 4: Freeze the reference source and verify its actual properties**

Record the private reference source through the GUI at strict `3840 × 2160 @ 30 FPS` only for the Eyes and Stopwatch experiment. Save it to `fixtures/private/checkpoint-01.mp4` (or the Qt-selected extension and update commands consistently). Use `probe_video()` and record codec, width, height, average FPS, duration, audio absence, file size, and SHA-256. If strict capture fails, preserve the honest diagnostic and use a supported mode for generic GUI acceptance; do not call that source the strict 4K reference.

- [ ] **Step 5: Run truthful model and hardware proofs**

```powershell
$env:RUN_MODEL_TESTS = "1"
$env:MODEL_TEST_DEVICE = "cpu"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
$env:MODEL_TEST_DEVICE = "cuda"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
```

Run two bounded CUDA configurations generated by the GUI into distinct empty directories, then compare with `--box-atol 0.01 --score-atol 0.0001`. Run one bounded CPU configuration. Expected: both device paths identify their real device, both CUDA runs complete and compare equivalent, and the CPU run completes without NVML being required.

- [ ] **Step 6: Write the checkpoint report and README workflow**

Document exact install/run commands, native GUI launch, config schema, CLI parity, requested/actual capture mode, Git commit, input/model hashes, model revision, dependency versions, driver/CUDA/device, warm-up, regions, threshold, frame counts, latency p50/p95/p99, RAM/VRAM, repeatability, annotated example paths, failures encountered, validated operating systems, and limitations. State explicitly that the checkpoint proves engineering behavior, not detector accuracy or RL policy quality.

- [ ] **Step 7: Re-run the final gate from a clean state**

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
.\.tools\uv.exe build
git status --short
```

Expected: only intended README/report/test changes are present; ignored captures and run artifacts do not appear.

- [ ] **Step 8: Commit the verified native checkpoint**

```powershell
git add README.md docs/checkpoints/eyes-and-stopwatch.md tests/test_gui_acceptance.py tests/test_capture.py
git commit -m "docs: publish native Eyes and Stopwatch checkpoint"
```

---

## Plan Self-Review Record

- **Spec coverage:** Tasks 1–3 establish the canonical config, detector selection, progress, cancellation, and isolated worker. Tasks 4–8 add only the optional native Qt adapter: bootstrap, file/region interaction, camera capture, worker control, and artifact viewing. Task 9 proves the complete user and hardware paths.
- **Placeholder scan:** the plan contains none of `TODO`, `TBD`, `FIXME`, deferred implementation language, unspecified error handling, or cross-task shorthand; supplemental behavior is expressed as named tests with exact oracles.
- **Minimality:** One window, one worker, one detector descriptor, one capture implementation, no database, no network service, no web runtime, no live inference, no charts, and no desktop bundler.
- **Parameter honesty:** Width, height, and FPS are independently nullable in `CaptureRequest`; strict decoded FPS uses the exact documented tolerance; the generic product does not require 4K.
- **Type consistency:** `RunConfig`, `CaptureRequest`, `CaptureResult`, `ProgressEvent`, `VideoMetadata`, and `RunViewData` are defined once in earlier tasks and consumed with the same field names later. `RunViewData.capture` is a parsed `CaptureResult | None`, and the GUI's persistent experiment path is exposed consistently as `RunController.config_path`.
- **Reproducibility:** `<output-name>.experiment.json` is immutable user-facing input and survives worker exit; only the uniquely named cancellation file is cleaned up. The displayed CLI tuple therefore remains executable after the GUI run.
- **Scientific honesty:** Qt's selected format is recorded but never treated as ground truth; PyAV probes the finalized source. CPU/CUDA/model/camera tests remain opt-in and failures are reported without relabeling.
- **Dirty-worktree safety:** the existing D-FINE model-device test correction is isolated in the execution preflight and must be committed before Task 1.
