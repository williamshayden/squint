from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QByteArray, QObject, QProcess, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton
from pytestqt.qtbot import QtBot

from edge_perception.config import RunConfig, load_run_config, render_run_cli
from edge_perception.contracts import Region
from edge_perception.gui.main_window import MainWindow
from edge_perception.gui.region_view import RegionView
from edge_perception.progress import ProgressEvent


def _latency(count: int, p50: float, p95: float, p99: float) -> dict[str, float | int]:
    return {"count": count, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


def write_completed_run_fixture(config: RunConfig) -> Path:
    """Write the canonical artifacts consumed by the real results widget."""

    run_dir = config.output_dir
    annotated = run_dir / "annotated"
    annotated.mkdir(parents=True)
    Image.new("RGB", (3, 2), color=(20, 40, 60)).save(annotated / "000000.png")

    run_id = "acceptance-run-id"
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "configuration": {
            "input_path": str(config.input_path.resolve()),
            "output_dir": str(run_dir.resolve()),
            "regions": [region.to_dict() for region in config.regions],
            "execution_regions": [
                Region("full-frame", 0, 0, 200, 100).to_dict(),
                *(region.to_dict() for region in config.regions),
            ],
            "threshold": config.threshold,
            "max_frames": config.max_frames,
            "warmup_runs": config.warmup_runs,
            "annotate_every": config.annotate_every,
            "detector_id": config.detector_id,
            "device": config.device,
            "batch_size": 1,
        },
        "source_video": {
            "path": str(config.input_path.resolve()),
            "sha256": "a" * 64,
            "frame_width": 200,
            "frame_height": 100,
            "capture": None,
        },
        "host": {},
        "detector": {
            "adapter": "tests.fake",
            "model_id": "tests/fake-detector",
            "revision": "test-revision",
            "weights_sha256": "b" * 64,
            "backend": "fake",
            "backend_version": "1.0",
            "device": "cpu",
            "dtype": "float32",
        },
        "dependencies": {},
        "timing_definitions": {},
    }
    summary = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "status": "complete",
        "frames_processed": 3,
        "inference_count": 6,
        "annotated_frame_count": 1,
        "latency_ms": {
            "full_frame": _latency(3, 0.5, 0.8, 1.0),
            "crop": _latency(3, 0.2, 0.3, 0.4),
            "complete_frame": _latency(3, 1.0, 1.5, 2.0),
        },
        "stage_latency_ms": {},
        "hardware_peaks": {
            "process_rss_bytes": 4096,
            "system_memory_used_bytes": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_bytes": None,
            "gpu_power_watts": None,
            "gpu_temperature_c": None,
        },
        "detector_peak_device_memory_bytes": 8192,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


class FakeAcceptanceProcess(QObject):
    """Offline QProcess boundary that emits canonical worker protocol traffic."""

    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    finished = Signal(int, QProcess.ExitStatus)
    errorOccurred = Signal(QProcess.ProcessError)

    def __init__(self, completed_run_factory: Callable[[RunConfig], Path]) -> None:
        super().__init__()
        self.program: str | None = None
        self.arguments: list[str] | None = None
        self.start_count = 0
        self.kill_count = 0
        self._completed_run_factory = completed_run_factory
        self._stdout = bytearray()
        self._stderr = bytearray()

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, arguments: list[str]) -> None:
        self.arguments = list(arguments)

    def start(self) -> None:
        self.start_count += 1
        QTimer.singleShot(0, self._complete_success)

    def kill(self) -> None:
        self.kill_count += 1

    def readAllStandardOutput(self) -> QByteArray:
        data = QByteArray(bytes(self._stdout))
        self._stdout.clear()
        return data

    def readAllStandardError(self) -> QByteArray:
        data = QByteArray(bytes(self._stderr))
        self._stderr.clear()
        return data

    def _complete_success(self) -> None:
        if self.arguments is None:
            raise RuntimeError("process arguments were not set")
        config_argument = self.arguments.index("--config") + 1
        config = load_run_config(Path(self.arguments[config_argument]))
        self._completed_run_factory(config)
        for event in (
            ProgressEvent("validating", 0, 0, 0.0, None),
            ProgressEvent("running", 1, 2, 1.0, None),
            ProgressEvent("complete", 3, 6, 2.0, None),
        ):
            self._stdout.extend(
                (json.dumps(event.to_dict(), allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
            )
            self.readyReadStandardOutput.emit()
        self.finished.emit(0, QProcess.ExitStatus.NormalExit)


def test_native_gui_vertical_slice_is_config_reproducible(
    qtbot: QtBot,
    tmp_path: Path,
    video_path: Path,
) -> None:
    process = FakeAcceptanceProcess(write_completed_run_fixture)
    window = MainWindow(process=process)
    qtbot.addWidget(window)
    assert window.resultsWidget.isHidden()

    window.load_video(video_path)
    source_view = window.findChild(RegionView, "source-view")
    output = window.findChild(QLineEdit, "output")
    run_button = window.findChild(QPushButton, "run-button")
    run_cli = window.findChild(QLineEdit, "run-cli")
    run_status = window.findChild(QLabel, "run-status")
    assert source_view is not None
    assert output is not None
    assert run_button is not None
    assert run_cli is not None
    assert run_status is not None

    regions = (
        Region("z-first", 10, 10, 40, 30),
        Region("a-second", 80, 50, 20, 25),
    )
    for region in regions:
        source_view.add_region(region)
    output.setText(str(tmp_path / "run"))
    assert run_button.isEnabled()
    progress_events: list[ProgressEvent] = []
    window.runController.progressChanged.connect(progress_events.append)
    config_path = (tmp_path / "run.experiment.json").resolve()

    with qtbot.waitSignal(window.runController.runFinished, timeout=1_000):
        qtbot.mouseClick(run_button, Qt.MouseButton.LeftButton)
        assert process.start_count == 1
        assert window.resultsWidget.isHidden()
        assert run_status.text() == "Run status: Running"
        assert process.program == sys.executable
        assert process.arguments == [
            "-m",
            "edge_perception.worker",
            "--config",
            str(config_path),
            "--cancel-file",
            str((tmp_path / ".run.cancel").resolve()),
        ]
        assert window.runController.config_path == config_path
        published_bytes = config_path.read_bytes()

    config = load_run_config(config_path)
    assert config.input_path == video_path.resolve()
    assert config.output_dir == (tmp_path / "run").resolve()
    assert config.regions == regions
    assert config.capture is None
    assert window.runController.last_config == config
    assert config_path.read_bytes() == published_bytes
    assert render_run_cli(config_path) == (
        "edge-perception",
        "run",
        "--config",
        str(config_path.resolve()),
    )
    assert run_cli.text() == shlex.join(render_run_cli(config_path))
    assert progress_events == [
        ProgressEvent("validating", 0, 0, 0.0, None),
        ProgressEvent("running", 1, 2, 1.0, None),
        ProgressEvent("complete", 3, 6, 2.0, None),
    ]
    assert not window.resultsWidget.isHidden()
    assert window.resultsWidget.statusLabel.text() == "Completed"
    assert run_status.text() == "Run status: Completed"
    assert window.resultsWidget.sourceLabel.text() == str(video_path.resolve())
    assert window.resultsWidget.regionsTable.rowCount() == len(regions)
    assert [
        [window.resultsWidget.regionsTable.item(row, column).text() for column in range(5)]
        for row in range(window.resultsWidget.regionsTable.rowCount())
    ] == [
        ["z-first", "10", "10", "40", "30"],
        ["a-second", "80", "50", "20", "25"],
    ]
    assert window.resultsWidget.annotationList.count() == 1
    assert window.resultsWidget.annotationList.item(0).text() == "000000.png"
