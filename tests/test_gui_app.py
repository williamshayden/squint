from __future__ import annotations

import json
import os
import shlex
import socket
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QObject, QPointF, QProcess, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsView,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
)
from pytestqt.qtbot import QtBot

from edge_perception.capture import (
    CameraDeviceInfo,
    CameraFormatInfo,
    select_camera_format,
)
from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    render_run_cli,
)
from edge_perception.contracts import Region
from edge_perception.detectors import registry as detector_registry
from edge_perception.gui import main_window
from edge_perception.gui import run_controller as run_controller_module
from edge_perception.gui.main_window import MainWindow
from edge_perception.gui.region_view import RegionView
from edge_perception.gui.results import ResultsWidget
from edge_perception.gui.run_controller import MALFORMED_PROGRESS_ERROR
from edge_perception.progress import ProgressEvent
from edge_perception.video import DecodedFrame


def _preview_frame(width: int = 200, height: int = 100) -> DecodedFrame:
    return DecodedFrame(
        frame_index=0,
        source_time_ms=0.0,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _stub_first_frame(monkeypatch: pytest.MonkeyPatch, frame: DecodedFrame) -> None:
    monkeypatch.setattr(main_window, "first_video_frame", lambda _path: frame)


def _draw_on_viewport(
    qtbot: QtBot,
    view: RegionView,
    start: QPointF,
    finish: QPointF,
) -> None:
    view.resetTransform()
    start_pos = view.mapFromScene(start)
    midpoint = view.mapFromScene(
        QPointF((start.x() + finish.x()) / 2.0, (start.y() + finish.y()) / 2.0)
    )
    finish_pos = view.mapFromScene(finish)
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start_pos)
    qtbot.mouseMove(view.viewport(), pos=midpoint)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=finish_pos)


class FakeWindowCaptureController(QObject):
    devicesChanged = Signal()
    previewStarted = Signal(object)
    previewStopped = Signal()
    recordingStarted = Signal()
    recordingFinished = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.formats = (
            CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object()),
            CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),
        )
        self.device = CameraDeviceInfo("camera-1", "EMEET", self.formats, object())
        self.preview_requests: list[CaptureRequest] = []
        self.video_outputs: list[object] = []
        self.record_calls = 0
        self.stop_recording_calls = 0
        self.stop_preview_calls = 0
        self.discard_calls = 0
        self.preview_active = False
        self.recording_active = False
        self.devices_calls = 0
        self.defer_preview_started = False
        self.preview_start_error: Exception | None = None
        self.preview_start_error_after_signal: Exception | None = None
        self.invalid_preview_started_value = False
        self.stop_preview_error: Exception | None = None

    @property
    def is_recording(self) -> bool:
        return self.recording_active

    def devices(self) -> tuple[CameraDeviceInfo, ...]:
        self.devices_calls += 1
        return (self.device,)

    def start_preview(
        self,
        request: CaptureRequest,
        video_output: object,
    ) -> CameraFormatInfo:
        self.preview_requests.append(request)
        self.video_outputs.append(video_output)
        if self.preview_start_error is not None:
            self.preview_active = True
            raise self.preview_start_error
        self.preview_active = True
        selected = select_camera_format(self.formats, request)
        if not self.defer_preview_started:
            self.previewStarted.emit(object() if self.invalid_preview_started_value else selected)
        if self.preview_start_error_after_signal is not None:
            raise self.preview_start_error_after_signal
        return selected

    def emit_preview_started(self) -> None:
        request = self.preview_requests[-1]
        self.previewStarted.emit(select_camera_format(self.formats, request))

    def stop_preview(self) -> None:
        self.stop_preview_calls += 1
        if self.stop_preview_error is not None:
            raise self.stop_preview_error
        if self.preview_active:
            self.preview_active = False
            self.previewStopped.emit()

    def start_recording(self, _final_path: Path | None = None) -> None:
        self.record_calls += 1
        self.recording_active = True
        self.recordingStarted.emit()

    def stop_recording(self) -> None:
        self.stop_recording_calls += 1

    def discard(self) -> None:
        self.discard_calls += 1
        self.recording_active = False
        self.stop_preview()

    def complete(self, result: CaptureResult) -> None:
        self.recording_active = False
        self.preview_active = False
        self.previewStopped.emit()
        self.recordingFinished.emit(result)

    def finish_before_preview_stopped(self, result: CaptureResult) -> None:
        self.recording_active = False
        self.recordingFinished.emit(result)

    def emit_late_preview_stopped(self) -> None:
        self.preview_active = False
        self.previewStopped.emit()

    def fail(self, message: str) -> None:
        self.recording_active = False
        self.preview_active = False
        self.errorOccurred.emit(message)


class FakeGuiProcess(QObject):
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    finished = Signal(int, QProcess.ExitStatus)
    errorOccurred = Signal(QProcess.ProcessError)

    def __init__(self) -> None:
        super().__init__()
        self.program: str | None = None
        self.arguments: list[str] | None = None
        self.start_count = 0
        self.kill_count = 0
        self._stdout = bytearray()
        self._stderr = bytearray()

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, arguments: list[str]) -> None:
        self.arguments = list(arguments)

    def start(self) -> None:
        self.start_count += 1

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

    def emit_stdout(self, data: str) -> None:
        self._stdout.extend(data.encode("utf-8"))
        self.readyReadStandardOutput.emit()

    def finish(
        self,
        exit_code: int = 0,
        exit_status: QProcess.ExitStatus = QProcess.ExitStatus.NormalExit,
    ) -> None:
        self.finished.emit(exit_code, exit_status)

    def emit_error(self, error: QProcess.ProcessError) -> None:
        self.errorOccurred.emit(error)


class FakeKillTimer(QObject):
    timeout = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.single_shot: bool | None = None
        self.start_intervals: list[int] = []
        self.stop_count = 0

    def setSingleShot(self, single_shot: bool) -> None:
        self.single_shot = single_shot

    def start(self, interval: int) -> None:
        self.start_intervals.append(interval)

    def stop(self) -> None:
        self.stop_count += 1

    def trigger(self) -> None:
        self.timeout.emit()


def _progress_record(event: ProgressEvent) -> str:
    return json.dumps(event.to_dict(), allow_nan=False, sort_keys=True) + "\n"


def _capture_result(request: CaptureRequest, path: Path) -> CaptureResult:
    return CaptureResult(
        request=request,
        selected_width=1280,
        selected_height=720,
        selected_min_fps=15.0,
        selected_max_fps=60.0,
        selected_pixel_format="NV12",
        actual_width=64,
        actual_height=48,
        actual_fps=29.97,
        container="mp4",
        codec="h264",
        duration_seconds=2.0,
        has_audio=False,
        file_size_bytes=7,
        path=path,
        sha256="a" * 64,
    )


def _finish_fake_capture(
    qtbot: QtBot,
    window: MainWindow,
    controller: FakeWindowCaptureController,
    path: Path,
) -> CaptureResult:
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    result = _capture_result(controller.preview_requests[-1], path)
    controller.complete(result)
    return result


def _capture_metadata_values(window: MainWindow) -> tuple[str, str, str, str]:
    applied = window.findChild(QLabel, "capture-selected-format")
    request = window.findChild(QLabel, "capture-requested-metadata")
    recorded = window.findChild(QLabel, "capture-actual-metadata")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert applied is not None
    assert request is not None
    assert recorded is not None
    assert checksum is not None
    return applied.text(), request.text(), recorded.text(), checksum.text()


def _write_completed_run(run_dir: Path) -> Path:
    annotated = run_dir / "annotated"
    annotated.mkdir(parents=True)
    run_id = "gui-run-id"
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "configuration": {
            "regions": [],
            "threshold": 0.3,
        },
        "source_video": {
            "path": str((run_dir.parent / "historical-source.mp4").resolve()),
            "frame_width": 640,
            "frame_height": 480,
            "capture": None,
        },
        "detector": {
            "model_id": "tests/fake-detector",
            "revision": "test-revision",
            "device": "cpu",
        },
    }
    empty_latency = {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    summary = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "status": "complete",
        "frames_processed": 0,
        "inference_count": 0,
        "annotated_frame_count": 0,
        "latency_ms": {"complete_frame": empty_latency},
        "hardware_peaks": {
            "process_rss_bytes": None,
            "gpu_memory_used_bytes": None,
        },
        "detector_peak_device_memory_bytes": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_main_window_is_one_native_qmain_window(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window, QMainWindow)
    assert window.objectName() == "edge-perception-main-window"
    assert window.findChild(QGraphicsView, "source-view") is not None
    assert window.findChild(QPushButton, "run-button") is not None


def test_workflow_state_labels_and_source_browse_start_with_explicit_values(
    qtbot: QtBot,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    run_readiness = window.findChild(QLabel, "run-readiness-status")
    run_status = window.findChild(QLabel, "run-status")
    browse_source = window.findChild(QPushButton, "browse-source-button")
    capture_sha256 = window.findChild(QLabel, "capture-sha256")
    assert source_status is not None
    assert acquisition_status is not None
    assert run_readiness is not None
    assert run_status is not None
    assert browse_source is not None
    assert capture_sha256 is not None

    assert source_status.text() == "Source status: No source"
    assert acquisition_status.text() == "Acquisition status: Idle"
    assert run_readiness.text() == "Run readiness: Not ready: select a source"
    assert run_status.text() == "Run status: Not started"
    assert browse_source.text() == "Browse…"
    assert capture_sha256.text() == "—"
    assert capture_sha256.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_workflow_language_uses_the_approved_native_research_terms(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    labels = {label.text() for label in window.findChildren(QLabel)}
    groups = {group.title() for group in window.findChildren(QGroupBox)}
    buttons = {button.text() for button in window.findChildren(QPushButton)}
    checkboxes = {checkbox.text() for checkbox in window.findChildren(QCheckBox)}
    visible_language = labels | groups | buttons | checkboxes
    expected = {
        "Source path",
        "Frame size",
        "Camera acquisition",
        "Device",
        "Frame rate",
        "Require specified frame size and rate",
        "Applied camera format",
        "Capture request",
        "Recorded format",
        "Regions of interest (ROIs)",
        "Add ROI",
        "Remove ROI",
        "Compute device",
        "Confidence threshold",
        "Warm-up iterations",
        "Annotation interval (frames)",
        "Output directory",
        "Run configuration",
        "CLI command",
        "Start recording",
        "Stop recording",
        "Cancel run",
    }

    assert expected <= visible_language
    strict = window.findChild(QCheckBox, "capture-strict")
    assert strict is not None
    assert strict.toolTip() == (
        "Reject the capture if any specified width, height, or frame-rate value is not met."
    )


def test_run_readiness_reports_output_then_config_collision_then_ready(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    readiness = window.findChild(QLabel, "run-readiness-status")
    output = window.findChild(QLineEdit, "output")
    assert readiness is not None
    assert output is not None

    output.setText(str(tmp_path / "run"))
    assert readiness.text() == "Run readiness: Not ready: select a source"
    window.load_video(tmp_path / "source.mp4")
    output.clear()
    assert readiness.text() == (
        "Run readiness: Not ready: choose an empty output directory"
    )

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "occupied.json").write_text("{}", encoding="utf-8")
    config_path = (tmp_path / "run.experiment.json").resolve()
    config_path.write_text("{}", encoding="utf-8")
    output.setText(str(output_dir))
    assert readiness.text() == (
        "Run readiness: Not ready: choose an empty output directory"
    )

    (output_dir / "occupied.json").unlink()
    output.setText(str(tmp_path / "temporary"))
    output.setText(str(output_dir))
    assert readiness.text() == (
        "Run readiness: Not ready: choose an output directory without an existing "
        "run configuration"
    )

    config_path.unlink()
    output.setText(str(tmp_path / "temporary"))
    output.setText(str(output_dir))
    assert readiness.text() == "Run readiness: Ready"


def test_run_readiness_prioritizes_active_camera_acquisition(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    readiness = window.findChild(QLabel, "run-readiness-status")
    assert source_mode is not None
    assert preview is not None
    assert readiness is not None

    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert readiness.text() == (
        "Run readiness: Not ready: finish camera acquisition"
    )


def test_gui_run_mode_is_camera_model_and_network_lazy(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"model": 0, "network": 0}

    def model_sentinel(*_args: object, **_kwargs: object) -> object:
        calls["model"] += 1
        raise AssertionError("completed-run viewing loaded a model")

    def network_sentinel(*_args: object, **_kwargs: object) -> object:
        calls["network"] += 1
        raise AssertionError("completed-run viewing accessed the network")

    monkeypatch.setattr(detector_registry, "load_detector", model_sentinel)
    monkeypatch.setattr(socket, "create_connection", network_sentinel)
    controller = FakeWindowCaptureController()
    run_dir = _write_completed_run(tmp_path / "completed")

    window = MainWindow(run_dir=run_dir, capture_controller=controller)
    qtbot.addWidget(window)

    results = window.findChild(ResultsWidget, "results-widget")
    output = window.findChild(QLineEdit, "output")
    assert results is not None
    assert output is not None
    assert results.isHidden() is False
    assert results.statusLabel.text() == "complete"
    assert output.text() == ""
    assert controller.devices_calls == 0
    assert calls == {"model": 0, "network": 0}


def test_constructor_run_mode_surfaces_invalid_artifacts_and_keeps_controls_usable(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path / "invalid-completed")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    window = MainWindow(run_dir=run_dir)
    qtbot.addWidget(window)

    expected = (
        "completed run could not be loaded: "
        "manifest.json.schema_version is unsupported: 9.9.9"
    )
    output = window.findChild(QLineEdit, "output")
    source_mode = window.findChild(QComboBox, "source-mode")
    open_action = window.findChild(QAction, "open-video-action")
    assert messages == [("Run failed", expected)]
    assert window.statusBar().currentMessage() == expected
    assert output is not None and output.isEnabled()
    assert source_mode is not None and source_mode.isEnabled()
    assert open_action is not None and open_action.isEnabled()


def test_region_view_owns_one_raw_video_item_then_restores_rgb_frame(
    qtbot: QtBot,
) -> None:
    view = RegionView()
    qtbot.addWidget(view)

    first = view.begin_video_preview(1280, 720)
    second = view.begin_video_preview(1920, 1080)

    video_items = [item for item in view.scene().items() if isinstance(item, QGraphicsVideoItem)]
    assert first is not second
    assert video_items == [second]
    assert view.scene().sceneRect().width() == 1920.0
    assert view.scene().sceneRect().height() == 1080.0

    view.set_rgb_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    assert not any(isinstance(item, QGraphicsVideoItem) for item in view.scene().items())
    assert sum(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items()) == 1
    assert view.scene().sceneRect().width() == 64.0
    assert view.scene().sceneRect().height() == 48.0


def test_camera_controls_map_auto_independently_and_show_selected_mode(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    width = window.findChild(QComboBox, "capture-width")
    height = window.findChild(QComboBox, "capture-height")
    fps = window.findChild(QComboBox, "capture-fps")
    strict = window.findChild(QCheckBox, "capture-strict")
    preview = window.findChild(QPushButton, "start-preview-button")
    selected_label = window.findChild(QLabel, "capture-selected-format")
    assert all(
        widget is not None
        for widget in (source_mode, width, height, fps, strict, preview, selected_label)
    )
    assert source_mode is not None
    assert width is not None
    assert height is not None
    assert fps is not None
    assert strict is not None
    assert preview is not None
    assert selected_label is not None
    source_mode.setCurrentText("Camera")
    width.setCurrentIndex(width.findData(1920))
    height.setCurrentIndex(height.findData(None))
    fps.setCurrentIndex(fps.findData(60.0))
    strict.setChecked(False)

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert controller.preview_requests == [
        CaptureRequest("camera-1", "EMEET", 1920, None, 60.0, False)
    ]
    assert len(controller.video_outputs) == 1
    assert isinstance(controller.video_outputs[0], QGraphicsVideoItem)
    assert selected_label.text() == "1920 × 1080 px · YUYV · 30–30 FPS"


def test_camera_record_and_stop_transitions_are_explicit(qtbot: QtBot) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    stop = window.findChild(QPushButton, "stop-recording-button")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert stop is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert record.isEnabled() is True
    assert stop.isEnabled() is False
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    assert controller.record_calls == 1
    assert record.isEnabled() is False
    assert stop.isEnabled() is True

    qtbot.mouseClick(stop, Qt.MouseButton.LeftButton)

    assert controller.stop_recording_calls == 1
    assert stop.isEnabled() is False


def test_source_mode_change_preserves_active_source_frame_and_ordered_rois(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    view = window.findChild(RegionView, "source-view")
    source_status = window.findChild(QLabel, "source-status")
    assert source_mode is not None
    assert view is not None
    assert source_status is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    regions = (Region("left", 1, 2, 10, 11), Region("right", 20, 3, 12, 13))
    for region in regions:
        view.add_region(region)

    source_mode.setCurrentText("Camera")
    source_mode.setCurrentText("Video file")

    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert source_status.text() == "Source status: Ready"
    assert view.regions() == regions
    assert view.scene().sceneRect().size().toSize().width() == 64
    assert view.scene().sceneRect().size().toSize().height() == 48
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())


def test_failed_file_decode_preserves_camera_source_capture_frame_and_ordered_rois(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replacement = tmp_path / "replacement.mp4"

    def decode(path: Path) -> DecodedFrame:
        if path == replacement:
            raise ValueError("decode failed")
        return _preview_frame(width=64, height=48)

    monkeypatch.setattr(main_window, "first_video_frame", decode)
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    view = window.findChild(RegionView, "source-view")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert view is not None
    assert checksum is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    capture = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")
    controller.complete(capture)
    regions = (Region("first", 1, 2, 10, 11), Region("second", 20, 3, 12, 13))
    for region in regions:
        view.add_region(region)

    with pytest.raises(ValueError, match="^decode failed$"):
        window.load_video(replacement)

    assert source_mode.currentText() == "Camera"
    assert window.findChild(QLabel, "source-path").text() == str(capture.path.resolve())
    assert checksum.text() == capture.sha256
    assert view.regions() == regions
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())
    output = window.findChild(QLineEdit, "output")
    assert output is not None
    output.setText(str(tmp_path / "run"))
    assert window.resolved_config().capture == capture
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    open_action = window.findChild(QAction, "open-video-action")
    assert open_action is not None
    open_action.trigger()
    assert window.resolved_config().capture == capture
    assert view.regions() == regions


def test_preview_invalidates_source_only_after_preview_started_signal(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.defer_preview_started = True
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    open_action = window.findChild(QAction, "open-video-action")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    assert source_status is not None
    assert acquisition_status is not None
    assert open_action is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert preview.isEnabled() is False
    assert source_mode.isEnabled() is False
    assert open_action.isEnabled() is False
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert source_status.text() == "Source status: Ready"
    assert view.regions() == (region,)
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())

    controller.emit_preview_started()

    assert source_status.text() == "Source status: No source"
    assert acquisition_status.text() == "Acquisition status: Previewing"
    assert view.regions() == ()
    assert any(isinstance(item, QGraphicsVideoItem) for item in view.scene().items())
    assert source_mode.isEnabled() is True
    assert open_action.isEnabled() is True


def test_preview_startup_failure_restores_prior_camera_source_and_ordered_rois(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    view = window.findChild(RegionView, "source-view")
    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert view is not None
    assert source_status is not None
    assert acquisition_status is not None
    assert checksum is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    capture = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")
    controller.complete(capture)
    regions = (Region("first", 1, 2, 10, 11), Region("second", 20, 3, 12, 13))
    for region in regions:
        view.add_region(region)
    controller.preview_start_error = RuntimeError("preview failed")

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert window.findChild(QLabel, "source-path").text() == str(capture.path.resolve())
    assert source_status.text() == "Source status: Ready"
    assert acquisition_status.text() == "Acquisition status: Failed"
    assert checksum.text() == capture.sha256
    assert view.regions() == regions
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())


def test_partially_started_preview_graph_is_stopped_before_source_rollback(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.preview_start_error = RuntimeError("graph startup failed")
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    assert acquisition_status is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert controller.stop_preview_calls == 1
    assert acquisition_status.text() == "Acquisition status: Failed"
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)
    assert window.statusBar().currentMessage() == "graph startup failed"


def test_synchronous_preview_started_then_start_failure_rolls_back_committed_source(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.preview_start_error_after_signal = RuntimeError("start returned failure")
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    assert source_status is not None
    assert acquisition_status is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert controller.stop_preview_calls == 1
    assert source_status.text() == "Source status: Ready"
    assert acquisition_status.text() == "Acquisition status: Failed"
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())


def test_invalid_preview_started_format_stops_graph_and_restores_source(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.invalid_preview_started_value = True
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert controller.stop_preview_calls == 1
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)
    assert window.statusBar().currentMessage() == (
        "camera returned an invalid selected format"
    )


def test_preview_commit_failure_stops_graph_and_restores_source(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")
    monkeypatch.setattr(
        view,
        "commit_video_preview",
        lambda: (_ for _ in ()).throw(RuntimeError("preview commit failed")),
    )

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert controller.stop_preview_calls == 1
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)
    assert window.statusBar().currentMessage() == "preview commit failed"


def test_asynchronous_precommit_error_cleans_pending_graph_and_ignores_late_start(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.defer_preview_started = True
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    assert acquisition_status is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    controller.fail("asynchronous preview failed")
    controller.emit_preview_started()

    assert controller.stop_preview_calls == 1
    assert acquisition_status.text() == "Acquisition status: Failed"
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())


def test_preview_rollback_uses_in_memory_frame_if_source_can_no_longer_decode(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.preview_start_error_after_signal = RuntimeError("start returned failure")
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    region = Region("roi", 1, 2, 10, 11)
    view.add_region(region)
    source_mode.setCurrentText("Camera")
    monkeypatch.setattr(
        main_window,
        "first_video_frame",
        lambda _path: (_ for _ in ()).throw(ValueError("source disappeared")),
    )

    with qtbot.captureExceptions() as exceptions:
        qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert exceptions == []
    assert window.findChild(QLabel, "source-path").text() == str(source)
    assert view.regions() == (region,)


def test_preview_restoration_failure_fails_closed_without_losing_primary_error(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.preview_start_error_after_signal = RuntimeError("start returned failure")
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    view = window.findChild(RegionView, "source-view")
    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert view is not None
    assert source_status is not None
    assert acquisition_status is not None
    window.load_video(tmp_path / "source.mp4")
    source_mode.setCurrentText("Camera")
    monkeypatch.setattr(
        view,
        "set_rgb_frame",
        lambda _image: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )

    with qtbot.captureExceptions() as exceptions:
        qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert exceptions == []
    assert source_status.text() == "Source status: No source"
    assert acquisition_status.text() == "Acquisition status: Failed"
    assert window.findChild(QLabel, "source-path").text() == "—"
    assert window.statusBar().currentMessage() == (
        "start returned failure; rollback failed: restore failed"
    )


def test_preview_cleanup_failure_is_context_without_replacing_primary_error(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    controller.preview_start_error = RuntimeError("graph startup failed")
    controller.stop_preview_error = RuntimeError("stop failed")
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    source_status = window.findChild(QLabel, "source-status")
    assert source_mode is not None
    assert preview is not None
    assert source_status is not None
    source = tmp_path / "source.mp4"
    window.load_video(source)
    source_mode.setCurrentText("Camera")

    with qtbot.captureExceptions() as exceptions:
        qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    controller.preview_active = False
    assert exceptions == []
    assert source_status.text() == "Source status: No source"
    assert window.findChild(QLabel, "source-path").text() == "—"
    assert window.statusBar().currentMessage() == (
        "graph startup failed; cleanup failed: stop failed"
    )


def test_stop_recording_reports_finalizing_until_capture_finishes(qtbot: QtBot) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    stop = window.findChild(QPushButton, "stop-recording-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    run_readiness = window.findChild(QLabel, "run-readiness-status")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert stop is not None
    assert acquisition_status is not None
    assert run_readiness is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)

    qtbot.mouseClick(stop, Qt.MouseButton.LeftButton)

    controller.recording_active = False
    assert acquisition_status.text() == "Acquisition status: Finalizing"
    assert run_readiness.text() == "Run readiness: Not ready: finish camera acquisition"


def test_successful_capture_stays_in_camera_mode_and_exposes_finalized_checksum(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    source_status = window.findChild(QLabel, "source-status")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert source_status is not None
    assert acquisition_status is not None
    assert checksum is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    capture = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")

    controller.complete(capture)

    assert source_mode.currentText() == "Camera"
    assert source_status.text() == "Source status: Ready"
    assert acquisition_status.text() == "Acquisition status: Finalized"
    assert checksum.text() == capture.sha256
    output = window.findChild(QLineEdit, "output")
    assert output is not None
    output.setText(str(tmp_path / "run"))
    source_mode.setCurrentText("Video file")
    assert window.resolved_config().capture == capture
    source_mode.setCurrentText("Camera")
    assert window.resolved_config().capture == capture
    window.load_video(tmp_path / "ordinary.mp4")
    assert checksum.text() == "—"


def test_capture_failure_reports_failed_without_promoting_an_incomplete_source(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    source_status = window.findChild(QLabel, "source-status")
    assert source_mode is not None
    assert preview is not None
    assert acquisition_status is not None
    assert source_status is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    controller.errorOccurred.emit("capture failed")

    assert acquisition_status.text() == "Acquisition status: Failed"
    assert source_status.text() == "Source status: No source"
    assert window.findChild(QLabel, "source-path").text() != "capture failed"


def test_ordinary_file_commit_clears_every_camera_metadata_field(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    _finish_fake_capture(qtbot, window, controller, tmp_path / "capture.mp4")
    assert _capture_metadata_values(window) == (
        "1280 × 720 px · NV12 · 15–60 FPS",
        "width Auto · height Auto · FPS Auto · normal",
        "64 × 48 px · 29.97 FPS · mp4/h264 · no audio",
        "a" * 64,
    )

    window.load_video(tmp_path / "ordinary.mp4")

    assert _capture_metadata_values(window) == ("—", "—", "—", "—")


def test_pending_preview_preserves_finalized_metadata_then_commit_reowns_fields(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    _finish_fake_capture(qtbot, window, controller, tmp_path / "capture.mp4")
    finalized_metadata = _capture_metadata_values(window)
    width = window.findChild(QComboBox, "capture-width")
    preview = window.findChild(QPushButton, "start-preview-button")
    assert width is not None
    assert preview is not None
    width.setCurrentIndex(width.findData(1920))
    controller.defer_preview_started = True

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert _capture_metadata_values(window) == finalized_metadata
    controller.emit_preview_started()
    assert _capture_metadata_values(window) == (
        "1920 × 1080 px · YUYV · 30–30 FPS",
        "—",
        "—",
        "—",
    )


def test_capture_failure_after_source_invalidation_does_not_show_old_metadata(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    _finish_fake_capture(qtbot, window, controller, tmp_path / "capture.mp4")
    width = window.findChild(QComboBox, "capture-width")
    preview = window.findChild(QPushButton, "start-preview-button")
    assert width is not None
    assert preview is not None
    width.setCurrentIndex(width.findData(1920))
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    controller.fail("capture failed")

    assert _capture_metadata_values(window) == (
        "1920 × 1080 px · YUYV · 30–30 FPS",
        "—",
        "—",
        "—",
    )
    assert window.findChild(QLabel, "source-status").text() == (
        "Source status: No source"
    )


def test_failed_preview_startup_preserves_every_prior_source_metadata_field(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    _finish_fake_capture(qtbot, window, controller, tmp_path / "capture.mp4")
    finalized_metadata = _capture_metadata_values(window)
    controller.preview_start_error = RuntimeError("graph startup failed")
    preview = window.findChild(QPushButton, "start-preview-button")
    assert preview is not None

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert _capture_metadata_values(window) == finalized_metadata
    assert window.findChild(QLabel, "source-status").text() == "Source status: Ready"


def test_camera_group_control_state_follows_mode_and_recording_guard(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    group = window.findChild(QGroupBox, "camera-controls")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    assert source_mode is not None
    assert group is not None
    assert preview is not None
    assert record is not None
    assert group.isEnabled() is False

    source_mode.setCurrentText("Camera")
    assert group.isEnabled() is True
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    source_mode.setCurrentText("Video file")

    guarded_state = source_mode.currentText(), group.isEnabled()
    controller.fail("test cleanup")
    assert guarded_state == ("Camera", True)


def test_acquisition_error_then_finish_keeps_failed_and_rejects_published_result(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    source_status = window.findChild(QLabel, "source-status")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert acquisition_status is not None
    assert source_status is not None
    assert checksum is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    result = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")

    controller.fail("capture published but cleanup failed")
    controller.recordingFinished.emit(result)

    assert acquisition_status.text() == "Acquisition status: Failed"
    assert source_status.text() == "Source status: No source"
    assert window.findChild(QLabel, "source-path").text() == "—"
    assert checksum.text() == "—"
    assert preview.isEnabled() is True


def test_acquisition_finish_then_error_keeps_finalized_source(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    source_status = window.findChild(QLabel, "source-status")
    checksum = window.findChild(QLabel, "capture-sha256")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert acquisition_status is not None
    assert source_status is not None
    assert checksum is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    result = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")

    controller.finish_before_preview_stopped(result)
    controller.errorOccurred.emit("late camera error")

    assert acquisition_status.text() == "Acquisition status: Finalized"
    assert source_status.text() == "Source status: Ready"
    assert checksum.text() == "a" * 64


def test_recording_finish_clears_activity_before_preview_stopped_and_stays_finalized(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    stop = window.findChild(QPushButton, "stop-recording-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert stop is not None
    assert acquisition_status is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    result = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")

    controller.finish_before_preview_stopped(result)

    state_before_preview_stopped = (
        acquisition_status.text(),
        preview.isEnabled(),
        record.isEnabled(),
        stop.isEnabled(),
    )
    controller.emit_late_preview_stopped()
    assert state_before_preview_stopped == (
        "Acquisition status: Finalized",
        True,
        False,
        False,
    )
    assert acquisition_status.text() == "Acquisition status: Finalized"


def test_production_preview_stopped_then_finish_reaches_finalized(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert acquisition_status is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    result = _capture_result(controller.preview_requests[-1], tmp_path / "capture.mp4")

    controller.complete(result)

    assert acquisition_status.text() == "Acquisition status: Finalized"


def test_mode_change_stopping_an_ordinary_preview_returns_acquisition_to_idle(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    acquisition_status = window.findChild(QLabel, "acquisition-status")
    run_readiness = window.findChild(QLabel, "run-readiness-status")
    assert source_mode is not None
    assert preview is not None
    assert acquisition_status is not None
    assert run_readiness is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    assert acquisition_status.text() == "Acquisition status: Previewing"

    source_mode.setCurrentText("Video file")

    assert acquisition_status.text() == "Acquisition status: Idle"
    assert run_readiness.text() == "Run readiness: Not ready: select a source"


def test_loading_file_stops_preview_before_replacing_source(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    assert source_mode is not None
    assert preview is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    video_path = tmp_path / "replacement.mp4"
    window.load_video(video_path)

    assert controller.stop_preview_calls == 1
    assert controller.discard_calls == 0
    assert controller.preview_active is False
    assert source_mode.currentText() == "Video file"
    assert window.findChild(QLabel, "source-path").text() == str(video_path)


def test_loading_file_while_recording_is_rejected_without_discard(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decoded: list[Path] = []

    def decode(path: Path) -> DecodedFrame:
        decoded.append(path)
        return _preview_frame()

    monkeypatch.setattr(main_window, "first_video_frame", decode)
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)

    with pytest.raises(
        RuntimeError,
        match="^cannot replace source while camera recording is active$",
    ):
        window.load_video(tmp_path / "replacement.mp4")

    assert decoded == []
    assert controller.discard_calls == 0
    assert controller.stop_preview_calls == 0
    assert controller.recording_active is True
    assert source_mode.currentText() == "Camera"


def test_switching_to_file_mode_while_recording_restores_camera_mode(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)

    source_mode.setCurrentText("Video file")

    assert source_mode.currentText() == "Camera"
    assert controller.discard_calls == 0
    assert controller.stop_preview_calls == 0
    assert controller.recording_active is True
    assert window.statusBar().currentMessage() == (
        "Stop or discard the camera recording before replacing the source"
    )


def test_completed_camera_capture_loads_final_rgb_and_probed_metadata(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    controller = FakeWindowCaptureController()
    window = MainWindow(capture_controller=controller)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    output = window.findChild(QLineEdit, "output")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    assert output is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)
    request = controller.preview_requests[-1]
    final_path = tmp_path / "capture.mp4"
    result = CaptureResult(
        request=request,
        selected_width=1280,
        selected_height=720,
        selected_min_fps=15.0,
        selected_max_fps=60.0,
        selected_pixel_format="NV12",
        actual_width=64,
        actual_height=48,
        actual_fps=29.97,
        container="mp4",
        codec="h264",
        duration_seconds=2.0,
        has_audio=False,
        file_size_bytes=7,
        path=final_path,
        sha256="a" * 64,
    )

    controller.complete(result)

    view = window.findChild(RegionView, "source-view")
    path_label = window.findChild(QLabel, "source-path")
    requested_label = window.findChild(QLabel, "capture-requested-metadata")
    actual_label = window.findChild(QLabel, "capture-actual-metadata")
    assert view is not None
    assert path_label is not None
    assert requested_label is not None
    assert actual_label is not None
    assert source_mode.currentText() == "Camera"
    assert path_label.text() == str(final_path.resolve())
    assert requested_label.text() == "width Auto · height Auto · FPS Auto · normal"
    assert actual_label.text() == "64 × 48 px · 29.97 FPS · mp4/h264 · no audio"
    assert not any(isinstance(item, QGraphicsVideoItem) for item in view.scene().items())
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())
    output.setText(str(tmp_path / "run"))
    config = window.resolved_config()
    assert config is not None
    assert config.capture == result


def test_load_video_replaces_stale_regions_and_shows_source_metadata(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    window = MainWindow()
    qtbot.addWidget(window)
    view = window.findChild(RegionView, "source-view")
    assert view is not None
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    window.load_video(first_path)
    view.add_region(Region("stale", 1, 2, 3, 4))

    window.load_video(second_path)

    path_label = window.findChild(QLabel, "source-path")
    dimensions_label = window.findChild(QLabel, "source-dimensions")
    assert path_label is not None
    assert dimensions_label is not None
    assert path_label.text() == str(second_path)
    assert dimensions_label.text() == "64 × 48 px"
    assert view.scene().sceneRect().width() == 64.0
    assert view.scene().sceneRect().height() == 48.0
    assert view.regions() == ()


def test_load_video_replaces_selected_region_without_qt_exception(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    view = window.findChild(RegionView, "source-view")
    assert view is not None
    window.load_video(tmp_path / "first.mp4")
    view.add_region(Region("selected", 10, 20, 40, 30))
    view.select_region("selected")

    with qtbot.captureExceptions() as exceptions:
        window.load_video(tmp_path / "second.mp4")

    assert exceptions == []
    assert view.regions() == ()


def test_new_region_button_draws_clipped_region_through_viewport_mouse_events(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.load_video(Path("source.mp4"))
    view = window.findChild(RegionView, "source-view")
    region_id = window.findChild(QLineEdit, "region-id")
    new_region = window.findChild(QPushButton, "new-region-button")
    assert view is not None
    assert region_id is not None
    assert new_region is not None
    view.resetTransform()
    region_id.setText("roi")

    qtbot.mouseClick(new_region, Qt.MouseButton.LeftButton)

    assert view.regions() == ()
    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    press = view.mapFromScene(QPointF(50.0, 40.0))
    move = view.mapFromScene(QPointF(20.0, 20.0))
    release = view.mapFromScene(QPointF(-10.0, 10.0))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=press)
    qtbot.mouseMove(view.viewport(), pos=move)
    assert sum(isinstance(item, QGraphicsRectItem) for item in view.scene().items()) == 1
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=release)

    assert view.regions() == (Region("roi", 0, 10, 50, 30),)
    assert view.selected_region() == Region("roi", 0, 10, 50, 30)
    assert view.viewport().cursor().shape() == Qt.CursorShape.ArrowCursor


def test_numeric_resize_updates_scene_and_contract(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.load_video(Path("source.mp4"))
    view = window.findChild(RegionView, "source-view")
    region_id = window.findChild(QLineEdit, "region-id")
    x = window.findChild(QSpinBox, "region-x")
    y = window.findChild(QSpinBox, "region-y")
    width = window.findChild(QSpinBox, "region-width")
    height = window.findChild(QSpinBox, "region-height")
    new_region = window.findChild(QPushButton, "new-region-button")
    assert all(widget is not None for widget in (view, region_id, x, y, width, height, new_region))
    assert view is not None
    assert region_id is not None
    assert x is not None
    assert y is not None
    assert width is not None
    assert height is not None
    assert new_region is not None
    region_id.setText("roi")
    qtbot.mouseClick(new_region, Qt.MouseButton.LeftButton)
    _draw_on_viewport(qtbot, view, QPointF(10.0, 20.0), QPointF(50.0, 50.0))

    x.setValue(25)
    y.setValue(15)
    width.setValue(80)
    with qtbot.waitSignal(view.regionsChanged) as changed:
        height.setValue(45)

    assert (x.value(), y.value(), width.value(), height.value()) == (25, 15, 80, 45)
    assert view.regions() == (Region("roi", 25, 15, 80, 45),)
    assert changed.args == [(Region("roi", 25, 15, 80, 45),)]
    item = next(
        item
        for item in view.scene().items()
        if isinstance(item, QGraphicsRectItem) and item.data(0) == "roi"
    )
    assert item.mapRectToScene(item.rect()) == item.rect().translated(25.0, 15.0)


def test_region_controls_add_and_delete_selected_region(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.load_video(Path("source.mp4"))
    region_id = window.findChild(QLineEdit, "region-id")
    new_region = window.findChild(QPushButton, "new-region-button")
    delete_region = window.findChild(QPushButton, "delete-region-button")
    view = window.findChild(RegionView, "source-view")
    assert region_id is not None
    assert new_region is not None
    assert delete_region is not None
    assert view is not None
    region_id.setText("roi")

    qtbot.mouseClick(new_region, Qt.MouseButton.LeftButton)
    _draw_on_viewport(qtbot, view, QPointF(10.0, 20.0), QPointF(50.0, 50.0))
    qtbot.mouseClick(delete_region, Qt.MouseButton.LeftButton)

    assert view.regions() == ()
    assert all(item.data(0) != "roi" for item in view.scene().items())


def test_deselect_clears_and_disables_geometry_controls(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.load_video(Path("source.mp4"))
    view = window.findChild(RegionView, "source-view")
    region_id = window.findChild(QLineEdit, "region-id")
    new_region = window.findChild(QPushButton, "new-region-button")
    spins = [
        window.findChild(QSpinBox, object_name)
        for object_name in ("region-x", "region-y", "region-width", "region-height")
    ]
    assert view is not None
    assert region_id is not None
    assert new_region is not None
    assert all(spin is not None for spin in spins)
    region_id.setText("roi")
    qtbot.mouseClick(new_region, Qt.MouseButton.LeftButton)
    _draw_on_viewport(qtbot, view, QPointF(10.0, 20.0), QPointF(50.0, 50.0))

    qtbot.mouseClick(
        view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=view.mapFromScene(QPointF(180.0, 90.0)),
    )

    assert view.selected_region() is None
    assert all(spin is not None and not spin.isEnabled() and spin.text() == "" for spin in spins)
    assert region_id.isEnabled() is True
    assert region_id.text() == "roi"


def test_delete_clears_and_disables_geometry_controls(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    window.load_video(Path("source.mp4"))
    view = window.findChild(RegionView, "source-view")
    region_id = window.findChild(QLineEdit, "region-id")
    new_region = window.findChild(QPushButton, "new-region-button")
    delete_region = window.findChild(QPushButton, "delete-region-button")
    spins = [
        window.findChild(QSpinBox, object_name)
        for object_name in ("region-x", "region-y", "region-width", "region-height")
    ]
    assert view is not None
    assert region_id is not None
    assert new_region is not None
    assert delete_region is not None
    assert all(spin is not None for spin in spins)
    region_id.setText("roi")
    qtbot.mouseClick(new_region, Qt.MouseButton.LeftButton)
    _draw_on_viewport(qtbot, view, QPointF(10.0, 20.0), QPointF(50.0, 50.0))

    qtbot.mouseClick(delete_region, Qt.MouseButton.LeftButton)

    assert view.regions() == ()
    assert all(spin is not None and not spin.isEnabled() and spin.text() == "" for spin in spins)


def test_run_enables_only_with_source_and_output(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(tmp_path / "runs"))
    assert run.isEnabled() is False
    window.load_video(tmp_path / "source.mp4")
    assert run.isEnabled() is True
    output.clear()
    assert run.isEnabled() is False


def test_run_rejects_source_path_as_output(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    source = tmp_path / "source.mp4"
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(source)
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(source))

    assert run.isEnabled() is False


def test_run_rejects_existing_non_directory_output(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    output_file = tmp_path / "output.txt"
    output_file.write_text("occupied", encoding="utf-8")
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(tmp_path / "source.mp4")
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(output_file))

    assert run.isEnabled() is False


def test_run_rejects_nonempty_output_directory(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "existing.json").write_text("{}", encoding="utf-8")
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(tmp_path / "source.mp4")
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(output_dir))

    assert run.isEnabled() is False


def test_run_accepts_existing_empty_output_directory(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    window = MainWindow()
    qtbot.addWidget(window)
    window.load_video(tmp_path / "source.mp4")
    run = window.findChild(QPushButton, "run-button")
    output = window.findChild(QLineEdit, "output")
    assert run is not None
    assert output is not None

    output.setText(str(output_dir))

    assert run.isEnabled() is True


def test_file_menu_opens_selected_video(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    selected_path = tmp_path / "selected.mp4"
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(selected_path), "Video files (*.mp4)"),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    action = window.findChild(QAction, "open-video-action")
    assert action is not None
    assert action.text() == "Open Video…"

    action.trigger()

    path_label = window.findChild(QLabel, "source-path")
    assert path_label is not None
    assert path_label.text() == str(selected_path)


def test_browse_source_button_uses_the_shared_video_selection_path(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame(width=64, height=48))
    selected_path = tmp_path / "selected.mp4"
    monkeypatch.setattr(
        main_window.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(selected_path), "Video files (*.mp4)"),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    browse = window.findChild(QPushButton, "browse-source-button")
    assert browse is not None

    qtbot.mouseClick(browse, Qt.MouseButton.LeftButton)

    path_label = window.findChild(QLabel, "source-path")
    assert path_label is not None
    assert path_label.text() == str(selected_path)


def test_resolved_config_reads_ordered_regions_and_execution_controls(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    window = MainWindow()
    qtbot.addWidget(window)
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "empty-run"
    output_dir.mkdir()
    window.load_video(source)
    view = window.findChild(RegionView, "source-view")
    output = window.findChild(QLineEdit, "output")
    device = window.findChild(QComboBox, "device")
    threshold = window.findChild(QDoubleSpinBox, "threshold")
    max_frames = window.findChild(QSpinBox, "max-frames")
    warmup_runs = window.findChild(QSpinBox, "warmup-runs")
    annotate_every = window.findChild(QSpinBox, "annotate-every")
    assert all(
        widget is not None
        for widget in (
            view,
            output,
            device,
            threshold,
            max_frames,
            warmup_runs,
            annotate_every,
        )
    )
    assert view is not None
    assert output is not None
    assert device is not None
    assert threshold is not None
    assert max_frames is not None
    assert warmup_runs is not None
    assert annotate_every is not None
    view.add_region(Region("first", 1, 2, 3, 4))
    view.add_region(Region("second", 5, 6, 7, 8))
    output.setText(str(output_dir))
    device.setCurrentText("CUDA")
    threshold.setValue(0.45)
    max_frames.setValue(12)
    warmup_runs.setValue(2)
    annotate_every.setValue(3)

    config = window.resolved_config()

    assert config == RunConfig(
        input_path=source,
        output_dir=output_dir,
        regions=(Region("first", 1, 2, 3, 4), Region("second", 5, 6, 7, 8)),
        threshold=0.45,
        max_frames=12,
        warmup_runs=2,
        annotate_every=3,
        detector_id="dfine-nano-coco",
        device="cuda",
        capture=None,
    )


def test_resolved_config_requires_a_finalized_source(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    output = window.findChild(QLineEdit, "output")
    assert output is not None
    output.setText(str(tmp_path / "run"))

    with pytest.raises(ValueError, match="^finalized video source is required$"):
        window.resolved_config()


def test_gui_run_displays_reproducibility_input_and_restores_controls(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    process = FakeGuiProcess()
    window = MainWindow(process=process)
    qtbot.addWidget(window)
    source = tmp_path / "source.mp4"
    output_dir = tmp_path / "run"
    window.load_video(source)
    output = window.findChild(QLineEdit, "output")
    run = window.findChild(QPushButton, "run-button")
    cancel = window.findChild(QPushButton, "cancel-button")
    source_mode = window.findChild(QComboBox, "source-mode")
    detector = window.findChild(QComboBox, "detector")
    progress = window.findChild(QLabel, "run-progress")
    config_path_line = window.findChild(QLineEdit, "config-path")
    cli_line = window.findChild(QLineEdit, "run-cli")
    completed = window.findChild(ResultsWidget, "results-widget")
    open_action = window.findChild(QAction, "open-video-action")
    view = window.findChild(RegionView, "source-view")
    assert all(
        widget is not None
        for widget in (
            output,
            run,
            cancel,
            source_mode,
            detector,
            progress,
            config_path_line,
            cli_line,
            completed,
            open_action,
            view,
        )
    )
    assert output is not None
    assert run is not None
    assert cancel is not None
    assert source_mode is not None
    assert detector is not None
    assert progress is not None
    assert config_path_line is not None
    assert cli_line is not None
    assert completed is not None
    assert open_action is not None
    assert view is not None
    output.setText(str(output_dir))

    qtbot.mouseClick(run, Qt.MouseButton.LeftButton)

    expected_config_path = (tmp_path / "run.experiment.json").resolve()
    assert process.start_count == 1
    assert config_path_line.text() == str(expected_config_path)
    assert cli_line.text() == shlex.join(render_run_cli(expected_config_path))
    assert source_mode.isEnabled() is False
    assert detector.isEnabled() is False
    assert output.isEnabled() is False
    assert open_action.isEnabled() is False
    assert view.isEnabled() is False
    assert run.isEnabled() is False
    assert cancel.isEnabled() is True
    with pytest.raises(RuntimeError, match="^cannot replace source while a run is active$"):
        window.load_video(tmp_path / "replacement.mp4")

    running = ProgressEvent("running", 2, 3, 4.5, None)
    process.emit_stdout(_progress_record(running))

    assert progress.text() == "running: 2 frames, 3 inferences, 4.5 ms"

    terminal = ProgressEvent("complete", 4, 5, 6.5, None)
    _write_completed_run(output_dir)
    process.emit_stdout(_progress_record(terminal))
    process.finish()

    assert source_mode.isEnabled() is True
    assert detector.isEnabled() is True
    assert output.isEnabled() is True
    assert open_action.isEnabled() is True
    assert view.isEnabled() is True
    assert cancel.isEnabled() is False
    assert run.isEnabled() is False
    assert completed.isHidden() is False
    assert completed.statusLabel.text() == "complete"


def test_gui_run_failure_shows_one_message_and_status(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )
    process = FakeGuiProcess()
    window = MainWindow(process=process)
    qtbot.addWidget(window)
    window.load_video(tmp_path / "source.mp4")
    output = window.findChild(QLineEdit, "output")
    run = window.findChild(QPushButton, "run-button")
    assert output is not None
    assert run is not None
    output.setText(str(tmp_path / "run"))
    qtbot.mouseClick(run, Qt.MouseButton.LeftButton)

    process.emit_stdout("not-json\n")

    assert messages == [("Run failed", MALFORMED_PROGRESS_ERROR)]
    assert window.statusBar().currentMessage() == MALFORMED_PROGRESS_ERROR
    assert output.isEnabled() is False

    process.finish(1, QProcess.ExitStatus.CrashExit)

    assert messages == [("Run failed", MALFORMED_PROGRESS_ERROR)]
    assert output.isEnabled() is True


def test_close_while_recording_uses_injected_keep_or_discard_decision(
    qtbot: QtBot,
) -> None:
    controller = FakeWindowCaptureController()
    decisions = iter(("Keep Window Open", "Stop and Discard"))
    activities: list[str] = []

    def decide(activity: str) -> str:
        activities.append(activity)
        return next(decisions)

    window = MainWindow(capture_controller=controller, close_decision=decide)
    qtbot.addWidget(window)
    source_mode = window.findChild(QComboBox, "source-mode")
    preview = window.findChild(QPushButton, "start-preview-button")
    record = window.findChild(QPushButton, "record-button")
    assert source_mode is not None
    assert preview is not None
    assert record is not None
    source_mode.setCurrentText("Camera")
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(record, Qt.MouseButton.LeftButton)

    keep_event = QCloseEvent()
    window.closeEvent(keep_event)

    assert keep_event.isAccepted() is False
    assert controller.discard_calls == 0

    discard_event = QCloseEvent()
    window.closeEvent(discard_event)

    assert discard_event.isAccepted() is True
    assert controller.discard_calls == 1
    assert activities == ["recording", "recording"]


def test_close_while_inference_cancels_restarts_kill_timer_and_waits_for_finished(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    process = FakeGuiProcess()
    timer = FakeKillTimer()
    activities: list[str] = []

    def decide(activity: str) -> str:
        activities.append(activity)
        return "Cancel Run and Exit"

    window = MainWindow(process=process, close_decision=decide, kill_timer=timer)
    qtbot.addWidget(window)
    window.show()
    output_dir = tmp_path / "run"
    window.load_video(tmp_path / "source.mp4")
    output = window.findChild(QLineEdit, "output")
    run = window.findChild(QPushButton, "run-button")
    assert output is not None
    assert run is not None
    output.setText(str(output_dir))
    qtbot.mouseClick(run, Qt.MouseButton.LeftButton)

    first_close = QCloseEvent()
    window.closeEvent(first_close)
    second_close = QCloseEvent()
    window.closeEvent(second_close)

    cancel_path = (tmp_path / ".run.cancel").resolve()
    assert first_close.isAccepted() is False
    assert second_close.isAccepted() is False
    assert cancel_path.exists()
    assert timer.single_shot is True
    assert timer.start_intervals == [5000, 5000]
    assert activities == ["inference"]
    assert window.isVisible() is True

    timer.trigger()

    assert process.kill_count == 1
    assert window.isVisible() is True

    process.emit_stdout(_progress_record(ProgressEvent("cancelled", 1, 1, 2.0, None)))
    critical_messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: critical_messages.append((title, message)),
    )
    process.finish()

    assert timer.stop_count == 1
    assert not cancel_path.exists()
    assert critical_messages == []
    assert window.resultsWidget.isHidden()
    assert window.isVisible() is False


def test_close_without_active_work_accepts_immediately(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    event = QCloseEvent()

    window.closeEvent(event)

    assert event.isAccepted() is True


@pytest.mark.parametrize("terminal_kind", ["finished", "failed_to_start"])
def test_gui_cleanup_failure_reports_once_and_restores_controls(
    terminal_kind: str,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_first_frame(monkeypatch, _preview_frame())
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )
    process = FakeGuiProcess()
    window = MainWindow(process=process)
    qtbot.addWidget(window)
    window.load_video(tmp_path / "source.mp4")
    output = window.findChild(QLineEdit, "output")
    run = window.findChild(QPushButton, "run-button")
    cancel = window.findChild(QPushButton, "cancel-button")
    assert output is not None
    assert run is not None
    assert cancel is not None
    output.setText(str(tmp_path / "run"))
    qtbot.mouseClick(run, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(cancel, Qt.MouseButton.LeftButton)
    cancel_path = tmp_path / ".run.cancel"
    actual_replace = os.replace
    actual_unlink = Path.unlink
    quarantines: list[Path] = []

    def capture_quarantine(source: Path | str, destination: Path | str) -> None:
        actual_replace(source, destination)
        if Path(source) == cancel_path:
            quarantines.append(Path(destination))

    def fail_owned_quarantine(path: Path, *, missing_ok: bool = False) -> None:
        if quarantines and path == quarantines[-1]:
            raise OSError("locked")
        actual_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(run_controller_module.os, "replace", capture_quarantine)
    monkeypatch.setattr(Path, "unlink", fail_owned_quarantine)
    if terminal_kind == "finished":
        process.emit_stdout(_progress_record(ProgressEvent("complete", 1, 1, 2.0, None)))
        process.finish()
    else:
        process.emit_error(QProcess.ProcessError.FailedToStart)

    quarantine = quarantines[0]
    cleanup_error = f"cancellation cleanup failed while removing {quarantine}: locked"
    expected = (
        cleanup_error
        if terminal_kind == "finished"
        else f"worker failed to start\n{cleanup_error}"
    )
    assert messages == [("Run failed", expected)]
    assert window.statusBar().currentMessage() == expected
    assert output.isEnabled() is True
    assert cancel.isEnabled() is False
    assert window.runController.is_active is False

    process.finish(1, QProcess.ExitStatus.CrashExit)

    assert messages == [("Run failed", expected)]
