from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
)
from pytestqt.qtbot import QtBot

from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.contracts import Region
from edge_perception.gui import main_window
from edge_perception.gui.capture import (
    CameraDeviceInfo,
    CameraFormatInfo,
    select_camera_format,
)
from edge_perception.gui.main_window import MainWindow
from edge_perception.gui.region_view import RegionView
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
        self.preview_active = False
        self.recording_active = False

    @property
    def is_recording(self) -> bool:
        return self.recording_active

    def devices(self) -> tuple[CameraDeviceInfo, ...]:
        return (self.device,)

    def start_preview(
        self,
        request: CaptureRequest,
        video_output: object,
    ) -> CameraFormatInfo:
        self.preview_requests.append(request)
        self.video_outputs.append(video_output)
        self.preview_active = True
        selected = select_camera_format(self.formats, request)
        self.previewStarted.emit(selected)
        return selected

    def stop_preview(self) -> None:
        self.stop_preview_calls += 1
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
        self.recording_active = False
        self.stop_preview()

    def complete(self, result: CaptureResult) -> None:
        self.recording_active = False
        self.preview_active = False
        self.previewStopped.emit()
        self.recordingFinished.emit(result)


def test_main_window_is_one_native_qmain_window(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)

    assert isinstance(window, QMainWindow)
    assert window.objectName() == "edge-perception-main-window"
    assert window.findChild(QGraphicsView, "source-view") is not None
    assert window.findChild(QPushButton, "run-button") is not None


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
    assert source_mode.currentText() == "Video file"
    assert path_label.text() == str(final_path.resolve())
    assert requested_label.text() == "Requested: width Auto · height Auto · FPS Auto · normal"
    assert actual_label.text() == "Actual: 64 × 48 px · 29.97 FPS · mp4/h264 · no audio"
    assert not any(isinstance(item, QGraphicsVideoItem) for item in view.scene().items())
    assert any(isinstance(item, QGraphicsPixmapItem) for item in view.scene().items())
    output.setText(str(tmp_path / "run"))
    config = window._current_run_config()
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
