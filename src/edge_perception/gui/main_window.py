"""Native file preview and source-region controls."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path
from typing import cast

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    render_run_cli,
)
from edge_perception.contracts import Region
from edge_perception.detectors.registry import detector_descriptors
from edge_perception.gui.capture import (
    CameraDeviceInfo,
    CameraFormatInfo,
    QtCaptureController,
    select_camera_format,
)
from edge_perception.gui.region_view import RegionView
from edge_perception.gui.results import ResultsWidget
from edge_perception.gui.run_controller import RunController
from edge_perception.progress import ProgressEvent
from edge_perception.runner import validate_output_directory
from edge_perception.video import first_video_frame


class MainWindow(QMainWindow):
    """Preview a local video and edit named source-pixel regions."""

    def __init__(
        self,
        run_dir: Path | None = None,
        *,
        capture_controller: QtCaptureController | None = None,
        process: QProcess | None = None,
        close_decision: Callable[[str], str] | None = None,
        kill_timer: QTimer | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("edge-perception-main-window")
        self.setWindowTitle("Edge Perception")
        self.resize(1100, 720)
        self._source_path: Path | None = None
        self._source_width = 0
        self._source_height = 0
        self._capture_result: CaptureResult | None = None
        self._capture_controller = capture_controller
        self._capture_controller_connected = False
        self._preview_active = False
        self._recording_active = False
        self._record_start_requested = False
        self._recording_stop_requested = False
        self._updating_region_controls = False
        self._close_decision = close_decision
        self._exit_after_run = False
        self._allow_close = False
        self._run_failure_shown = False
        self.runController = RunController(process=process)
        self.runController.setParent(self)
        self._kill_timer = QTimer(self) if kill_timer is None else kill_timer
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_active_run)

        self._create_file_menu()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_view = RegionView()
        splitter.addWidget(self._source_view)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        self._source_mode_combo = self._source_mode()
        form.addRow("Source mode", self._source_mode_combo)
        self._source_path_label = QLabel("No video selected")
        self._source_path_label.setObjectName("source-path")
        self._source_path_label.setWordWrap(True)
        form.addRow("Video", self._source_path_label)
        self._source_dimensions_label = QLabel("—")
        self._source_dimensions_label.setObjectName("source-dimensions")
        form.addRow("Dimensions", self._source_dimensions_label)
        self._detector_combo = self._detector()
        self._device_combo = self._device()
        self._threshold_spin = self._threshold()
        form.addRow("Detector", self._detector_combo)
        form.addRow("Device", self._device_combo)
        form.addRow("Threshold", self._threshold_spin)
        self._max_frames_spin = self._max_frames()
        self._warmup_runs_spin = self._execution_count("warmup-runs")
        self._annotate_every_spin = self._execution_count("annotate-every")
        form.addRow("Max frames", self._max_frames_spin)
        form.addRow("Warm-up runs", self._warmup_runs_spin)
        form.addRow("Annotate every", self._annotate_every_spin)
        self._output_line = self._output()
        form.addRow("Output", self._output_line)
        self._config_path_line = self._readonly_line("config-path")
        self._run_cli_line = self._readonly_line("run-cli")
        form.addRow("Experiment config", self._config_path_line)
        form.addRow("CLI", self._run_cli_line)
        controls_layout.addLayout(form)
        controls_layout.addWidget(self._camera_controls())
        controls_layout.addWidget(self._region_controls())
        self._run_progress = QLabel("No active run")
        self._run_progress.setObjectName("run-progress")
        self._run_progress.setWordWrap(True)
        controls_layout.addWidget(self._run_progress)
        controls_layout.addLayout(self._actions())

        self.resultsWidget = ResultsWidget()
        self.resultsWidget.setVisible(False)
        controls_layout.addWidget(self.resultsWidget)
        controls_layout.addStretch()
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self._output_line.textChanged.connect(self._update_control_state)
        self._region_id.textChanged.connect(self._update_control_state)
        self._new_region_button.clicked.connect(self._begin_region_draw)
        self._delete_region_button.clicked.connect(self._delete_selected_region)
        for spin_box in self._region_spin_boxes:
            spin_box.valueChanged.connect(self._apply_numeric_region)
        self._source_view.scene().selectionChanged.connect(self._sync_selected_region)
        self._source_view.regionsChanged.connect(self._regions_changed)
        self._source_view.regionDrawn.connect(self._region_drawn)
        self._source_mode_combo.currentTextChanged.connect(self._source_mode_changed)
        self._camera_combo.currentIndexChanged.connect(self._camera_device_changed)
        self._start_preview_button.clicked.connect(self._start_camera_preview)
        self._record_button.clicked.connect(self._start_camera_recording)
        self._stop_recording_button.clicked.connect(self._stop_camera_recording)
        self._run_button.clicked.connect(self._start_run)
        self._cancel_button.clicked.connect(self._cancel_run)
        self.runController.progressChanged.connect(self._run_progress_changed)
        self.runController.runFinished.connect(self._run_finished)
        self.runController.runFailed.connect(self._run_failed)
        self.runController.processTerminated.connect(self._run_process_terminated)
        self._clear_region_values()
        self.statusBar().showMessage("Ready")
        if run_dir is not None:
            self._load_completed_run_or_report(run_dir)
        if self._capture_controller is not None:
            self._connect_capture_controller()
        self._update_control_state()

    def load_video(self, path: Path) -> None:
        """Decode and display one bounded preview frame from a local video."""

        if self.runController.is_active:
            raise RuntimeError("cannot replace source while a run is active")
        if self._camera_recording_in_progress():
            raise RuntimeError("cannot replace source while camera recording is active")
        if self._preview_active and self._capture_controller is not None:
            self._capture_controller.stop_preview()
        self._source_mode_combo.setCurrentText("Video file")
        self._load_video(Path(path), capture=None)

    def _load_video(self, path: Path, *, capture: CaptureResult | None) -> None:
        source_path = Path(path)
        frame = first_video_frame(source_path)
        self._source_view.set_rgb_frame(frame.image)
        height, width = frame.image.shape[:2]
        self._source_path = source_path
        self._source_width = int(width)
        self._source_height = int(height)
        self._capture_result = capture
        self._source_path_label.setText(str(source_path))
        self._source_dimensions_label.setText(f"{width} × {height} px")
        if capture is None:
            self._capture_requested_label.setText("Requested: —")
            self._capture_actual_label.setText("Actual: —")
        else:
            self._show_capture_metadata(capture)
        self._configure_region_controls()
        self._update_control_state()
        self.statusBar().showMessage(f"Loaded {source_path}")

    def _create_file_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._open_video_action = QAction("Open Video…", self)
        self._open_video_action.setObjectName("open-video-action")
        self._open_video_action.triggered.connect(self._choose_video)
        file_menu.addAction(self._open_video_action)

    def _choose_video(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)",
        )
        if not selected:
            return
        try:
            self.load_video(Path(selected))
        except (OSError, RuntimeError, ValueError) as error:
            self.statusBar().showMessage(str(error))

    def _camera_controls(self) -> QGroupBox:
        group = QGroupBox("Camera capture")
        group.setObjectName("camera-controls")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self._camera_combo = QComboBox()
        self._camera_combo.setObjectName("camera")
        self._capture_width_combo = self._capture_constraint_combo("capture-width")
        self._capture_height_combo = self._capture_constraint_combo("capture-height")
        self._capture_fps_combo = self._capture_constraint_combo("capture-fps")
        self._capture_strict = QCheckBox("Require exact supplied constraints")
        self._capture_strict.setObjectName("capture-strict")
        form.addRow("Camera", self._camera_combo)
        form.addRow("Width", self._capture_width_combo)
        form.addRow("Height", self._capture_height_combo)
        form.addRow("FPS", self._capture_fps_combo)
        form.addRow("Strict", self._capture_strict)
        self._capture_selected_label = QLabel("—")
        self._capture_selected_label.setObjectName("capture-selected-format")
        self._capture_selected_label.setWordWrap(True)
        self._capture_requested_label = QLabel("Requested: —")
        self._capture_requested_label.setObjectName("capture-requested-metadata")
        self._capture_requested_label.setWordWrap(True)
        self._capture_actual_label = QLabel("Actual: —")
        self._capture_actual_label.setObjectName("capture-actual-metadata")
        self._capture_actual_label.setWordWrap(True)
        form.addRow("Selected", self._capture_selected_label)
        form.addRow(self._capture_requested_label)
        form.addRow(self._capture_actual_label)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._start_preview_button = QPushButton("Start Preview")
        self._start_preview_button.setObjectName("start-preview-button")
        self._record_button = QPushButton("Record")
        self._record_button.setObjectName("record-button")
        self._stop_recording_button = QPushButton("Stop")
        self._stop_recording_button.setObjectName("stop-recording-button")
        buttons.addWidget(self._start_preview_button)
        buttons.addWidget(self._record_button)
        buttons.addWidget(self._stop_recording_button)
        layout.addLayout(buttons)
        group.setEnabled(False)
        self._camera_group = group
        return group

    @staticmethod
    def _capture_constraint_combo(object_name: str) -> QComboBox:
        combo = QComboBox()
        combo.setObjectName(object_name)
        combo.addItem("Auto", None)
        return combo

    def _connect_capture_controller(self) -> QtCaptureController:
        controller = self._capture_controller
        if controller is None:
            controller = QtCaptureController(self)
            self._capture_controller = controller
        if not self._capture_controller_connected:
            controller.devicesChanged.connect(self._populate_camera_devices)
            controller.previewStarted.connect(self._camera_preview_started)
            controller.previewStopped.connect(self._camera_preview_stopped)
            controller.recordingStarted.connect(self._camera_recording_started)
            controller.recordingFinished.connect(self._camera_recording_finished)
            controller.errorOccurred.connect(self._camera_error)
            self._capture_controller_connected = True
        return controller

    def _source_mode_changed(self, source_mode: str) -> None:
        camera_mode = source_mode == "Camera"
        if not camera_mode and self._camera_recording_in_progress():
            self._source_mode_combo.blockSignals(True)
            self._source_mode_combo.setCurrentText("Camera")
            self._source_mode_combo.blockSignals(False)
            self._camera_group.setEnabled(True)
            self.statusBar().showMessage(
                "Stop or discard the camera recording before replacing the source"
            )
            self._update_control_state()
            return
        self._camera_group.setEnabled(camera_mode)
        if camera_mode:
            self._connect_capture_controller()
            self._populate_camera_devices()
        elif self._capture_controller is not None:
            if self._preview_active:
                self._capture_controller.stop_preview()
        self._update_control_state()

    def _camera_recording_in_progress(self) -> bool:
        controller = self._capture_controller
        return (
            self._record_start_requested
            or self._recording_active
            or (controller is not None and controller.is_recording)
        )

    def _populate_camera_devices(self) -> None:
        controller = self._connect_capture_controller()
        previous_id = None
        previous = self._camera_combo.currentData()
        if isinstance(previous, CameraDeviceInfo):
            previous_id = previous.device_id
        devices = controller.devices()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        for device in devices:
            self._camera_combo.addItem(device.description, device)
        if previous_id is not None:
            for index in range(self._camera_combo.count()):
                candidate = self._camera_combo.itemData(index)
                if isinstance(candidate, CameraDeviceInfo) and candidate.device_id == previous_id:
                    self._camera_combo.setCurrentIndex(index)
                    break
        self._camera_combo.blockSignals(False)
        self._camera_device_changed()

    def _camera_device_changed(self) -> None:
        device = self._camera_combo.currentData()
        formats = device.formats if isinstance(device, CameraDeviceInfo) else ()
        self._populate_constraint_combo(
            self._capture_width_combo,
            sorted({camera_format.width for camera_format in formats}),
        )
        self._populate_constraint_combo(
            self._capture_height_combo,
            sorted({camera_format.height for camera_format in formats}),
        )
        self._populate_constraint_combo(
            self._capture_fps_combo,
            sorted(
                {
                    fps
                    for camera_format in formats
                    for fps in (camera_format.min_fps, camera_format.max_fps)
                }
            ),
        )
        self._update_control_state()

    @staticmethod
    def _populate_constraint_combo(combo: QComboBox, values: list[int] | list[float]) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto", None)
        for value in values:
            combo.addItem(f"{value:g}", value)
        combo.blockSignals(False)

    def _current_capture_request(self) -> CaptureRequest | None:
        device = self._camera_combo.currentData()
        if not isinstance(device, CameraDeviceInfo):
            return None
        width = self._capture_width_combo.currentData()
        height = self._capture_height_combo.currentData()
        fps = self._capture_fps_combo.currentData()
        return CaptureRequest(
            device.device_id,
            device.description,
            cast(int | None, width),
            cast(int | None, height),
            cast(float | None, fps),
            self._capture_strict.isChecked(),
        )

    def _start_camera_preview(self) -> None:
        controller = self._connect_capture_controller()
        request = self._current_capture_request()
        device = self._camera_combo.currentData()
        if request is None or not isinstance(device, CameraDeviceInfo):
            return
        try:
            selected = select_camera_format(device.formats, request)
            video_output = self._source_view.begin_video_preview(
                selected.width,
                selected.height,
            )
            controller.start_preview(request, video_output)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._source_view.end_video_preview()
            self._camera_error(str(error))

    def _camera_preview_started(self, selected: object) -> None:
        if not isinstance(selected, CameraFormatInfo):
            self._camera_error("camera returned an invalid selected format")
            return
        self._preview_active = True
        self._recording_active = False
        self._record_start_requested = False
        self._recording_stop_requested = False
        self._source_path = None
        self._capture_result = None
        self._source_width = selected.width
        self._source_height = selected.height
        self._source_path_label.setText("Live camera preview")
        self._source_dimensions_label.setText(f"{selected.width} × {selected.height} px")
        self._capture_selected_label.setText(
            f"{selected.width} × {selected.height} px · {selected.pixel_format} · "
            f"{selected.min_fps:g}–{selected.max_fps:g} FPS"
        )
        self._update_control_state()
        self.statusBar().showMessage("Camera preview started")

    def _camera_preview_stopped(self) -> None:
        self._preview_active = False
        self._recording_active = False
        self._record_start_requested = False
        self._recording_stop_requested = False
        if self._source_path is None:
            self._source_view.end_video_preview()
        self._update_control_state()

    def _start_camera_recording(self) -> None:
        controller = self._connect_capture_controller()
        self._record_start_requested = True
        self._update_control_state()
        try:
            controller.start_recording()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._record_start_requested = False
            self._camera_error(str(error))

    def _camera_recording_started(self) -> None:
        self._record_start_requested = False
        self._recording_active = True
        self._recording_stop_requested = False
        self._update_control_state()
        self.statusBar().showMessage("Camera recording started")

    def _stop_camera_recording(self) -> None:
        controller = self._capture_controller
        if controller is None:
            return
        self._recording_stop_requested = True
        self._update_control_state()
        try:
            controller.stop_recording()
        except RuntimeError as error:
            self._recording_stop_requested = False
            self._camera_error(str(error))

    def _camera_recording_finished(self, result: object) -> None:
        if not isinstance(result, CaptureResult):
            self._camera_error("camera returned an invalid capture result")
            return
        self._source_mode_combo.setCurrentText("Video file")
        try:
            self._load_video(result.path, capture=result)
        except (OSError, ValueError) as error:
            self._camera_error(str(error))
            return
        self.statusBar().showMessage(f"Captured {result.path}")

    def _show_capture_metadata(self, result: CaptureResult) -> None:
        request = result.request
        requested_width = "Auto" if request.requested_width is None else str(request.requested_width)
        requested_height = (
            "Auto" if request.requested_height is None else str(request.requested_height)
        )
        requested_fps = "Auto" if request.requested_fps is None else f"{request.requested_fps:g}"
        strictness = "strict" if request.strict else "normal"
        self._capture_requested_label.setText(
            f"Requested: width {requested_width} · height {requested_height} · "
            f"FPS {requested_fps} · {strictness}"
        )
        audio = "audio" if result.has_audio else "no audio"
        self._capture_actual_label.setText(
            f"Actual: {result.actual_width} × {result.actual_height} px · "
            f"{result.actual_fps:g} FPS · {result.container}/{result.codec} · {audio}"
        )

    def _camera_error(self, message: str) -> None:
        self._record_start_requested = False
        self._recording_active = False
        self._recording_stop_requested = False
        self._update_control_state()
        self.statusBar().showMessage(message)

    def _region_controls(self) -> QGroupBox:
        group = QGroupBox("Source regions")
        self._region_group = group
        region_layout = QVBoxLayout(group)
        form = QFormLayout()
        self._region_id = QLineEdit()
        self._region_id.setObjectName("region-id")
        self._region_id.setPlaceholderText("Region ID")
        form.addRow("ID", self._region_id)
        self._region_x = self._coordinate_spin_box("region-x")
        self._region_y = self._coordinate_spin_box("region-y")
        self._region_width = self._size_spin_box("region-width")
        self._region_height = self._size_spin_box("region-height")
        self._region_spin_boxes = (
            self._region_x,
            self._region_y,
            self._region_width,
            self._region_height,
        )
        form.addRow("x", self._region_x)
        form.addRow("y", self._region_y)
        form.addRow("width", self._region_width)
        form.addRow("height", self._region_height)
        region_layout.addLayout(form)
        buttons = QHBoxLayout()
        self._new_region_button = QPushButton("New Region")
        self._new_region_button.setObjectName("new-region-button")
        self._delete_region_button = QPushButton("Delete Region")
        self._delete_region_button.setObjectName("delete-region-button")
        buttons.addWidget(self._new_region_button)
        buttons.addWidget(self._delete_region_button)
        region_layout.addLayout(buttons)
        return group

    @staticmethod
    def _coordinate_spin_box(object_name: str) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setObjectName(object_name)
        spin_box.setRange(0, 0)
        spin_box.setEnabled(False)
        return spin_box

    @staticmethod
    def _size_spin_box(object_name: str) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setObjectName(object_name)
        spin_box.setRange(1, 1)
        spin_box.setEnabled(False)
        return spin_box

    def _configure_region_controls(self) -> None:
        self._updating_region_controls = True
        self._region_x.setRange(0, self._source_width - 1)
        self._region_y.setRange(0, self._source_height - 1)
        self._region_width.setRange(1, self._source_width)
        self._region_height.setRange(1, self._source_height)
        self._updating_region_controls = False
        self._clear_region_values()

    def _begin_region_draw(self) -> None:
        region_id = self._region_id.text().strip()
        if not region_id:
            return
        try:
            self._source_view.begin_region_draw(region_id)
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        self._update_control_state()
        self.statusBar().showMessage(f"Draw region {region_id} on the preview")

    def _region_drawn(self, region: object) -> None:
        if isinstance(region, Region):
            self.statusBar().showMessage(f"Added region {region.region_id}")
        else:
            self.statusBar().showMessage("Region draw had zero area")
        self._update_control_state()

    def _delete_selected_region(self) -> None:
        selected = self._source_view.selected_region()
        self._source_view.delete_selected_region()
        if selected is not None:
            self.statusBar().showMessage(f"Deleted region {selected.region_id}")
        self._update_control_state()

    def _apply_numeric_region(self) -> None:
        if self._updating_region_controls:
            return
        selected = self._source_view.selected_region()
        if selected is None:
            return
        updated = self._source_view.update_region(
            selected.region_id,
            x=self._region_x.value(),
            y=self._region_y.value(),
            width=self._region_width.value(),
            height=self._region_height.value(),
        )
        self._set_region_values(updated)

    def _regions_changed(self, _regions: object) -> None:
        self._sync_selected_region()
        self._update_control_state()

    def _sync_selected_region(self) -> None:
        selected = self._source_view.selected_region()
        if selected is not None:
            self._region_id.setText(selected.region_id)
            self._set_region_values(selected)
        else:
            self._clear_region_values()
        self._update_control_state()

    def _set_region_values(self, region: Region) -> None:
        self._updating_region_controls = True
        self._region_x.setValue(region.x)
        self._region_y.setValue(region.y)
        self._region_width.setValue(region.width)
        self._region_height.setValue(region.height)
        self._updating_region_controls = False

    def _clear_region_values(self) -> None:
        self._updating_region_controls = True
        for spin_box in self._region_spin_boxes:
            spin_box.clear()
            spin_box.setEnabled(False)
        self._updating_region_controls = False

    def _update_control_state(self) -> None:
        run_active = self.runController.is_active
        has_source = self._source_path is not None
        camera_mode = self._source_mode_combo.currentText() == "Camera"
        has_camera = isinstance(self._camera_combo.currentData(), CameraDeviceInfo)
        selected = self._source_view.selected_region()
        self._open_video_action.setEnabled(not run_active)
        self._source_mode_combo.setEnabled(not run_active)
        self._detector_combo.setEnabled(not run_active)
        self._device_combo.setEnabled(not run_active)
        self._threshold_spin.setEnabled(not run_active)
        self._max_frames_spin.setEnabled(not run_active)
        self._warmup_runs_spin.setEnabled(not run_active)
        self._annotate_every_spin.setEnabled(not run_active)
        self._output_line.setEnabled(not run_active)
        self._source_view.setEnabled(not run_active)
        self._region_group.setEnabled(not run_active)
        self._camera_group.setEnabled(camera_mode and not run_active)
        for spin_box in self._region_spin_boxes:
            spin_box.setEnabled(selected is not None and not run_active)
        self._new_region_button.setEnabled(
            not run_active
            and has_source
            and bool(self._region_id.text().strip())
            and not self._source_view.is_drawing_region()
        )
        self._delete_region_button.setEnabled(
            not run_active and has_source and selected is not None
        )
        self._start_preview_button.setEnabled(
            not run_active
            and camera_mode
            and has_camera
            and not self._preview_active
            and not self._record_start_requested
            and not self._recording_active
        )
        self._record_button.setEnabled(
            not run_active
            and camera_mode
            and self._preview_active
            and not self._record_start_requested
            and not self._recording_active
        )
        self._stop_recording_button.setEnabled(
            not run_active
            and camera_mode
            and self._recording_active
            and not self._recording_stop_requested
        )
        self._run_button.setEnabled(not run_active and self._current_run_config() is not None)
        self._cancel_button.setEnabled(run_active)

    def _current_run_config(self) -> RunConfig | None:
        try:
            return self.resolved_config()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    def resolved_config(self) -> RunConfig:
        """Resolve and validate the current native controls into one run config."""

        if self._source_path is None or self._camera_recording_in_progress():
            raise ValueError("finalized video source is required")
        output_text = self._output_line.text().strip()
        if not output_text:
            raise ValueError("new empty output directory is required")
        max_frames_value = self._max_frames_spin.value()
        config = RunConfig(
            input_path=self._source_path,
            output_dir=Path(output_text),
            regions=self._source_view.regions(),
            threshold=self._threshold_spin.value(),
            max_frames=None if max_frames_value < 0 else max_frames_value,
            warmup_runs=self._warmup_runs_spin.value(),
            annotate_every=self._annotate_every_spin.value(),
            detector_id=str(self._detector_combo.currentData()),
            device=self._device_combo.currentText().lower(),
            capture=self._capture_result,
        )
        validate_output_directory(config.output_dir)
        config_path = (
            config.output_dir.parent / f"{config.output_dir.name}.experiment.json"
        ).resolve()
        if config_path.exists():
            raise FileExistsError(f"experiment config already exists: {config_path}")
        return config

    def _start_run(self) -> None:
        try:
            config = self.resolved_config()
            self._run_failure_shown = False
            self.runController.start(config)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._run_failed(str(error))
            self._update_control_state()
            return
        config_path = self.runController.config_path
        if config_path is not None:
            self._config_path_line.setText(str(config_path))
            self._run_cli_line.setText(shlex.join(render_run_cli(config_path)))
        self._update_control_state()
        if self.runController.is_active:
            self.statusBar().showMessage(f"Run started: {config.output_dir}")

    def _cancel_run(self) -> None:
        try:
            self.runController.cancel()
        except OSError as error:
            self._run_failed(str(error))
            return
        self.statusBar().showMessage("Cancellation requested")

    def _run_progress_changed(self, value: object) -> None:
        if not isinstance(value, ProgressEvent):
            return
        self._run_progress.setText(
            f"{value.phase}: {value.frames_processed} frames, "
            f"{value.inference_count} inferences, {value.elapsed_ms:.1f} ms"
        )

    def _run_finished(self, run_dir: Path, payload: dict[str, object]) -> None:
        path = self._load_completed_run_or_report(run_dir)
        if path is None:
            return
        phase = payload.get("phase")
        self.statusBar().showMessage(f"Run {phase}: {path}")
        self._update_control_state()

    def _load_completed_run_or_report(self, run_dir: Path) -> Path | None:
        try:
            return self._load_completed_run(run_dir)
        except (OSError, TypeError, ValueError) as error:
            self._run_failed(f"completed run could not be loaded: {error}")
            return None

    def _load_completed_run(self, run_dir: Path) -> Path:
        path = Path(run_dir).resolve()
        self.resultsWidget.load_run(path)
        self.resultsWidget.setVisible(True)
        self.statusBar().showMessage(f"Loaded completed run: {path}")
        return path

    def _run_failed(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if not self._run_failure_shown:
            self._run_failure_shown = True
            QMessageBox.critical(self, "Run failed", message)
        self._update_control_state()

    def _run_process_terminated(self) -> None:
        self._update_control_state()
        if not self._exit_after_run:
            return
        self._kill_timer.stop()
        self._exit_after_run = False
        self._allow_close = True
        self.close()

    def _kill_active_run(self) -> None:
        self.runController.kill()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Make recording discard and inference cancellation explicit."""

        if self._allow_close:
            event.accept()
            return
        if self._camera_recording_in_progress():
            decision = self._resolve_close_decision("recording")
            if decision != "Stop and Discard":
                event.ignore()
                return
            controller = self._capture_controller
            if controller is not None:
                try:
                    controller.discard()
                except (OSError, RuntimeError) as error:
                    self.statusBar().showMessage(str(error))
                    event.ignore()
                    return
            event.accept()
            return
        if self.runController.is_active:
            if self._exit_after_run:
                self._kill_timer.start(5_000)
                event.ignore()
                return
            decision = self._resolve_close_decision("inference")
            if decision != "Cancel Run and Exit":
                event.ignore()
                return
            try:
                self.runController.cancel()
            except OSError as error:
                self.statusBar().showMessage(str(error))
                event.ignore()
                return
            self._exit_after_run = True
            self._kill_timer.start(5_000)
            event.ignore()
            return
        event.accept()

    def _resolve_close_decision(self, activity: str) -> str:
        if self._close_decision is not None:
            return self._close_decision(activity)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Active work")
        if activity == "recording":
            dialog.setText("A camera recording is still active.")
            action_text = "Stop and Discard"
        else:
            dialog.setText("An inference run is still active.")
            action_text = "Cancel Run and Exit"
        keep = dialog.addButton("Keep Window Open", QMessageBox.ButtonRole.RejectRole)
        action = dialog.addButton(action_text, QMessageBox.ButtonRole.DestructiveRole)
        dialog.exec()
        return action_text if dialog.clickedButton() is action else keep.text()

    @staticmethod
    def _source_mode() -> QComboBox:
        source_mode = QComboBox()
        source_mode.setObjectName("source-mode")
        source_mode.addItems(["Video file", "Camera"])
        return source_mode

    @staticmethod
    def _detector() -> QComboBox:
        detector = QComboBox()
        detector.setObjectName("detector")
        for descriptor in detector_descriptors():
            detector.addItem(descriptor.display_name, descriptor.detector_id)
        return detector

    @staticmethod
    def _device() -> QComboBox:
        device = QComboBox()
        device.setObjectName("device")
        device.addItems(["Auto", "CPU", "CUDA"])
        return device

    @staticmethod
    def _threshold() -> QDoubleSpinBox:
        threshold = QDoubleSpinBox()
        threshold.setObjectName("threshold")
        threshold.setRange(0.0, 1.0)
        threshold.setSingleStep(0.05)
        threshold.setValue(0.3)
        return threshold

    @staticmethod
    def _max_frames() -> QSpinBox:
        max_frames = QSpinBox()
        max_frames.setObjectName("max-frames")
        max_frames.setRange(-1, 2_147_483_647)
        max_frames.setSpecialValueText("All")
        max_frames.setValue(-1)
        return max_frames

    @staticmethod
    def _execution_count(object_name: str) -> QSpinBox:
        count = QSpinBox()
        count.setObjectName(object_name)
        count.setRange(0, 2_147_483_647)
        return count

    @staticmethod
    def _output() -> QLineEdit:
        output = QLineEdit()
        output.setObjectName("output")
        output.setPlaceholderText("Output directory")
        return output

    @staticmethod
    def _readonly_line(object_name: str) -> QLineEdit:
        line = QLineEdit()
        line.setObjectName(object_name)
        line.setReadOnly(True)
        return line

    def _actions(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.setObjectName("run-button")
        self._run_button.setEnabled(False)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("cancel-button")
        self._cancel_button.setEnabled(False)
        actions.addWidget(self._run_button)
        actions.addWidget(self._cancel_button)
        return actions
