"""Native file preview and source-region controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from edge_perception.contracts import Region
from edge_perception.detectors.registry import detector_descriptors
from edge_perception.gui.region_view import RegionView
from edge_perception.video import first_video_frame


class MainWindow(QMainWindow):
    """Preview a local video and edit named source-pixel regions."""

    def __init__(self, run_dir: Path | None = None) -> None:
        super().__init__()
        self.setObjectName("edge-perception-main-window")
        self.setWindowTitle("Edge Perception")
        self.resize(1100, 720)
        self._source_path: Path | None = None
        self._source_width = 0
        self._source_height = 0
        self._updating_region_controls = False

        self._create_file_menu()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_view = RegionView()
        splitter.addWidget(self._source_view)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.addRow("Source mode", self._source_mode())
        self._source_path_label = QLabel("No video selected")
        self._source_path_label.setObjectName("source-path")
        self._source_path_label.setWordWrap(True)
        form.addRow("Video", self._source_path_label)
        self._source_dimensions_label = QLabel("—")
        self._source_dimensions_label.setObjectName("source-dimensions")
        form.addRow("Dimensions", self._source_dimensions_label)
        form.addRow("Detector", self._detector())
        form.addRow("Device", self._device())
        form.addRow("Threshold", self._threshold())
        self._output_line = self._output(run_dir)
        form.addRow("Output", self._output_line)
        controls_layout.addLayout(form)
        controls_layout.addWidget(self._region_controls())
        controls_layout.addLayout(self._actions())

        completed_run = QGroupBox("Completed run")
        completed_run.setObjectName("completed-run")
        completed_run.setVisible(False)
        completed_layout = QVBoxLayout(completed_run)
        completed_layout.addWidget(QLabel("Results will appear here in a later task."))
        controls_layout.addWidget(completed_run)
        controls_layout.addStretch()
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self._output_line.textChanged.connect(self._update_control_state)
        self._region_id.textChanged.connect(self._update_control_state)
        self._new_region_button.clicked.connect(self._add_region_from_controls)
        self._delete_region_button.clicked.connect(self._delete_selected_region)
        for spin_box in self._region_spin_boxes:
            spin_box.valueChanged.connect(self._apply_numeric_region)
        self._source_view.scene().selectionChanged.connect(self._sync_selected_region)
        self._source_view.regionsChanged.connect(self._regions_changed)
        self._update_control_state()
        self.statusBar().showMessage("Ready")

    def load_video(self, path: Path) -> None:
        """Decode and display one bounded preview frame from a local video."""

        source_path = Path(path)
        frame = first_video_frame(source_path)
        self._source_view.set_rgb_frame(frame.image)
        height, width = frame.image.shape[:2]
        self._source_path = source_path
        self._source_width = int(width)
        self._source_height = int(height)
        self._source_path_label.setText(str(source_path))
        self._source_dimensions_label.setText(f"{width} × {height} px")
        self._configure_region_controls()
        self._update_control_state()
        self.statusBar().showMessage(f"Loaded {source_path}")

    def _create_file_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_video = QAction("Open Video…", self)
        open_video.setObjectName("open-video-action")
        open_video.triggered.connect(self._choose_video)
        file_menu.addAction(open_video)

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
        except (OSError, ValueError) as error:
            self.statusBar().showMessage(str(error))

    def _region_controls(self) -> QGroupBox:
        group = QGroupBox("Source regions")
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
        self._region_x.setValue(0)
        self._region_y.setValue(0)
        self._region_width.setValue(self._source_width)
        self._region_height.setValue(self._source_height)
        self._updating_region_controls = False

    def _add_region_from_controls(self) -> None:
        region_id = self._region_id.text().strip()
        if not region_id:
            return
        region = Region(
            region_id,
            self._region_x.value(),
            self._region_y.value(),
            self._region_width.value(),
            self._region_height.value(),
        )
        try:
            self._source_view.add_region(region)
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        self._source_view.select_region(region_id)
        self.statusBar().showMessage(f"Added region {region_id}")

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
        self._update_control_state()

    def _set_region_values(self, region: Region) -> None:
        self._updating_region_controls = True
        self._region_x.setValue(region.x)
        self._region_y.setValue(region.y)
        self._region_width.setValue(region.width)
        self._region_height.setValue(region.height)
        self._updating_region_controls = False

    def _update_control_state(self) -> None:
        has_source = self._source_path is not None
        for spin_box in self._region_spin_boxes:
            spin_box.setEnabled(has_source)
        self._new_region_button.setEnabled(has_source and bool(self._region_id.text().strip()))
        self._delete_region_button.setEnabled(
            has_source and self._source_view.selected_region() is not None
        )
        self._run_button.setEnabled(has_source and bool(self._output_line.text().strip()))

    @staticmethod
    def _source_mode() -> QComboBox:
        source_mode = QComboBox()
        source_mode.setObjectName("source-mode")
        source_mode.addItems(["Video file", "Camera (later)"])
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
    def _output(run_dir: Path | None) -> QLineEdit:
        output = QLineEdit()
        output.setObjectName("output")
        output.setPlaceholderText("Output directory")
        if run_dir is not None:
            output.setText(str(run_dir))
        return output

    def _actions(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.setObjectName("run-button")
        self._run_button.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel-button")
        cancel.setEnabled(False)
        actions.addWidget(self._run_button)
        actions.addWidget(cancel)
        return actions
