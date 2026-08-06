"""Small native shell for future Edge Perception research controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from edge_perception.detectors.registry import detector_descriptors


class MainWindow(QMainWindow):
    """Present the non-functional native controls that future tasks will connect."""

    def __init__(self, run_dir: Path | None = None) -> None:
        super().__init__()
        self.setObjectName("edge-perception-main-window")
        self.setWindowTitle("Edge Perception")
        self.resize(1100, 720)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        source_view = QGraphicsView()
        source_view.setObjectName("source-view")
        source_view.setScene(QGraphicsScene(source_view))
        splitter.addWidget(source_view)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        form = QFormLayout()
        form.addRow("Source mode", self._source_mode())
        form.addRow("Detector", self._detector())
        form.addRow("Device", self._device())
        form.addRow("Threshold", self._threshold())
        form.addRow("Output", self._output(run_dir))
        controls_layout.addLayout(form)
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

        self.statusBar().showMessage("Ready")

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

    @staticmethod
    def _actions() -> QHBoxLayout:
        actions = QHBoxLayout()
        run = QPushButton("Run")
        run.setObjectName("run-button")
        run.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel-button")
        cancel.setEnabled(False)
        actions.addWidget(run)
        actions.addWidget(cancel)
        return actions
