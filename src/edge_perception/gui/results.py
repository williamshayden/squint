"""Native read-only viewer for canonical run results."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edge_perception.run_view import RunViewData, load_run_view

_STATUS_LABELS = {
    "complete": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _optional_metric(value: float | None, suffix: str) -> str:
    return "N/A" if value is None else f"{value:.3f} {suffix}"


def _optional_bytes(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,} bytes"


class ResultsWidget(QWidget):
    """Display immutable summary values, regions, and annotated PNGs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("results-widget")
        self._view_data: RunViewData | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        details = QWidget(splitter)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(6, 0, 0, 0)

        overview = QGroupBox("Overview", details)
        overview.setObjectName("results-overview")
        overview_form = QFormLayout(overview)
        self.runPathLabel = self._label("run-path")
        self.statusLabel = self._label("result-status")
        self.errorLabel = self._label("result-error")
        overview_form.addRow("Run directory", self.runPathLabel)
        overview_form.addRow("Status", self.statusLabel)
        overview_form.addRow("Error", self.errorLabel)
        details_layout.addWidget(overview)

        metrics = QGroupBox("Metrics", details)
        metrics.setObjectName("results-metrics")
        metrics_form = QFormLayout(metrics)
        self.framesLabel = self._label("result-frames")
        self.inferencesLabel = self._label("result-inferences")
        self.p50Label = self._label("result-p50")
        self.p95Label = self._label("result-p95")
        self.p99Label = self._label("result-p99")
        self.peakRssLabel = self._label("result-peak-rss")
        self.peakVramLabel = self._label("result-peak-vram")
        for title, label in (
            ("Frames processed", self.framesLabel),
            ("Inference count", self.inferencesLabel),
            ("Frame latency p50", self.p50Label),
            ("Frame latency p95", self.p95Label),
            ("Frame latency p99", self.p99Label),
            ("Peak RSS", self.peakRssLabel),
            ("Peak VRAM", self.peakVramLabel),
        ):
            metrics_form.addRow(title, label)
        details_layout.addWidget(metrics)

        run_configuration = QGroupBox("Run configuration", details)
        run_configuration.setObjectName("results-run-configuration")
        run_configuration_layout = QVBoxLayout(run_configuration)
        run_configuration_form = QFormLayout()
        self.detectorLabel = self._label("result-detector")
        self.revisionLabel = self._label("result-revision")
        self.deviceLabel = self._label("result-device")
        self.thresholdLabel = self._label("result-threshold")
        for title, label in (
            ("Detector", self.detectorLabel),
            ("Revision", self.revisionLabel),
            ("Compute device", self.deviceLabel),
            ("Confidence threshold", self.thresholdLabel),
        ):
            run_configuration_form.addRow(title, label)
        run_configuration_layout.addLayout(run_configuration_form)
        self.regionsTable = QTableWidget(0, 5, run_configuration)
        self.regionsTable.setObjectName("regions-table")
        self.regionsTable.setHorizontalHeaderLabels(
            ["Region", "X", "Y", "Width", "Height"]
        )
        self.regionsTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.regionsTable.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.regionsTable.setSortingEnabled(False)
        self.regionsTable.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        run_configuration_layout.addWidget(self.regionsTable)
        details_layout.addWidget(run_configuration)

        source_provenance = QGroupBox("Source provenance", details)
        source_provenance.setObjectName("results-source-provenance")
        source_provenance_form = QFormLayout(source_provenance)
        self.sourceLabel = self._label("result-source")
        self.sourceDimensionsLabel = self._label("result-source-dimensions")
        self.sourceFpsLabel = self._label("result-source-fps")
        self.capturePathLabel = self._label("result-capture-path")
        self.captureSha256Label = self._label("result-capture-sha256")
        self.captureRequestLabel = self._label("result-capture-request")
        self.captureRecordedFormatLabel = self._label(
            "result-capture-recorded-format"
        )
        for title, label in (
            ("Source path", self.sourceLabel),
            ("Frame size", self.sourceDimensionsLabel),
            ("Frame rate", self.sourceFpsLabel),
            ("Capture path", self.capturePathLabel),
            ("Capture SHA-256", self.captureSha256Label),
            ("Capture request", self.captureRequestLabel),
            ("Recorded format", self.captureRecordedFormatLabel),
        ):
            source_provenance_form.addRow(title, label)
        details_layout.addWidget(source_provenance)

        annotations = QGroupBox("Annotations", details)
        annotations.setObjectName("results-annotations")
        annotations_layout = QVBoxLayout(annotations)
        annotations_form = QFormLayout()
        self.annotationCountLabel = self._label("result-annotation-count")
        annotations_form.addRow("Count", self.annotationCountLabel)
        annotations_layout.addLayout(annotations_form)
        annotation_splitter = QSplitter(Qt.Orientation.Horizontal, annotations)
        self.annotationList = QListWidget(annotation_splitter)
        self.annotationList.setObjectName("annotation-list")
        self.annotationList.setMinimumWidth(130)
        self.imageView = QGraphicsView(annotation_splitter)
        self.imageView.setObjectName("annotation-view")
        self.imageScene = QGraphicsScene(self.imageView)
        self.imageView.setScene(self.imageScene)
        self.imageView.setMinimumWidth(260)
        annotations_layout.addWidget(annotation_splitter)

        self.imageErrorLabel = self._label("annotation-error")
        self.imageErrorLabel.setWordWrap(True)
        annotations_layout.addWidget(self.imageErrorLabel)
        details_layout.addWidget(annotations)
        details_layout.addStretch()

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.annotationList.currentRowChanged.connect(self._annotation_selected)

    @staticmethod
    def _label(object_name: str) -> QLabel:
        label = QLabel("N/A")
        label.setObjectName(object_name)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def load_run(self, run_dir: Path) -> RunViewData:
        """Load one run through the shared canonical projection and render it."""

        view = load_run_view(Path(run_dir))
        self._view_data = view
        self._render(view)
        return view

    def _render(self, view: RunViewData) -> None:
        self.runPathLabel.setText(str(view.run_dir))
        self.statusLabel.setText(_STATUS_LABELS[view.status])
        self.errorLabel.setText(view.error if view.error is not None else "N/A")
        self.framesLabel.setText(str(view.frames_processed))
        self.inferencesLabel.setText(str(view.inference_count))
        self.p50Label.setText(_optional_metric(view.frame_p50_ms, "ms"))
        self.p95Label.setText(_optional_metric(view.frame_p95_ms, "ms"))
        self.p99Label.setText(_optional_metric(view.frame_p99_ms, "ms"))
        self.peakRssLabel.setText(_optional_bytes(view.peak_rss_bytes))
        self.peakVramLabel.setText(_optional_bytes(view.peak_vram_bytes))
        self.detectorLabel.setText(view.detector_model_id)
        self.revisionLabel.setText(view.detector_revision)
        self.deviceLabel.setText(view.device)
        self.thresholdLabel.setText(f"{view.threshold:.3f}")
        self.sourceLabel.setText(str(view.source_path))
        self.sourceDimensionsLabel.setText(f"{view.source_width} × {view.source_height} px")
        self.sourceFpsLabel.setText(
            "N/A" if view.capture is None else f"{view.capture.actual_fps:.3f} FPS"
        )
        self.annotationCountLabel.setText(str(view.annotated_frame_count))
        capture = view.capture
        if capture is None:
            self.capturePathLabel.setText("N/A")
            self.captureSha256Label.setText("N/A")
            self.captureRequestLabel.setText("N/A")
            self.captureRecordedFormatLabel.setText("N/A")
        else:
            request = capture.request
            requested_width = (
                "Auto" if request.requested_width is None else str(request.requested_width)
            )
            requested_height = (
                "Auto" if request.requested_height is None else str(request.requested_height)
            )
            requested_fps = (
                "Auto" if request.requested_fps is None else f"{request.requested_fps:g}"
            )
            strictness = "strict" if request.strict else "normal"
            audio = "audio" if capture.has_audio else "no audio"
            self.capturePathLabel.setText(str(capture.path))
            self.captureSha256Label.setText(capture.sha256)
            self.captureRequestLabel.setText(
                f"{request.device_description} ({request.device_id}) · "
                f"width {requested_width} · height {requested_height} · "
                f"{requested_fps} FPS · {strictness}"
            )
            self.captureRecordedFormatLabel.setText(
                f"{capture.actual_width} × {capture.actual_height} px · "
                f"{capture.actual_fps:.3f} FPS · {capture.container}/{capture.codec} · "
                f"{audio}"
            )

        self.regionsTable.setRowCount(0)
        for row, region in enumerate(view.regions):
            self.regionsTable.insertRow(row)
            values = (region.region_id, region.x, region.y, region.width, region.height)
            for column, value in enumerate(values):
                self.regionsTable.setItem(row, column, QTableWidgetItem(str(value)))

        blocker = QSignalBlocker(self.annotationList)
        self.annotationList.clear()
        for path in view.annotation_paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.annotationList.addItem(item)
        del blocker
        self.imageScene.clear()
        self.imageErrorLabel.setText("")
        if view.annotation_paths:
            self.annotationList.setCurrentRow(0)
        else:
            self.annotationList.setCurrentRow(-1)

    def _annotation_selected(self, row: int) -> None:
        if row < 0:
            self.imageScene.clear()
            self.imageErrorLabel.setText("")
            return
        item = self.annotationList.item(row)
        if item is None:
            return
        stored_path = item.data(Qt.ItemDataRole.UserRole)
        path = stored_path if isinstance(stored_path, Path) else Path(str(stored_path))
        pixmap = QPixmap(str(path))
        self.imageScene.clear()
        if pixmap.isNull():
            self.imageErrorLabel.setText(f"Unable to load annotation: {path}")
            return
        self.imageErrorLabel.setText("")
        pixmap_item = self.imageScene.addPixmap(pixmap)
        self.imageScene.setSceneRect(pixmap_item.boundingRect())
        self.imageView.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
