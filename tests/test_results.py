from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QAbstractItemView, QGraphicsPixmapItem
from pytestqt.qtbot import QtBot

from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.contracts import Region
from edge_perception.gui.results import ResultsWidget
from edge_perception.run_view import load_run_view


def _latency(count: int, p50: float | None, p95: float | None, p99: float | None) -> dict[str, object]:
    return {"count": count, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


def _capture(path: Path) -> CaptureResult:
    return CaptureResult(
        request=CaptureRequest("camera-1", "Test camera", 640, 480, 30.0, True),
        selected_width=640,
        selected_height=480,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=640,
        actual_height=480,
        actual_fps=29.97,
        container="mp4",
        codec="h264",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=1234,
        path=path,
        sha256="b" * 64,
    )


def write_completed_run_fixture(
    tmp_path: Path,
    *,
    name: str = "run",
    annotation_names: tuple[str, ...] = ("000002.png", "000000.png"),
    capture: CaptureResult | None = None,
) -> Path:
    run_dir = tmp_path / name
    annotated = run_dir / "annotated"
    annotated.mkdir(parents=True)
    for index, filename in enumerate(annotation_names):
        Image.new("RGB", (3, 2), color=(index * 40, 20, 10)).save(annotated / filename)
    run_id = f"{name}-run-id"
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "configuration": {
            "input_path": str((tmp_path / "historical-source.mp4").resolve()),
            "output_dir": str(run_dir.resolve()),
            "regions": [
                {"region_id": "left", "x": 0, "y": 0, "width": 320, "height": 480},
                {"region_id": "top-right", "x": 320, "y": 0, "width": 320, "height": 240},
            ],
            "execution_regions": [
                {"region_id": "full-frame", "x": 0, "y": 0, "width": 640, "height": 480},
                {"region_id": "left", "x": 0, "y": 0, "width": 320, "height": 480},
                {"region_id": "top-right", "x": 320, "y": 0, "width": 320, "height": 240},
            ],
            "threshold": 0.35,
            "max_frames": None,
            "warmup_runs": 1,
            "annotate_every": 2,
            "detector_id": "fake",
            "device": "cuda",
            "batch_size": 1,
        },
        "source_video": {
            "path": str((tmp_path / "historical-source.mp4").resolve()),
            "sha256": "a" * 64,
            "frame_width": 640,
            "frame_height": 480,
            "capture": None if capture is None else capture.to_dict(),
        },
        "host": {},
        "detector": {
            "adapter": "tests.fake",
            "model_id": "tests/fake-detector",
            "revision": "test-revision",
            "weights_sha256": "c" * 64,
            "backend": "fake",
            "backend_version": "1.0",
            "device": "cuda:1",
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
        "inference_count": 9,
        "annotated_frame_count": len(annotation_names),
        "latency_ms": {
            "full_frame": _latency(3, 4.0, 5.0, 6.0),
            "crop": _latency(6, 2.0, 3.0, 4.0),
            "complete_frame": _latency(3, 10.0, 12.0, 13.0),
        },
        "stage_latency_ms": {},
        "hardware_peaks": {
            "process_rss_bytes": 4096,
            "system_memory_used_bytes": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_bytes": 999999,
            "gpu_power_watts": None,
            "gpu_temperature_c": None,
        },
        "detector_peak_device_memory_bytes": 8192,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def snapshot_tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_load_run_view_uses_only_canonical_artifacts(tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path)
    before = snapshot_tree(run_dir)

    view = load_run_view(run_dir)

    assert view.status == "complete"
    assert view.frames_processed == 3
    assert view.inference_count == 9
    assert view.annotated_frame_count == 2
    assert view.detector_model_id == "tests/fake-detector"
    assert view.detector_revision == "test-revision"
    assert view.device == "cuda:1"
    assert view.threshold == pytest.approx(0.35)
    assert view.frame_p50_ms == pytest.approx(10.0)
    assert view.frame_p95_ms == pytest.approx(12.0)
    assert view.frame_p99_ms == pytest.approx(13.0)
    assert view.peak_rss_bytes == 4096
    assert view.peak_vram_bytes == 8192
    assert view.source_path == (tmp_path / "historical-source.mp4").resolve()
    assert (view.source_width, view.source_height) == (640, 480)
    assert view.capture is None
    assert view.regions == (
        Region("left", 0, 0, 320, 480),
        Region("top-right", 320, 0, 320, 240),
    )
    assert [path.name for path in view.annotation_paths] == ["000000.png", "000002.png"]
    assert view.error is None
    assert snapshot_tree(run_dir) == before


@pytest.mark.parametrize(
    ("status", "error"),
    [("failed", "RuntimeError: detector failed"), ("cancelled", None)],
)
def test_load_run_view_accepts_failed_and_cancelled_status(
    status: str,
    error: str | None,
    tmp_path: Path,
) -> None:
    run_dir = write_completed_run_fixture(tmp_path, annotation_names=())
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary["status"] = status
    summary["frames_processed"] = 0
    summary["inference_count"] = 0
    summary["annotated_frame_count"] = 0
    summary["latency_ms"] = {
        "full_frame": _latency(0, None, None, None),
        "crop": _latency(0, None, None, None),
        "complete_frame": _latency(0, None, None, None),
    }
    if error is not None:
        summary["error"] = error
    _write_json(summary_path, summary)

    view = load_run_view(run_dir)

    assert view.status == status
    assert view.error == error


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-manifest", "manifest.json"),
        ("malformed-summary", "summary.json"),
        ("wrong-manifest-schema", "manifest.json.schema_version"),
        ("wrong-summary-schema", "summary.json.schema_version"),
        ("mismatched-run-id", "run_id"),
        ("bool-frame-count", "summary.json.frames_processed"),
        ("negative-width", "manifest.json.source_video.frame_width"),
        ("missing-detector-device", "manifest.json.detector.device"),
        ("duplicate-region", "manifest.json.configuration.regions"),
        ("out-of-frame-region", "manifest.json.configuration.regions"),
        ("error-on-complete", "summary.json.error"),
    ],
)
def test_load_run_view_rejects_missing_malformed_or_wrong_schema_artifacts(
    mutation: str,
    message: str,
    tmp_path: Path,
) -> None:
    run_dir = write_completed_run_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    if mutation == "missing-manifest":
        manifest_path.unlink()
    elif mutation == "malformed-summary":
        summary_path.write_text("{", encoding="utf-8")
    elif mutation == "wrong-manifest-schema":
        manifest["schema_version"] = "9.9.9"
        _write_json(manifest_path, manifest)
    elif mutation == "wrong-summary-schema":
        summary["schema_version"] = "9.9.9"
        _write_json(summary_path, summary)
    elif mutation == "mismatched-run-id":
        summary["run_id"] = "different"
        _write_json(summary_path, summary)
    elif mutation == "bool-frame-count":
        summary["frames_processed"] = True
        _write_json(summary_path, summary)
    elif mutation == "negative-width":
        source = manifest["source_video"]
        assert isinstance(source, dict)
        source["frame_width"] = -1
        _write_json(manifest_path, manifest)
    elif mutation == "missing-detector-device":
        detector = manifest["detector"]
        assert isinstance(detector, dict)
        del detector["device"]
        _write_json(manifest_path, manifest)
    elif mutation == "duplicate-region":
        configuration = manifest["configuration"]
        assert isinstance(configuration, dict)
        regions = configuration["regions"]
        assert isinstance(regions, list)
        regions.append(dict(regions[0]))
        _write_json(manifest_path, manifest)
    elif mutation == "out-of-frame-region":
        configuration = manifest["configuration"]
        assert isinstance(configuration, dict)
        regions = configuration["regions"]
        assert isinstance(regions, list)
        region = regions[0]
        assert isinstance(region, dict)
        region["width"] = 641
        _write_json(manifest_path, manifest)
    else:
        summary["error"] = "not canonical"
        _write_json(summary_path, summary)

    with pytest.raises(ValueError, match=message):
        load_run_view(run_dir)


def test_load_run_view_allows_no_annotations(tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path, annotation_names=())

    view = load_run_view(run_dir)

    assert view.annotation_paths == ()
    assert view.annotated_frame_count == 0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_load_run_view_rejects_non_finite_metrics(constant: str, tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path)
    summary_path = run_dir / "summary.json"
    serialized = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(serialized.replace('"p50_ms": 10.0', f'"p50_ms": {constant}'), encoding="utf-8")

    with pytest.raises(ValueError, match="summary.json.latency_ms.complete_frame.p50_ms"):
        load_run_view(run_dir)


def test_annotation_paths_are_filename_sorted_and_contained(tmp_path: Path) -> None:
    run_dir = write_completed_run_fixture(tmp_path)

    assert [path.name for path in load_run_view(run_dir).annotation_paths] == [
        "000000.png",
        "000002.png",
    ]

    escaped = tmp_path / "escaped.png"
    Image.new("RGB", (1, 1)).save(escaped)
    link = run_dir / "annotated" / "000001.png"
    try:
        os.symlink(escaped, link)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary["annotated_frame_count"] = 3
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="annotated/000001.png"):
        load_run_view(run_dir)


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
    assert widget.annotationList.item(0).text() == "000000.png"
    assert len([item for item in widget.imageScene.items() if isinstance(item, QGraphicsPixmapItem)]) == 1
    assert widget.regionsTable.rowCount() == 2
    assert widget.regionsTable.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert snapshot_tree(run_dir) == before


def test_results_widget_formats_unavailable_telemetry_as_na(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    run_dir = write_completed_run_fixture(tmp_path, annotation_names=())
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    hardware = summary["hardware_peaks"]
    assert isinstance(hardware, dict)
    hardware["process_rss_bytes"] = None
    summary["detector_peak_device_memory_bytes"] = None
    _write_json(summary_path, summary)
    widget = ResultsWidget()
    qtbot.addWidget(widget)

    widget.load_run(run_dir)

    assert widget.peakRssLabel.text() == "N/A"
    assert widget.peakVramLabel.text() == "N/A"
    assert widget.nvmlGpuMemoryLabel.text() == "N/A"


def test_capture_provenance_is_typed(tmp_path: Path) -> None:
    expected = _capture(tmp_path / "historical-capture.mp4")
    run_dir = write_completed_run_fixture(tmp_path, capture=expected)

    view = load_run_view(run_dir)

    assert view.capture == expected
