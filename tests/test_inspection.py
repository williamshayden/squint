from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.contracts import Region
from edge_perception.inspection import render_run_inspection
from edge_perception.run_view import RunViewData, load_run_view


def _capture(
    tmp_path: Path,
    *,
    path: Path | None = None,
    sha256: str = "b" * 64,
) -> CaptureResult:
    return CaptureResult(
        request=CaptureRequest("camera-1", "Desk camera", 1920, None, 30.0, True),
        selected_width=1920,
        selected_height=1080,
        selected_min_fps=24.0,
        selected_max_fps=60.0,
        selected_pixel_format="NV12",
        actual_width=1920,
        actual_height=1080,
        actual_fps=29.97,
        container="mp4",
        codec="h264",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=1234,
        path=tmp_path / "capture.mp4" if path is None else path,
        sha256=sha256,
    )


def _write_run_view_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    (run_dir / "annotated").mkdir(parents=True)
    source_path = (tmp_path / "historical-source.mp4").resolve()
    source_sha256 = "b" * 64
    manifest = {
        "schema_version": "0.1.0",
        "run_id": "inspection-run-id",
        "configuration": {
            "threshold": 0.35,
            "regions": [],
        },
        "source_video": {
            "path": str(source_path),
            "sha256": source_sha256,
            "frame_width": 640,
            "frame_height": 480,
            "capture": _capture(
                tmp_path,
                path=source_path,
                sha256=source_sha256,
            ).to_dict(),
        },
        "detector": {
            "model_id": "tests/fake-detector",
            "revision": "test-revision",
            "device": "cpu",
        },
    }
    summary = {
        "schema_version": "0.1.0",
        "run_id": "inspection-run-id",
        "status": "complete",
        "frames_processed": 0,
        "inference_count": 0,
        "annotated_frame_count": 0,
        "latency_ms": {
            "complete_frame": {
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
            }
        },
        "hardware_peaks": {"process_rss_bytes": None},
        "detector_peak_device_memory_bytes": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir, source_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("capture_sha", r"manifest\.json\.source_video\.capture\.sha256"),
        ("capture_path", r"manifest\.json\.source_video\.capture\.path"),
    ],
)
def test_load_run_view_rejects_inconsistent_capture_provenance(
    mutation: str,
    message: str,
    tmp_path: Path,
) -> None:
    run_dir, _source_path = _write_run_view_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["source_video"]["capture"]
    if mutation == "capture_sha":
        capture["sha256"] = "c" * 64
    else:
        capture["path"] = str((tmp_path / "different-source.mp4").resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_run_view(run_dir)


def test_load_run_view_rejects_invalid_source_sha256(tmp_path: Path) -> None:
    run_dir, _source_path = _write_run_view_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_video"]["sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest\.json\.source_video\.sha256"):
        load_run_view(run_dir)


def test_load_run_view_accepts_consistent_capture_without_original_media(
    tmp_path: Path,
) -> None:
    run_dir, source_path = _write_run_view_fixture(tmp_path)

    view = load_run_view(run_dir)

    assert not source_path.exists()
    assert view.source_path == source_path
    assert view.capture is not None
    assert view.capture.path == source_path
    assert view.capture.sha256 == "b" * 64


def _view(
    tmp_path: Path,
    *,
    status: str = "complete",
    frame_p50_ms: float | None = 0.5,
    frame_p95_ms: float | None = 1.5,
    frame_p99_ms: float | None = 2.5,
    peak_rss_bytes: int | None = 4096,
    peak_vram_bytes: int | None = 8192,
    capture: CaptureResult | None = None,
    annotation_paths: tuple[Path, ...] = (),
    error: str | None = None,
) -> RunViewData:
    return RunViewData(
        run_dir=tmp_path / "run",
        status=status,  # type: ignore[arg-type]
        frames_processed=3,
        inference_count=6,
        annotated_frame_count=len(annotation_paths),
        frame_p50_ms=frame_p50_ms,
        frame_p95_ms=frame_p95_ms,
        frame_p99_ms=frame_p99_ms,
        peak_rss_bytes=peak_rss_bytes,
        peak_vram_bytes=peak_vram_bytes,
        detector_model_id="tests/fake-detector",
        detector_revision="test-revision",
        device="cuda:1",
        threshold=0.35,
        source_path=tmp_path / "source.mp4",
        source_width=640,
        source_height=480,
        capture=capture,
        regions=(Region("focus", 10, 20, 300, 200),),
        annotation_paths=annotation_paths,
        error=error,
    )


def test_render_run_inspection_renders_complete_canonical_view(tmp_path: Path) -> None:
    annotation = tmp_path / "run" / "annotated" / "000000.png"
    view = _view(tmp_path, capture=_capture(tmp_path), annotation_paths=(annotation,))

    rendered = render_run_inspection(view)

    assert "Run\n  status: Completed" in rendered
    assert f"  directory: {view.run_dir}" in rendered
    assert "Metrics\n  frames processed: 3" in rendered
    assert "  inference count: 6" in rendered
    assert "  frame latency p50: 0.500 ms" in rendered
    assert "  frame latency p95: 1.500 ms" in rendered
    assert "  frame latency p99: 2.500 ms" in rendered
    assert "  peak process RSS: 4096 bytes" in rendered
    assert "  peak device memory: 8192 bytes" in rendered
    assert "Run configuration\n  detector: tests/fake-detector" in rendered
    assert "  detector revision: test-revision" in rendered
    assert "  device: cuda:1" in rendered
    assert "  threshold: 0.350" in rendered
    assert "  regions:\n    focus: x=10 px y=20 px width=300 px height=200 px" in rendered
    assert "Source provenance\n  path:" in rendered
    assert "  dimensions: 640 x 480 px" in rendered
    assert "Capture provenance\n  request:" in rendered
    assert "    device: camera-1" in rendered
    assert "    device description: Desk camera" in rendered
    assert "    requested width: 1920 px" in rendered
    assert "    requested height: N/A" in rendered
    assert "    requested FPS: 30.000 fps" in rendered
    assert "    strict: true" in rendered
    assert "  selected format:\n    dimensions: 1920 x 1080 px" in rendered
    assert "    FPS range: 24.000-60.000 fps" in rendered
    assert "    pixel format: NV12" in rendered
    assert "  recorded format:\n    dimensions: 1920 x 1080 px" in rendered
    assert "    FPS: 29.970 fps" in rendered
    assert "    container: mp4" in rendered
    assert "    codec: h264" in rendered
    assert "    duration: 0.100 s" in rendered
    assert "    audio: false" in rendered
    assert "    file size: 1234 bytes" in rendered
    assert f"    path: {tmp_path / 'capture.mp4'}" in rendered
    assert "  SHA-256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in rendered
    assert "Annotations\n  count: 1" in rendered
    assert f"  paths:\n    {annotation}" in rendered
    assert "Error" not in rendered


@pytest.mark.parametrize(
    ("status", "error", "expected_status", "expected_error"),
    [
        ("failed", "RuntimeError: detector failed", "Failed", "  error: RuntimeError: detector failed"),
        ("cancelled", None, "Cancelled", None),
    ],
)
def test_render_run_inspection_maps_status_and_only_renders_present_error(
    tmp_path: Path,
    status: str,
    error: str | None,
    expected_status: str,
    expected_error: str | None,
) -> None:
    rendered = render_run_inspection(_view(tmp_path, status=status, error=error))

    assert f"Run\n  status: {expected_status}" in rendered
    if expected_error is None:
        assert "  error:" not in rendered
    else:
        assert expected_error in rendered


def test_render_run_inspection_renders_missing_metrics_and_annotations_as_na(tmp_path: Path) -> None:
    rendered = render_run_inspection(
        _view(
            tmp_path,
            frame_p50_ms=None,
            frame_p95_ms=None,
            frame_p99_ms=None,
            peak_rss_bytes=None,
            peak_vram_bytes=None,
        )
    )

    assert "  frame latency p50: N/A" in rendered
    assert "  frame latency p95: N/A" in rendered
    assert "  frame latency p99: N/A" in rendered
    assert "  peak process RSS: N/A" in rendered
    assert "  peak device memory: N/A" in rendered
    assert "Annotations\n  count: 0\n  paths: N/A" in rendered
