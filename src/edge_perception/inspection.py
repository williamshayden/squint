"""Terminal presentation for an already-validated canonical run."""

from __future__ import annotations

from edge_perception.run_view import RunViewData

_STATUS_LABELS = {
    "complete": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _optional_measure(value: float | None, unit: str) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f} {unit}"


def _optional_count(value: int | None, unit: str) -> str:
    if value is None:
        return "N/A"
    return f"{value} {unit}"


def _capture_provenance(view: RunViewData) -> str:
    capture = view.capture
    if capture is None:
        return "Capture provenance\n  capture: N/A"
    request = capture.request
    return "\n".join(
        (
            "Capture provenance",
            "  request:",
            f"    device: {request.device_id}",
            f"    device description: {request.device_description}",
            f"    requested width: {_optional_count(request.requested_width, 'px')}",
            f"    requested height: {_optional_count(request.requested_height, 'px')}",
            f"    requested FPS: {_optional_measure(request.requested_fps, 'fps')}",
            f"    strict: {str(request.strict).lower()}",
            "  selected format:",
            f"    dimensions: {capture.selected_width} x {capture.selected_height} px",
            (
                "    FPS range: "
                f"{capture.selected_min_fps:.3f}-{capture.selected_max_fps:.3f} fps"
            ),
            f"    pixel format: {capture.selected_pixel_format}",
            "  recorded format:",
            f"    dimensions: {capture.actual_width} x {capture.actual_height} px",
            f"    FPS: {capture.actual_fps:.3f} fps",
            f"    container: {capture.container}",
            f"    codec: {capture.codec}",
            f"    duration: {capture.duration_seconds:.3f} s",
            f"    audio: {str(capture.has_audio).lower()}",
            f"    file size: {capture.file_size_bytes} bytes",
            f"    path: {capture.path}",
            f"  SHA-256: {capture.sha256}",
        )
    )


def _regions(view: RunViewData) -> tuple[str, ...]:
    if not view.regions:
        return ("  regions: N/A",)
    return (
        "  regions:",
        *(
            f"    {region.region_id}: x={region.x} px y={region.y} px "
            f"width={region.width} px height={region.height} px"
            for region in view.regions
        ),
    )


def _annotations(view: RunViewData) -> str:
    if not view.annotation_paths:
        return "Annotations\n  count: 0\n  paths: N/A"
    return "\n".join(
        (
            "Annotations",
            f"  count: {view.annotated_frame_count}",
            "  paths:",
            *(f"    {path}" for path in view.annotation_paths),
        )
    )


def render_run_inspection(view: RunViewData) -> str:
    """Render one validated canonical run for human terminal inspection."""

    run_lines = [
        "Run",
        f"  status: {_STATUS_LABELS[view.status]}",
        f"  directory: {view.run_dir}",
    ]
    if view.error is not None:
        run_lines.append(f"  error: {view.error}")
    configuration_lines = [
        "Run configuration",
        f"  detector: {view.detector_model_id}",
        f"  detector revision: {view.detector_revision}",
        f"  device: {view.device}",
        f"  threshold: {view.threshold:.3f}",
        *_regions(view),
    ]
    metrics = "\n".join(
        (
            "Metrics",
            f"  frames processed: {view.frames_processed}",
            f"  inference count: {view.inference_count}",
            f"  frame latency p50: {_optional_measure(view.frame_p50_ms, 'ms')}",
            f"  frame latency p95: {_optional_measure(view.frame_p95_ms, 'ms')}",
            f"  frame latency p99: {_optional_measure(view.frame_p99_ms, 'ms')}",
            f"  peak process RSS: {_optional_count(view.peak_rss_bytes, 'bytes')}",
            f"  peak device memory: {_optional_count(view.peak_vram_bytes, 'bytes')}",
        )
    )
    source = "\n".join(
        (
            "Source provenance",
            f"  path: {view.source_path}",
            f"  dimensions: {view.source_width} x {view.source_height} px",
        )
    )
    return "\n\n".join(
        (
            "\n".join(run_lines),
            metrics,
            "\n".join(configuration_lines),
            source,
            _capture_provenance(view),
            _annotations(view),
        )
    )
