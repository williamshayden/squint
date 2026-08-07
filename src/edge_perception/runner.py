"""Synchronous full-frame and declared-crop checkpoint orchestration."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, cast
from uuid import uuid4

from edge_perception.config import RunConfig as _RunConfig
from edge_perception.contracts import Detection, Region
from edge_perception.detector import Detector
from edge_perception.geometry import (
    crop_region,
    full_frame_region,
    map_detection_to_source,
    validate_region,
)
from edge_perception.outputs import RunOutputs, summarize_latencies
from edge_perception.progress import (
    CancelCheck,
    ProgressCallback,
    ProgressEvent,
    ProgressPhase,
)
from edge_perception.telemetry import TelemetryMonitor, collect_host_report
from edge_perception.video import DecodedFrame, iter_video

RunConfig = _RunConfig
RunStatus = Literal["complete", "cancelled", "failed"]


def _milliseconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("adaptive-edge-perception", "av", "numpy", "pillow", "psutil"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _video_iterator(path: Path) -> Generator[DecodedFrame, None, None]:
    return cast(Generator[DecodedFrame, None, None], iter_video(path))


def validate_output_directory(output_dir: Path) -> None:
    """Allow absent or empty run directories while preserving existing contents."""

    resolved = Path(output_dir).resolve()
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise ValueError(f"output is not a directory: {resolved}")
    if any(resolved.iterdir()):
        raise ValueError(f"output directory must be empty: {resolved}")


def _preview_and_validate(
    config: RunConfig,
) -> tuple[DecodedFrame, Region, Generator[DecodedFrame, None, None]]:
    preview = _video_iterator(config.input_path)
    try:
        try:
            first_frame = next(preview)
        except StopIteration as error:
            raise ValueError(f"video contains no decoded frames: {config.input_path}") from error

        frame_height, frame_width, _channels = first_frame.image.shape
        full_region = full_frame_region(frame_width, frame_height)
        for region in config.regions:
            validate_region(region, frame_width, frame_height)
        return first_frame, full_region, preview
    except BaseException:
        preview.close()
        raise


def _warm_up(config: RunConfig, detector: Detector, first_frame: DecodedFrame) -> None:
    warmed_shapes: set[tuple[int, int]] = set()
    warmup_images = [first_frame.image]
    warmup_images.extend(crop_region(first_frame.image, region) for region in config.regions)
    for image in warmup_images:
        shape = (int(image.shape[0]), int(image.shape[1]))
        if shape in warmed_shapes:
            continue
        warmed_shapes.add(shape)
        detector.warmup(image, config.warmup_runs)


def _timing_definitions() -> dict[str, str]:
    return {
        "decode": "wall time spent requesting one chronological decoded RGB frame",
        "crop": "wall time spent validating and copying one declared crop",
        "detector": "detector-reported total preprocessing, inference, and postprocessing time",
        "coordinate_mapping": "wall time spent mapping one inference's detections to source pixels",
        "frame_pipeline": "wall time from decode request through annotation and durable frame flush",
        "serialization": "wall time spent writing inference and detection rows and flushing the frame",
        "annotation": "wall time spent rendering and saving a scheduled diagnostic PNG",
    }


def _manifest(
    config: RunConfig,
    detector: Detector,
    first_frame: DecodedFrame,
    full_region: Region,
) -> dict[str, object]:
    return {
        "configuration": {
            "input_path": str(config.input_path.resolve()),
            "output_dir": str(config.output_dir.resolve()),
            "regions": [region.to_dict() for region in config.regions],
            "execution_regions": [
                full_region.to_dict(),
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
            "sha256": _sha256_file(config.input_path),
            "frame_width": int(first_frame.image.shape[1]),
            "frame_height": int(first_frame.image.shape[0]),
            "capture": None if config.capture is None else config.capture.to_dict(),
        },
        "host": collect_host_report(),
        "detector": detector.identity.to_dict(),
        "dependencies": _dependency_versions(),
        "timing_definitions": _timing_definitions(),
    }


def _empty_stage_values() -> dict[str, list[float]]:
    return {name: [] for name in _timing_definitions()}


def _empty_hardware_peaks() -> dict[str, int | float | None]:
    return {
        "process_rss_bytes": None,
        "system_memory_used_bytes": None,
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }


@contextmanager
def _optional_telemetry_monitor() -> Generator[TelemetryMonitor | None, None, None]:
    try:
        monitor = TelemetryMonitor()
    except Exception:  # noqa: BLE001 - telemetry construction is optional
        yield None
        return

    try:
        monitor.__enter__()
    except Exception:  # noqa: BLE001 - telemetry startup is optional
        try:
            monitor.__exit__(*sys.exc_info())
        except Exception:  # noqa: BLE001, S110 - telemetry shutdown is optional
            pass
        yield None
        return

    try:
        yield monitor
    finally:
        try:
            monitor.__exit__(*sys.exc_info())
        except Exception:  # noqa: BLE001, S110 - telemetry shutdown is optional
            pass


def _summary(
    *,
    status: RunStatus,
    failure: BaseException | None,
    frames_processed: int,
    inference_count: int,
    annotated_frame_count: int,
    full_frame_latencies: list[float],
    crop_latencies: list[float],
    complete_frame_latencies: list[float],
    stage_values: dict[str, list[float]],
    hardware_peaks: dict[str, int | float | None],
    peak_device_memory_bytes: int | None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": status,
        "frames_processed": frames_processed,
        "inference_count": inference_count,
        "annotated_frame_count": annotated_frame_count,
        "latency_ms": {
            "full_frame": summarize_latencies(full_frame_latencies),
            "crop": summarize_latencies(crop_latencies),
            "complete_frame": summarize_latencies(complete_frame_latencies),
        },
        "stage_latency_ms": {
            name: summarize_latencies(values) for name, values in stage_values.items()
        },
        "hardware_peaks": hardware_peaks,
        "detector_peak_device_memory_bytes": peak_device_memory_bytes,
    }
    if failure is not None:
        summary["error"] = f"{type(failure).__name__}: {failure}"
    return summary


def run_checkpoint(
    config: RunConfig,
    detector: Detector,
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCheck | None = None,
) -> dict[str, object]:
    """Run one chronological, batch-size-one checkpoint and finalize its artifacts."""

    started_ns = perf_counter_ns()
    frames_processed = 0
    inference_count = 0
    annotated_frame_count = 0
    full_frame_latencies: list[float] = []
    crop_latencies: list[float] = []
    complete_frame_latencies: list[float] = []
    stage_values = _empty_stage_values()
    failure: BaseException | None = None
    status: RunStatus = "complete"
    summary: dict[str, object] = {}
    monitor: TelemetryMonitor | None = None

    def emit(phase: ProgressPhase, error: str | None = None) -> None:
        if progress is None:
            return
        progress(
            ProgressEvent(
                phase=phase,
                frames_processed=frames_processed,
                inference_count=inference_count,
                elapsed_ms=_milliseconds(started_ns, perf_counter_ns()),
                error=error,
            )
        )

    emit("validating")
    try:
        if not config.input_path.is_file():
            raise FileNotFoundError(f"input video does not exist: {config.input_path}")
        validate_output_directory(config.output_dir)
        first_frame, full_region, preview = _preview_and_validate(config)

        try:
            manifest = _manifest(config, detector, first_frame, full_region)
            run_id = uuid4().hex
            with RunOutputs(config.output_dir, run_id=run_id, manifest=manifest) as outputs:
                try:
                    try:
                        emit("warming_up")
                        _warm_up(config, detector, first_frame)
                    finally:
                        preview.close()

                    if cancel_requested is not None and cancel_requested():
                        status = "cancelled"
                    else:
                        with _optional_telemetry_monitor() as monitor:
                            measured = _video_iterator(config.input_path)
                            try:
                                while (
                                    config.max_frames is None
                                    or frames_processed < config.max_frames
                                ):
                                    frame_pipeline_start_ns = perf_counter_ns()
                                    decode_start_ns = frame_pipeline_start_ns
                                    try:
                                        frame = next(measured)
                                    except StopIteration:
                                        break
                                    decode_end_ns = perf_counter_ns()
                                    frame_stage_values = _empty_stage_values()
                                    frame_stage_values["decode"].append(
                                        _milliseconds(decode_start_ns, decode_end_ns)
                                    )

                                    frame_height, frame_width, _channels = frame.image.shape
                                    if (
                                        frame_width != full_region.width
                                        or frame_height != full_region.height
                                    ):
                                        raise ValueError(
                                            "video frame dimensions changed during checkpoint"
                                        )

                                    frame_id = f"frame-{frame.frame_index:06d}"
                                    frame_detections: list[Detection] = []
                                    inference_records: list[dict[str, object]] = []
                                    detection_batches: list[
                                        tuple[str, str, str, tuple[Detection, ...]]
                                    ] = []
                                    frame_full_frame_latencies: list[float] = []
                                    frame_crop_latencies: list[float] = []
                                    execution_regions = (full_region, *config.regions)

                                    for region_index, region in enumerate(execution_regions):
                                        region_pipeline_start_ns = perf_counter_ns()
                                        crop_ms = 0.0
                                        if region_index == 0:
                                            image = frame.image
                                        else:
                                            crop_start_ns = perf_counter_ns()
                                            image = crop_region(frame.image, region)
                                            crop_end_ns = perf_counter_ns()
                                            crop_ms = _milliseconds(
                                                crop_start_ns, crop_end_ns
                                            )
                                            frame_stage_values["crop"].append(crop_ms)

                                        prediction = detector.predict((image,))
                                        if len(prediction.detections) != 1:
                                            raise RuntimeError(
                                                "detector prediction cardinality mismatch: "
                                                f"expected 1, got {len(prediction.detections)}"
                                            )
                                        frame_stage_values["detector"].append(
                                            prediction.timing.total_ms
                                        )

                                        mapping_start_ns = perf_counter_ns()
                                        mapped = tuple(
                                            map_detection_to_source(
                                                detection,
                                                region,
                                                frame_width,
                                                frame_height,
                                            )
                                            for detection in prediction.detections[0]
                                        )
                                        mapping_end_ns = perf_counter_ns()
                                        frame_stage_values["coordinate_mapping"].append(
                                            _milliseconds(mapping_start_ns, mapping_end_ns)
                                        )
                                        region_latency = _milliseconds(
                                            region_pipeline_start_ns,
                                            mapping_end_ns,
                                        )
                                        if region_index == 0:
                                            frame_full_frame_latencies.append(region_latency)
                                        else:
                                            frame_crop_latencies.append(region_latency)

                                        inference_id = (
                                            f"inference-{frame.frame_index:06d}-{region_index:03d}"
                                        )
                                        inference_records.append(
                                            {
                                                "frame_id": frame_id,
                                                "frame_index": frame.frame_index,
                                                "source_time_ms": frame.source_time_ms,
                                                "inference_id": inference_id,
                                                "region_id": region.region_id,
                                                "region": region.to_dict(),
                                                "input_shape": [
                                                    int(value) for value in image.shape
                                                ],
                                                "frame_decode_ms": _milliseconds(
                                                    decode_start_ns,
                                                    decode_end_ns,
                                                ),
                                                "crop_ms": crop_ms,
                                                "coordinate_mapping_ms": _milliseconds(
                                                    mapping_start_ns,
                                                    mapping_end_ns,
                                                ),
                                                "region_pipeline_ms": region_latency,
                                                "detector_timing": prediction.timing.to_dict(),
                                            }
                                        )
                                        detection_batches.append(
                                            (
                                                frame_id,
                                                inference_id,
                                                region.region_id,
                                                mapped,
                                            )
                                        )
                                        frame_detections.extend(mapped)

                                    annotation = None
                                    if (
                                        config.annotate_every > 0
                                        and frame.frame_index % config.annotate_every == 0
                                    ):
                                        annotation = (
                                            frame.frame_index,
                                            frame.image,
                                            execution_regions,
                                            tuple(frame_detections),
                                        )

                                    annotation_ms, serialization_ms = outputs.commit_frame(
                                        inferences=inference_records,
                                        detection_batches=detection_batches,
                                        annotation=annotation,
                                    )
                                    frame_stage_values["serialization"].append(
                                        serialization_ms
                                    )
                                    if annotation_ms is not None:
                                        frame_stage_values["annotation"].append(annotation_ms)
                                    frame_pipeline_end_ns = perf_counter_ns()
                                    complete_frame_latency = _milliseconds(
                                        frame_pipeline_start_ns,
                                        frame_pipeline_end_ns,
                                    )
                                    frame_stage_values["frame_pipeline"].append(
                                        complete_frame_latency
                                    )

                                    for stage_name, values in frame_stage_values.items():
                                        stage_values[stage_name].extend(values)
                                    full_frame_latencies.extend(
                                        frame_full_frame_latencies
                                    )
                                    crop_latencies.extend(frame_crop_latencies)
                                    complete_frame_latencies.append(complete_frame_latency)
                                    inference_count += len(inference_records)
                                    if annotation is not None:
                                        annotated_frame_count += 1
                                    frames_processed += 1
                                    emit("running")
                                    if (
                                        cancel_requested is not None
                                        and cancel_requested()
                                    ):
                                        status = "cancelled"
                                        break
                            finally:
                                measured.close()
                except BaseException as error:
                    failure = error
                    status = "failed"
                    raise
                finally:
                    if monitor is None:
                        hardware_peaks = _empty_hardware_peaks()
                    else:
                        try:
                            for sample in monitor.samples:
                                outputs.write_hardware(sample.to_dict())
                            hardware_peaks = monitor.peaks()
                        except Exception:  # noqa: BLE001 - telemetry is optional
                            hardware_peaks = _empty_hardware_peaks()
                    try:
                        peak_device_memory_bytes = detector.peak_device_memory_bytes()
                    except Exception:  # noqa: BLE001 - optional detector telemetry
                        peak_device_memory_bytes = None
                    summary = _summary(
                        status=status,
                        failure=failure,
                        frames_processed=frames_processed,
                        inference_count=inference_count,
                        annotated_frame_count=annotated_frame_count,
                        full_frame_latencies=full_frame_latencies,
                        crop_latencies=crop_latencies,
                        complete_frame_latencies=complete_frame_latencies,
                        stage_values=stage_values,
                        hardware_peaks=hardware_peaks,
                        peak_device_memory_bytes=peak_device_memory_bytes,
                    )
                    outputs.write_summary(summary)
        finally:
            preview.close()
    except BaseException as error:
        emit("failed", f"{type(error).__name__}: {error}")
        raise

    emit(status)
    return summary
