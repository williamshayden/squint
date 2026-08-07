from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Self

import av
import numpy as np
import pytest
from conftest import FakeDetector

from edge_perception import runner as runner_module
from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.contracts import Region
from edge_perception.outputs import RunOutputs
from edge_perception.progress import ProgressEvent
from edge_perception.runner import RunConfig, run_checkpoint
from edge_perception.telemetry import TelemetrySample
from edge_perception.video import iter_video


def _json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _capture_for_video(video_path: Path) -> CaptureResult:
    return CaptureResult(
        request=CaptureRequest("camera-1", "EMEET", 200, 100, 30.0, True),
        selected_width=200,
        selected_height=100,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=200,
        actual_height=100,
        actual_fps=30.0,
        container="mp4",
        codec="mpeg4",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=video_path.stat().st_size,
        path=video_path,
        sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
    )


def _write_red_video(path: Path, red_values: tuple[int, ...]) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 200
        stream.height = 100
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)

        for frame_index, red_value in enumerate(red_values):
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[..., 0] = red_value
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _isolated_snapshot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    snapshot_dir = tmp_path / "private-snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(snapshot_dir))
    return snapshot_dir


def test_run_checkpoint_processes_full_frame_then_declared_crops(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = RunConfig(
        input_path=video_path,
        output_dir=run_dir,
        regions=(Region("right", 100, 20, 80, 60),),
        threshold=0.3,
        max_frames=None,
        warmup_runs=2,
        annotate_every=1,
    )

    returned = run_checkpoint(config, fake_detector)

    inferences = _json_lines(run_dir / "inferences.jsonl")
    detections = _json_lines(run_dir / "detections.jsonl")
    hardware = _json_lines(run_dir / "hardware.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(inferences) == 6
    assert [(row["frame_id"], row["region_id"]) for row in inferences] == [
        ("frame-000000", "full-frame"),
        ("frame-000000", "right"),
        ("frame-000001", "full-frame"),
        ("frame-000001", "right"),
        ("frame-000002", "full-frame"),
        ("frame-000002", "right"),
    ]
    assert [row["inference_id"] for row in inferences] == [
        "inference-000000-000",
        "inference-000000-001",
        "inference-000001-000",
        "inference-000001-001",
        "inference-000002-000",
        "inference-000002-001",
    ]
    assert fake_detector.predict_batch_sizes == [1] * 6
    assert fake_detector.warmup_calls == [((100, 200, 3), 2), ((60, 80, 3), 2)]
    assert [row["box"] for row in detections[:2]] == [
        [10.0, 5.0, 20.0, 15.0],
        [110.0, 25.0, 120.0, 35.0],
    ]
    assert len(list((run_dir / "annotated").glob("*.png"))) == 3
    assert hardware

    for section, count in (("full_frame", 3), ("crop", 3), ("complete_frame", 3)):
        assert summary["latency_ms"][section]["count"] == count
        assert set(summary["latency_ms"][section]) == {"count", "p50_ms", "p95_ms", "p99_ms"}
    assert set(summary["stage_latency_ms"]) == {
        "decode",
        "crop",
        "detector",
        "coordinate_mapping",
        "frame_pipeline",
        "serialization",
        "annotation",
    }
    assert returned == {key: value for key, value in summary.items() if key not in {"run_id", "schema_version"}}
    assert manifest["configuration"]["threshold"] == 0.3
    assert manifest["configuration"]["batch_size"] == 1
    assert manifest["source_video"]["sha256"] == hashlib.sha256(video_path.read_bytes()).hexdigest()
    assert manifest["detector"] == fake_detector.identity.to_dict()
    assert set(manifest["timing_definitions"]) == set(summary["stage_latency_ms"])
    assert manifest["source_video"]["capture"] is None
    assert manifest["configuration"]["detector_id"] == "dfine-nano-coco"
    assert manifest["configuration"]["device"] == "auto"


def test_run_checkpoint_records_capture_provenance_and_requested_overrides(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    request = CaptureRequest("camera-1", "EMEET", 200, 100, 30.0, True)
    capture = CaptureResult(
        request=request,
        selected_width=200,
        selected_height=100,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=200,
        actual_height=100,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=video_path.stat().st_size,
        path=video_path,
        sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
    )
    config = RunConfig(
        input_path=video_path,
        output_dir=tmp_path / "captured-run",
        regions=(),
        threshold=0.3,
        max_frames=0,
        warmup_runs=0,
        annotate_every=0,
        detector_id="custom-detector",
        device="cuda",
        capture=capture,
    )

    run_checkpoint(config, fake_detector)

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_video"]["capture"] == capture.to_dict()
    assert manifest["configuration"]["detector_id"] == "custom-detector"
    assert manifest["configuration"]["device"] == "cuda"


def test_run_checkpoint_preflight_rejects_capture_sha_before_output_claim(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    capture = CaptureResult(
        request=CaptureRequest("camera-1", "EMEET", 200, 100, 30.0, True),
        selected_width=200,
        selected_height=100,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=200,
        actual_height=100,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=video_path.stat().st_size,
        path=video_path,
        sha256="0" * 64,
    )
    output_dir = tmp_path / "captured-run"
    config = RunConfig(
        input_path=video_path,
        output_dir=output_dir,
        regions=(),
        threshold=0.3,
        max_frames=0,
        warmup_runs=0,
        annotate_every=0,
        capture=capture,
    )

    with pytest.raises(ValueError, match="capture SHA-256"):
        run_checkpoint(config, fake_detector)

    assert not output_dir.exists()
    assert fake_detector.warmup_calls == []
    assert fake_detector.predict_batch_sizes == []


def test_capture_replacement_after_preflight_cannot_claim_output(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = _isolated_snapshot_directory(tmp_path, monkeypatch)
    capture = _capture_for_video(video_path)
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(video_path.read_bytes() + b"replacement")
    output_dir = tmp_path / "capture-replaced-before-snapshot"
    actual_preflight = runner_module.preflight_run

    def replace_after_preflight(config: RunConfig) -> object:
        result = actual_preflight(config)
        os.replace(replacement, video_path)
        return result

    monkeypatch.setattr(runner_module, "preflight_run", replace_after_preflight)
    config = RunConfig(video_path, output_dir, (), 0.3, 0, 0, 0, capture=capture)

    with pytest.raises(ValueError, match="capture source changed after preflight"):
        run_checkpoint(config, fake_detector)

    assert not output_dir.exists()
    assert list(snapshot_dir.iterdir()) == []
    assert fake_detector.warmup_calls == []
    assert fake_detector.predict_batch_sizes == []


def test_capture_run_processes_snapshot_after_original_is_replaced(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = _isolated_snapshot_directory(tmp_path, monkeypatch)
    capture = _capture_for_video(video_path)
    original_red_values = tuple(
        int(frame.image[0, 0, 0]) for frame in iter_video(video_path)
    )
    replacement = tmp_path / "replacement.mp4"
    _write_red_video(replacement, (240, 140, 40))
    replacement_sha256 = hashlib.sha256(replacement.read_bytes()).hexdigest()
    observed_red_values: list[int] = []
    original_predict = fake_detector.predict
    replaced = False

    def replace_original_without_cancelling() -> bool:
        nonlocal replaced
        if not replaced:
            os.replace(replacement, video_path)
            replaced = True
        return False

    def record_red_values(images: tuple[np.ndarray, ...]) -> object:
        observed_red_values.append(int(images[0][0, 0, 0]))
        return original_predict(images)

    fake_detector.predict = record_red_values  # type: ignore[method-assign]
    output_dir = tmp_path / "capture-pinned"
    config = RunConfig(video_path, output_dir, (), 0.3, None, 0, 0, capture=capture)

    summary = run_checkpoint(
        config,
        fake_detector,
        cancel_requested=replace_original_without_cancelling,
    )

    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert summary["status"] == "complete"
    assert observed_red_values == list(original_red_values)
    assert hashlib.sha256(video_path.read_bytes()).hexdigest() == replacement_sha256
    assert manifest["source_video"]["path"] == str(video_path.resolve())
    assert manifest["source_video"]["sha256"] == capture.sha256
    assert manifest["source_video"]["capture"] == capture.to_dict()
    assert str(snapshot_dir) not in manifest_text
    assert list(snapshot_dir.iterdir()) == []


def test_capture_snapshot_cleanup_failure_preserves_primary_runner_failure(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = _isolated_snapshot_directory(tmp_path, monkeypatch)
    capture = _capture_for_video(video_path)
    original_unlink = Path.unlink
    cleanup_attempts: list[Path] = []

    def fail_snapshot_cleanup(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path.parent == snapshot_dir:
            cleanup_attempts.append(path)
            raise OSError("snapshot cleanup failed")
        original_unlink(path, missing_ok=missing_ok)

    def fail_predict(_images: tuple[object, ...]) -> object:
        raise RuntimeError("primary inference failure")

    monkeypatch.setattr(Path, "unlink", fail_snapshot_cleanup)
    fake_detector.predict = fail_predict  # type: ignore[method-assign]
    output_dir = tmp_path / "capture-primary-failure"
    config = RunConfig(video_path, output_dir, (), 0.3, 1, 0, 0, capture=capture)

    with pytest.raises(RuntimeError, match="primary inference failure"):
        run_checkpoint(config, fake_detector)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: primary inference failure"
    assert cleanup_attempts == list(snapshot_dir.iterdir())
    assert len(cleanup_attempts) == 1
    original_unlink(cleanup_attempts[0])


def test_run_checkpoint_finalizes_outputs_and_telemetry_when_detector_fails(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "failed-run"

    def fail(_images: tuple[object, ...]) -> object:
        raise RuntimeError("detector failed")

    fake_detector.predict = fail  # type: ignore[method-assign]
    config = RunConfig(
        input_path=video_path,
        output_dir=run_dir,
        regions=(),
        threshold=0.3,
        max_frames=1,
        warmup_runs=0,
        annotate_every=0,
    )

    with pytest.raises(RuntimeError, match="detector failed"):
        run_checkpoint(config, fake_detector)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: detector failed"
    assert _json_lines(run_dir / "hardware.jsonl")
    for stream_name in ("inferences.jsonl", "detections.jsonl", "hardware.jsonl"):
        with (run_dir / stream_name).open("a", encoding="utf-8") as stream:
            stream.write("")


def test_partial_frame_failure_publishes_no_rows_counts_or_annotation(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "partial-frame"
    successful_predict = fake_detector.predict
    predict_count = 0

    def fail_first_crop(images: tuple[object, ...]) -> object:
        nonlocal predict_count
        predict_count += 1
        if predict_count == 2:
            raise RuntimeError("crop inference failed")
        return successful_predict(images)  # type: ignore[arg-type]

    fake_detector.predict = fail_first_crop  # type: ignore[method-assign]
    config = RunConfig(
        input_path=video_path,
        output_dir=run_dir,
        regions=(Region("right", 100, 20, 80, 60),),
        threshold=0.3,
        max_frames=1,
        warmup_runs=0,
        annotate_every=1,
    )

    with pytest.raises(RuntimeError, match="crop inference failed"):
        run_checkpoint(config, fake_detector)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: crop inference failed"
    assert summary["frames_processed"] == 0
    assert summary["inference_count"] == 0
    assert summary["annotated_frame_count"] == 0
    assert _json_lines(run_dir / "inferences.jsonl") == []
    assert _json_lines(run_dir / "detections.jsonl") == []
    assert list((run_dir / "annotated").glob("*.png")) == []


def test_primary_inference_error_survives_secondary_peak_memory_error(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "primary-error"

    def fail_predict(_images: tuple[object, ...]) -> object:
        raise RuntimeError("primary inference failure")

    def fail_peak_memory() -> int | None:
        raise RuntimeError("secondary peak failure")

    fake_detector.predict = fail_predict  # type: ignore[method-assign]
    fake_detector.peak_device_memory_bytes = fail_peak_memory  # type: ignore[method-assign]
    config = RunConfig(video_path, run_dir, (), 0.3, 1, 0, 0)

    with pytest.raises(RuntimeError, match="primary inference failure"):
        run_checkpoint(config, fake_detector)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: primary inference failure"
    assert summary["detector_peak_device_memory_bytes"] is None


@pytest.mark.parametrize("telemetry_failure", ["shutdown", "peaks", "sample"])
def test_primary_inference_error_survives_optional_telemetry_failure(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    telemetry_failure: str,
) -> None:
    run_dir = tmp_path / f"telemetry-{telemetry_failure}"
    unavailable = TelemetrySample.unavailable(1)

    class InvalidSample:
        def to_dict(self) -> dict[str, object]:
            raise RuntimeError("sample conversion failed")

    class FailingTelemetryMonitor:
        @property
        def samples(self) -> tuple[TelemetrySample | InvalidSample, ...]:
            if telemetry_failure == "sample":
                return (InvalidSample(),)
            return (unavailable,)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *unused: object) -> None:
            if telemetry_failure == "shutdown":
                raise RuntimeError("telemetry shutdown failed")

        def peaks(self) -> dict[str, int | float | None]:
            if telemetry_failure == "peaks":
                raise RuntimeError("telemetry peaks failed")
            return {
                "process_rss_bytes": None,
                "system_memory_used_bytes": None,
                "gpu_utilization_percent": None,
                "gpu_memory_used_bytes": None,
                "gpu_power_watts": None,
                "gpu_temperature_c": None,
            }

    def fail_predict(_images: tuple[object, ...]) -> object:
        raise RuntimeError("primary inference failure")

    monkeypatch.setattr(
        runner_module,
        "TelemetryMonitor",
        lambda: FailingTelemetryMonitor(),
    )
    fake_detector.predict = fail_predict  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="primary inference failure"):
        run_checkpoint(RunConfig(video_path, run_dir, (), 0.3, 1, 0, 0), fake_detector)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: primary inference failure"
    assert summary["hardware_peaks"] == {
        "process_rss_bytes": None,
        "system_memory_used_bytes": None,
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }


def test_summary_publication_waits_for_closed_complete_jsonl_streams(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "durable-summary"
    hardware_records: list[dict[str, object]] = []
    active_outputs: RunOutputs | None = None
    summary_published = False
    original_write_hardware = RunOutputs.write_hardware
    original_write_summary = RunOutputs.write_summary
    original_replace = os.replace

    def record_hardware(self: RunOutputs, record: Mapping[str, object]) -> None:
        hardware_records.append(dict(record))
        original_write_hardware(self, record)

    def capture_outputs(self: RunOutputs, summary: Mapping[str, object]) -> None:
        nonlocal active_outputs
        active_outputs = self
        original_write_summary(self, summary)

    def inspect_summary_replace(source: object, destination: object) -> None:
        nonlocal summary_published
        destination_path = Path(destination)  # type: ignore[arg-type]
        if destination_path.name == "summary.json":
            assert active_outputs is not None
            assert not destination_path.exists()
            assert all(stream.closed for stream in active_outputs._streams.values())
            persisted_hardware = _json_lines(run_dir / "hardware.jsonl")
            assert hardware_records
            assert len(persisted_hardware) == len(hardware_records)
            for persisted, expected in zip(persisted_hardware, hardware_records, strict=True):
                assert all(persisted[key] == value for key, value in expected.items())
            summary_published = True
        original_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(RunOutputs, "write_hardware", record_hardware)
    monkeypatch.setattr(RunOutputs, "write_summary", capture_outputs)
    monkeypatch.setattr("edge_perception.outputs.os.replace", inspect_summary_replace)

    run_checkpoint(RunConfig(video_path, run_dir, (), 0.3, 1, 0, 0), fake_detector)

    assert summary_published


def test_run_checkpoint_finalizes_failed_artifacts_when_warmup_fails(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "failed-warmup"

    def fail_warmup(_image: object, _runs: int) -> None:
        raise RuntimeError("warmup failed")

    fake_detector.warmup = fail_warmup  # type: ignore[method-assign]
    config = RunConfig(
        input_path=video_path,
        output_dir=run_dir,
        regions=(Region("right", 100, 20, 80, 60),),
        threshold=0.3,
        max_frames=1,
        warmup_runs=1,
        annotate_every=0,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        run_checkpoint(config, fake_detector)

    assert (run_dir / "manifest.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error"] == "RuntimeError: warmup failed"
    assert summary["frames_processed"] == 0
    assert summary["inference_count"] == 0
    assert _json_lines(run_dir / "inferences.jsonl") == []
    assert _json_lines(run_dir / "detections.jsonl") == []
    assert _json_lines(run_dir / "hardware.jsonl") == []
    video_path.rename(tmp_path / "preview-was-closed.mp4")


def test_runner_cancels_between_completed_frames(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    events: list[ProgressEvent] = []
    config = RunConfig(video_path, tmp_path / "run", (), 0.3, None, 0, 0)

    summary = run_checkpoint(
        config,
        fake_detector,
        progress=events.append,
        cancel_requested=lambda: len(fake_detector.predict_batch_sizes) >= 1,
    )

    persisted = json.loads((config.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "cancelled"
    assert summary["frames_processed"] == 1
    assert persisted["status"] == "cancelled"
    assert len(_json_lines(config.output_dir / "inferences.jsonl")) == 1
    assert events[-1].phase == "cancelled"
    assert sum(event.phase in {"complete", "cancelled", "failed"} for event in events) == 1


def test_runner_emits_one_failed_terminal_event_when_detector_fails(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    events: list[ProgressEvent] = []

    def fail(_images: tuple[object, ...]) -> object:
        raise RuntimeError("detector failed")

    fake_detector.predict = fail  # type: ignore[method-assign]
    config = RunConfig(video_path, tmp_path / "failed-progress", (), 0.3, 1, 0, 0)

    with pytest.raises(RuntimeError, match="detector failed"):
        run_checkpoint(config, fake_detector, progress=events.append)

    assert [event.phase for event in events] == ["validating", "warming_up", "failed"]
    assert events[-1].error == "RuntimeError: detector failed"
    assert sum(event.phase in {"complete", "cancelled", "failed"} for event in events) == 1


def test_run_checkpoint_rejects_nonempty_output_before_detector_work(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    marker = run_dir / "keep-me.txt"
    marker.write_text("user data", encoding="utf-8")
    config = RunConfig(
        input_path=video_path,
        output_dir=run_dir,
        regions=(),
        threshold=0.3,
        max_frames=0,
        warmup_runs=0,
        annotate_every=0,
    )

    with pytest.raises(ValueError, match="output directory must be empty"):
        run_checkpoint(config, fake_detector)

    assert fake_detector.warmup_calls == []
    assert fake_detector.predict_batch_sizes == []
    assert marker.read_text(encoding="utf-8") == "user data"
    assert list(run_dir.iterdir()) == [marker]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_run_config_rejects_threshold_outside_closed_unit_interval(
    threshold: float,
    video_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        RunConfig(video_path, tmp_path / "run", (), threshold, None, 0, 0)
