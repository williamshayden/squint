from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import FakeDetector

from edge_perception.contracts import Region
from edge_perception.runner import RunConfig, run_checkpoint


def _json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_checkpoint_processes_full_frame_then_declared_crops(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "run"
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


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_run_config_rejects_threshold_outside_closed_unit_interval(
    threshold: float,
    video_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        RunConfig(video_path, tmp_path / "run", (), threshold, None, 0, 0)
