from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from edge_perception.compare import compare_runs


def _manifest(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "configuration": {
            "threshold": 0.3,
            "regions": [
                {"region_id": "right", "x": 100, "y": 20, "width": 80, "height": 60},
                {"region_id": "empty", "x": 0, "y": 0, "width": 50, "height": 50},
            ],
        },
        "source_video": {"sha256": "b" * 64},
        "detector": {
            "adapter": "tests.fake",
            "model_id": "tests/fake-detector",
            "revision": "revision-1",
            "weights_sha256": "a" * 64,
            "backend": "fake",
            "backend_version": "1.0",
            "device": "cpu",
            "dtype": "float32",
        },
        "host": {"os": "ignored"},
        "timing_definitions": {"detector": "ignored"},
    }


def _summary(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "status": "complete",
        "frames_processed": 1,
        "inference_count": 3,
        "annotated_frame_count": 1,
        "latency_ms": {"complete_frame": {"count": 1, "p50_ms": 8.0}},
        "stage_latency_ms": {"detector": {"count": 3, "p50_ms": 1.0}},
        "hardware_peaks": {"process_rss_bytes": 1024},
        "detector_peak_device_memory_bytes": None,
    }


def _inferences(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_index": 0,
            "frame_id": "frame-000000",
            "source_time_ms": 0.0,
            "inference_id": "inference-000000-000",
            "region_id": "full-frame",
            "region": {
                "region_id": "full-frame",
                "x": 0,
                "y": 0,
                "width": 200,
                "height": 100,
            },
            "input_shape": [100, 200, 3],
            "frame_decode_ms": 0.5,
            "crop_ms": 0.0,
            "coordinate_mapping_ms": 0.1,
            "region_pipeline_ms": 1.0,
            "detector_timing": {"total_ms": 0.8},
        },
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_index": 0,
            "frame_id": "frame-000000",
            "source_time_ms": 0.0,
            "inference_id": "inference-000000-001",
            "region_id": "right",
            "region": {"region_id": "right", "x": 100, "y": 20, "width": 80, "height": 60},
            "input_shape": [60, 80, 3],
            "frame_decode_ms": 0.5,
            "crop_ms": 0.2,
            "coordinate_mapping_ms": 0.1,
            "region_pipeline_ms": 1.2,
            "detector_timing": {"total_ms": 0.9},
        },
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_index": 0,
            "frame_id": "frame-000000",
            "source_time_ms": 0.0,
            "inference_id": "inference-000000-002",
            "region_id": "empty",
            "region": {"region_id": "empty", "x": 0, "y": 0, "width": 50, "height": 50},
            "input_shape": [50, 50, 3],
            "frame_decode_ms": 0.5,
            "crop_ms": 0.2,
            "coordinate_mapping_ms": 0.1,
            "region_pipeline_ms": 1.1,
            "detector_timing": {"total_ms": 0.8},
        },
    ]


def _detections(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_id": "frame-000000",
            "inference_id": "inference-000000-000",
            "region_id": "full-frame",
            "detection_index": 0,
            "class_id": 1,
            "label": "object",
            "box": [10.0, 5.0, 20.0, 15.0],
            "score": 0.75,
            "timing_ms": 1.0,
        },
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_id": "frame-000000",
            "inference_id": "inference-000000-001",
            "region_id": "right",
            "detection_index": 0,
            "class_id": 1,
            "label": "object",
            "box": [110.0, 25.0, 120.0, 35.0],
            "score": 0.8,
            "timing_ms": 2.0,
        },
    ]


def _write_run(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    inferences: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "inferences.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in inferences),
        encoding="utf-8",
    )
    (path / "detections.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in detections),
        encoding="utf-8",
    )


def test_compare_runs_ignores_run_timing_and_hardware_fields_within_tolerance(tmp_path: Path) -> None:
    left_manifest = _manifest("left-run")
    right_manifest = _manifest("right-run")
    right_manifest["detector"]["backend"] = "different-and-ignored"
    right_manifest["detector"]["backend_version"] = "99.0"
    right_manifest["detector"]["device"] = "cuda"
    right_manifest["detector"]["dtype"] = "float16"
    right_manifest["host"] = {"os": "different and ignored"}
    left_summary = _summary("left-run")
    right_summary = _summary("right-run")
    right_summary["latency_ms"] = {"complete_frame": {"count": 1, "p50_ms": 800.0}}
    right_summary["stage_latency_ms"] = {"detector": {"count": 3, "p50_ms": 100.0}}
    right_summary["hardware_peaks"] = {"process_rss_bytes": 999999}
    right_summary["detector_peak_device_memory_bytes"] = 123456
    left_inferences = _inferences("left-run")
    right_inferences = _inferences("right-run")
    right_inferences[0]["inference_id"] = "different-full-frame-link"
    right_inferences[0]["frame_decode_ms"] = 50.0
    right_inferences[0]["detector_timing"] = {"total_ms": 80.0}
    right_inferences[1]["inference_id"] = "different-right-link"
    right_inferences[1]["crop_ms"] = 20.0
    right_inferences[1]["region_pipeline_ms"] = 120.0
    right_inferences[2]["inference_id"] = "different-empty-link"
    right_inferences[2]["coordinate_mapping_ms"] = 10.0
    left_detections = _detections("left-run")
    right_detections = list(reversed(_detections("right-run")))
    right_detections[0]["inference_id"] = "different-right-link"
    right_detections[1]["inference_id"] = "different-full-frame-link"
    right_detections[0]["box"][0] += 0.009
    right_detections[1]["score"] += 0.00009
    _write_run(tmp_path / "left", left_manifest, left_summary, left_inferences, left_detections)
    _write_run(
        tmp_path / "right",
        right_manifest,
        right_summary,
        list(reversed(right_inferences)),
        right_detections,
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report == {
        "equivalent": True,
        "box_atol": 0.01,
        "score_atol": 0.0001,
        "left_detection_count": 2,
        "right_detection_count": 2,
        "matched_detection_count": 2,
        "mismatch_count": 0,
        "first_mismatch": None,
    }
    json.dumps(report, allow_nan=False)


def test_compare_runs_reports_terminal_status_mismatch(tmp_path: Path) -> None:
    left_summary = _summary("left-run")
    right_summary = _summary("right-run")
    right_summary["status"] = "cancelled"
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        left_summary,
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        right_summary,
        _inferences("right-run"),
        _detections("right-run"),
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["equivalent"] is False
    assert report["mismatch_count"] == 1
    assert report["first_mismatch"] == {
        "kind": "summary",
        "field": "status",
        "left": "complete",
        "right": "cancelled",
    }


@pytest.mark.parametrize(
    ("field", "left_value", "right_value"),
    [("frames_processed", 1, 2), ("inference_count", 3, 4)],
)
def test_compare_runs_reports_terminal_count_mismatch(
    field: str,
    left_value: int,
    right_value: int,
    tmp_path: Path,
) -> None:
    left_summary = _summary("left-run")
    right_summary = _summary("right-run")
    right_summary[field] = right_value
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        left_summary,
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        right_summary,
        _inferences("right-run"),
        _detections("right-run"),
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["equivalent"] is False
    assert report["mismatch_count"] == 1
    assert report["first_mismatch"] == {
        "kind": "summary",
        "field": field,
        "left": left_value,
        "right": right_value,
    }


def test_compare_runs_reports_missing_zero_detection_inference(tmp_path: Path) -> None:
    right_inferences = _inferences("right-run")
    right_inferences.pop()
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        _summary("left-run"),
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        _summary("right-run"),
        right_inferences,
        _detections("right-run"),
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["equivalent"] is False
    assert report["mismatch_count"] == 1
    assert report["first_mismatch"] == {
        "kind": "inference_schedule",
        "key": [
            0,
            "frame-000000",
            "empty",
            {"region_id": "empty", "x": 0, "y": 0, "width": 50, "height": 50},
            [50, 50, 3],
            0.0,
        ],
        "left_present": True,
        "right_present": False,
    }


def test_compare_runs_reports_detector_weights_mismatch(tmp_path: Path) -> None:
    right_manifest = _manifest("right-run")
    right_manifest["detector"]["weights_sha256"] = "c" * 64
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        _summary("left-run"),
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        right_manifest,
        _summary("right-run"),
        _inferences("right-run"),
        _detections("right-run"),
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["mismatch_count"] == 1
    assert report["first_mismatch"] == {
        "kind": "manifest",
        "field": "weights_sha256",
        "left": "a" * 64,
        "right": "c" * 64,
    }


def test_compare_runs_reports_detector_adapter_mismatch(tmp_path: Path) -> None:
    right_manifest = _manifest("right-run")
    right_manifest["detector"]["adapter"] = "tests.other"
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        _summary("left-run"),
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        right_manifest,
        _summary("right-run"),
        _inferences("right-run"),
        _detections("right-run"),
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["mismatch_count"] == 1
    assert report["first_mismatch"] == {
        "kind": "manifest",
        "field": "adapter",
        "left": "tests.fake",
        "right": "tests.other",
    }


def test_compare_runs_reports_earliest_mismatch_in_sorted_semantic_union(tmp_path: Path) -> None:
    left_detections = _detections("left-run")
    right_detections = deepcopy(_detections("right-run"))
    right_detections[0]["box"][0] += 1.0
    right_detections.pop()
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        _summary("left-run"),
        _inferences("left-run"),
        left_detections,
    )
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        _summary("right-run"),
        _inferences("right-run"),
        right_detections,
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["mismatch_count"] == 2
    assert report["first_mismatch"] == {
        "kind": "box",
        "key": ["frame-000000", "full-frame", 0, 1, "object"],
        "left": [10.0, 5.0, 20.0, 15.0],
        "right": [11.0, 5.0, 20.0, 15.0],
        "box_atol": 0.01,
    }


@pytest.mark.parametrize(
    ("mutation", "mismatch_kind"),
    [
        ("missing", "semantic_key"),
        ("class", "semantic_key"),
        ("box", "box"),
        ("score", "score"),
        ("revision", "manifest"),
    ],
)
def test_compare_runs_reports_first_semantic_or_manifest_mismatch(
    mutation: str,
    mismatch_kind: str,
    tmp_path: Path,
) -> None:
    left_manifest = _manifest("left-run")
    right_manifest = _manifest("right-run")
    left_detections = _detections("left-run")
    right_detections = deepcopy(_detections("right-run"))
    if mutation == "missing":
        right_detections.pop()
    elif mutation == "class":
        right_detections[0]["class_id"] = 2
    elif mutation == "box":
        right_detections[0]["box"][0] += 0.011
    elif mutation == "score":
        right_detections[0]["score"] += 0.00011
    elif mutation == "revision":
        right_manifest["detector"]["revision"] = "revision-2"
    _write_run(
        tmp_path / "left",
        left_manifest,
        _summary("left-run"),
        _inferences("left-run"),
        left_detections,
    )
    _write_run(
        tmp_path / "right",
        right_manifest,
        _summary("right-run"),
        _inferences("right-run"),
        right_detections,
    )

    report = compare_runs(tmp_path / "left", tmp_path / "right")

    assert report["equivalent"] is False
    assert report["mismatch_count"] >= 1
    assert report["first_mismatch"]["kind"] == mismatch_kind


@pytest.mark.parametrize(("box_atol", "score_atol"), [(-1.0, 0.1), (0.1, -1.0)])
def test_compare_runs_rejects_negative_tolerances(
    box_atol: float,
    score_atol: float,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        compare_runs(tmp_path / "left", tmp_path / "right", box_atol=box_atol, score_atol=score_atol)


@pytest.mark.parametrize(
    ("malformed_field", "message"),
    [
        ("schema_version", "schema_version"),
        ("model_id", "model_id"),
        ("revision", "revision"),
        ("threshold", "threshold"),
        ("source_sha256", "sha256"),
        ("regions", "regions"),
    ],
)
def test_compare_runs_rejects_malformed_required_manifest_invariants(
    malformed_field: str,
    message: str,
    tmp_path: Path,
) -> None:
    left_manifest = _manifest("left-run")
    right_manifest = _manifest("right-run")
    for manifest in (left_manifest, right_manifest):
        if malformed_field == "schema_version":
            manifest.pop("schema_version")
        elif malformed_field == "model_id":
            manifest["detector"]["model_id"] = 123
        elif malformed_field == "revision":
            manifest["detector"].pop("revision")
        elif malformed_field == "threshold":
            manifest["configuration"]["threshold"] = "0.3"
        elif malformed_field == "source_sha256":
            manifest["source_video"]["sha256"] = "not-a-sha256"
        elif malformed_field == "regions":
            manifest["configuration"]["regions"] = [
                {"region_id": "right", "x": 100, "y": 20, "width": 80}
            ]
    _write_run(
        tmp_path / "left",
        left_manifest,
        _summary("left-run"),
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        right_manifest,
        _summary("right-run"),
        _inferences("right-run"),
        _detections("right-run"),
    )

    with pytest.raises(ValueError, match=message):
        compare_runs(tmp_path / "left", tmp_path / "right")


def test_compare_runs_reports_non_object_manifest_as_controlled_error(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "left",
        _manifest("left-run"),
        _summary("left-run"),
        _inferences("left-run"),
        _detections("left-run"),
    )
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        _summary("right-run"),
        _inferences("right-run"),
        _detections("right-run"),
    )
    (tmp_path / "left" / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json"):
        compare_runs(tmp_path / "left", tmp_path / "right")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest_schema", "manifest.json.schema_version"),
        ("summary_run_id", "summary.json.run_id"),
        ("inference_schema", "inferences.jsonl row 1.schema_version"),
        ("detection_run_id", "detections.jsonl row 1.run_id"),
    ],
)
def test_compare_runs_rejects_cross_artifact_schema_or_run_id_drift(
    mutation: str,
    message: str,
    tmp_path: Path,
) -> None:
    manifest = _manifest("left-run")
    summary = _summary("left-run")
    inferences = _inferences("left-run")
    detections = _detections("left-run")
    if mutation == "manifest_schema":
        manifest["schema_version"] = "0.2.0"
    elif mutation == "summary_run_id":
        summary["run_id"] = "another-run"
    elif mutation == "inference_schema":
        inferences[0]["schema_version"] = "0.2.0"
    elif mutation == "detection_run_id":
        detections[0]["run_id"] = "another-run"
    _write_run(tmp_path / "left", manifest, summary, inferences, detections)
    _write_run(
        tmp_path / "right",
        _manifest("right-run"),
        _summary("right-run"),
        _inferences("right-run"),
        _detections("right-run"),
    )

    with pytest.raises(ValueError, match=message):
        compare_runs(tmp_path / "left", tmp_path / "right")
