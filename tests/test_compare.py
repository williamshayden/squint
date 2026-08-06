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
            "regions": [{"region_id": "right", "x": 100, "y": 20, "width": 80, "height": 60}],
        },
        "source_video": {"sha256": "b" * 64},
        "detector": {
            "model_id": "tests/fake-detector",
            "revision": "revision-1",
            "device": "cpu",
        },
        "host": {"os": "ignored"},
        "timing_definitions": {"detector": "ignored"},
    }


def _detections(run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "frame_id": "frame-000000",
            "inference_id": "ignored-between-runs",
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
            "inference_id": "also-ignored",
            "region_id": "right",
            "detection_index": 0,
            "class_id": 1,
            "label": "object",
            "box": [110.0, 25.0, 120.0, 35.0],
            "score": 0.8,
            "timing_ms": 2.0,
        },
    ]


def _write_run(path: Path, manifest: dict[str, Any], detections: list[dict[str, Any]]) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "detections.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in detections),
        encoding="utf-8",
    )


def test_compare_runs_ignores_run_timing_and_hardware_fields_within_tolerance(tmp_path: Path) -> None:
    left_manifest = _manifest("left-run")
    right_manifest = _manifest("right-run")
    right_manifest["detector"]["device"] = "cuda"
    right_manifest["host"] = {"os": "different and ignored"}
    left_detections = _detections("left-run")
    right_detections = list(reversed(_detections("right-run")))
    right_detections[0]["box"][0] += 0.009
    right_detections[1]["score"] += 0.00009
    _write_run(tmp_path / "left", left_manifest, left_detections)
    _write_run(tmp_path / "right", right_manifest, right_detections)

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


def test_compare_runs_reports_earliest_mismatch_in_sorted_semantic_union(tmp_path: Path) -> None:
    left_detections = _detections("left-run")
    right_detections = deepcopy(_detections("right-run"))
    right_detections[0]["box"][0] += 1.0
    right_detections.pop()
    _write_run(tmp_path / "left", _manifest("left-run"), left_detections)
    _write_run(tmp_path / "right", _manifest("right-run"), right_detections)

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
    _write_run(tmp_path / "left", left_manifest, left_detections)
    _write_run(tmp_path / "right", right_manifest, right_detections)

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
    _write_run(tmp_path / "left", left_manifest, _detections("left-run"))
    _write_run(tmp_path / "right", right_manifest, _detections("right-run"))

    with pytest.raises(ValueError, match=message):
        compare_runs(tmp_path / "left", tmp_path / "right")


def test_compare_runs_reports_non_object_manifest_as_controlled_error(tmp_path: Path) -> None:
    _write_run(tmp_path / "left", _manifest("left-run"), _detections("left-run"))
    _write_run(tmp_path / "right", _manifest("right-run"), _detections("right-run"))
    (tmp_path / "left" / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json"):
        compare_runs(tmp_path / "left", tmp_path / "right")
