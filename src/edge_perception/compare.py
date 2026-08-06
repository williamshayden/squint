"""Timing- and hardware-neutral semantic comparison of checkpoint runs."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import cast

type SemanticKey = tuple[str, str, int, int, str | None]


def _json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"run artifact does not exist: {path}") from None
    if not isinstance(value, dict):
        raise TypeError(f"run artifact must contain a JSON object: {path}")
    return cast(dict[str, object], value)


def _json_lines(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"run artifact does not exist: {path}") from None
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"detection row {line_number} must be a JSON object: {path}")
        records.append(cast(dict[str, object], value))
    return records


def _mapping(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"manifest field {field!r} must be an object")
    return cast(dict[str, object], value)


def _manifest_fields(manifest: dict[str, object]) -> tuple[tuple[str, object], ...]:
    configuration = _mapping(manifest, "configuration")
    source_video = _mapping(manifest, "source_video")
    detector = _mapping(manifest, "detector")
    return (
        ("schema_version", manifest.get("schema_version")),
        ("model_id", detector.get("model_id")),
        ("revision", detector.get("revision")),
        ("threshold", configuration.get("threshold")),
        ("source_video_sha256", source_video.get("sha256")),
        ("regions", configuration.get("regions")),
    )


def _semantic_key(record: dict[str, object]) -> SemanticKey:
    frame_id = record.get("frame_id")
    region_id = record.get("region_id")
    detection_index = record.get("detection_index")
    class_id = record.get("class_id")
    label = record.get("label")
    if not isinstance(frame_id, str) or not isinstance(region_id, str):
        raise TypeError("detection frame_id and region_id must be strings")
    if (
        not isinstance(detection_index, int)
        or isinstance(detection_index, bool)
        or not isinstance(class_id, int)
        or isinstance(class_id, bool)
    ):
        raise TypeError("detection_index and class_id must be integers")
    if label is not None and not isinstance(label, str):
        raise ValueError("detection label must be a string or null")
    return (frame_id, region_id, detection_index, class_id, label)


def _sort_key(key: SemanticKey) -> tuple[str, str, int, int, str]:
    return (*key[:4], "\0" if key[4] is None else key[4])


def _indexed(records: list[dict[str, object]]) -> dict[SemanticKey, dict[str, object]]:
    indexed: dict[SemanticKey, dict[str, object]] = {}
    for record in records:
        key = _semantic_key(record)
        if key in indexed:
            raise ValueError(f"duplicate semantic detection key: {key}")
        indexed[key] = record
    return indexed


def _box(record: dict[str, object]) -> tuple[float, float, float, float]:
    value = record.get("box")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("detection box must contain four numbers")
    try:
        box = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as error:
        raise ValueError("detection box must contain four numbers") from error
    if not all(isfinite(coordinate) for coordinate in box):
        raise ValueError("detection box coordinates must be finite")
    return cast(tuple[float, float, float, float], box)


def _score(record: dict[str, object]) -> float:
    value = record.get("score")
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise TypeError("detection score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("detection score must be a finite number") from error
    if not isfinite(score):
        raise ValueError("detection score must be a finite number")
    return score


def _key_json(key: SemanticKey) -> list[str | int | None]:
    return list(key)


def compare_runs(
    left: Path,
    right: Path,
    *,
    box_atol: float = 0.01,
    score_atol: float = 1e-4,
) -> dict[str, object]:
    """Compare repeat-run manifests and semantic detections within tolerances."""

    if (
        not isfinite(box_atol)
        or not isfinite(score_atol)
        or box_atol < 0.0
        or score_atol < 0.0
    ):
        raise ValueError("comparison tolerances must be finite and non-negative")

    left_manifest = _json_object(Path(left) / "manifest.json")
    right_manifest = _json_object(Path(right) / "manifest.json")
    left_records = _json_lines(Path(left) / "detections.jsonl")
    right_records = _json_lines(Path(right) / "detections.jsonl")
    left_index = _indexed(left_records)
    right_index = _indexed(right_records)

    first_mismatch: dict[str, object] | None = None
    mismatch_count = 0
    for (field, left_value), (right_field, right_value) in zip(
        _manifest_fields(left_manifest),
        _manifest_fields(right_manifest),
        strict=True,
    ):
        if field != right_field:
            raise AssertionError("manifest comparison fields are inconsistent")
        if left_value != right_value:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "kind": "manifest",
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                }

    shared_keys = set(left_index) & set(right_index)
    unmatched_keys = set(left_index) ^ set(right_index)
    mismatch_count += len(unmatched_keys)
    if first_mismatch is None and unmatched_keys:
        key = min(unmatched_keys, key=_sort_key)
        first_mismatch = {
            "kind": "semantic_key",
            "key": _key_json(key),
            "left_present": key in left_index,
            "right_present": key in right_index,
        }

    matched_detection_count = 0
    for key in sorted(shared_keys, key=_sort_key):
        left_record = left_index[key]
        right_record = right_index[key]
        left_box = _box(left_record)
        right_box = _box(right_record)
        differing_coordinates = [
            index
            for index, (left_value, right_value) in enumerate(zip(left_box, right_box, strict=True))
            if abs(left_value - right_value) > box_atol
        ]
        left_score = _score(left_record)
        right_score = _score(right_record)
        score_differs = abs(left_score - right_score) > score_atol
        if differing_coordinates or score_differs:
            mismatch_count += 1
            if first_mismatch is None:
                if differing_coordinates:
                    first_mismatch = {
                        "kind": "box",
                        "key": _key_json(key),
                        "left": list(left_box),
                        "right": list(right_box),
                        "box_atol": box_atol,
                    }
                else:
                    first_mismatch = {
                        "kind": "score",
                        "key": _key_json(key),
                        "left": left_score,
                        "right": right_score,
                        "score_atol": score_atol,
                    }
        else:
            matched_detection_count += 1

    return {
        "equivalent": mismatch_count == 0,
        "box_atol": float(box_atol),
        "score_atol": float(score_atol),
        "left_detection_count": len(left_records),
        "right_detection_count": len(right_records),
        "matched_detection_count": matched_detection_count,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
    }
