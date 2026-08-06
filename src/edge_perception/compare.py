"""Timing- and hardware-neutral semantic comparison of checkpoint runs."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from re import fullmatch
from typing import cast

type SemanticKey = tuple[str, str, int, int, str | None]


class ComparisonError(ValueError):
    """A controlled validation failure for malformed comparison artifacts."""


def _json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"run artifact does not exist: {path}") from None
    if not isinstance(value, dict):
        raise ComparisonError(f"run artifact must contain a JSON object: {path}")
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
    if field not in record:
        raise ValueError(f"manifest is missing required field {field}")
    value = record[field]
    if not isinstance(value, dict):
        raise ComparisonError(f"manifest field {field!r} must be an object")
    return cast(dict[str, object], value)


def _required_string(record: dict[str, object], field: str, context: str) -> str:
    if field not in record:
        raise ValueError(f"{context} is missing required field {field}")
    value = record[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value


def _required_integer(record: dict[str, object], field: str, context: str) -> int:
    if field not in record:
        raise ValueError(f"{context} is missing required field {field}")
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComparisonError(f"{context}.{field} must be an integer")
    return value


def _regions(configuration: dict[str, object]) -> list[dict[str, str | int]]:
    if "regions" not in configuration:
        raise ValueError("configuration is missing required field regions")
    value = configuration["regions"]
    if not isinstance(value, list):
        raise ComparisonError("configuration.regions must be a list")
    normalized: list[dict[str, str | int]] = []
    region_ids: set[str] = set()
    required_fields = {"region_id", "x", "y", "width", "height"}
    for index, item in enumerate(value):
        context = f"configuration.regions[{index}]"
        if not isinstance(item, dict):
            raise ComparisonError(f"{context} must be an object")
        region = cast(dict[str, object], item)
        if set(region) != required_fields:
            raise ValueError(f"{context} must contain exactly {sorted(required_fields)}")
        region_id = _required_string(region, "region_id", context)
        x = _required_integer(region, "x", context)
        y = _required_integer(region, "y", context)
        width = _required_integer(region, "width", context)
        height = _required_integer(region, "height", context)
        if x < 0 or y < 0:
            raise ValueError(f"{context} origin must not be negative")
        if width <= 0 or height <= 0:
            raise ValueError(f"{context} width and height must be positive")
        if region_id in region_ids:
            raise ValueError(f"configuration.regions has duplicate region_id {region_id!r}")
        region_ids.add(region_id)
        normalized.append(
            {"region_id": region_id, "x": x, "y": y, "width": width, "height": height}
        )
    return normalized


def _manifest_fields(manifest: dict[str, object]) -> tuple[tuple[str, object], ...]:
    configuration = _mapping(manifest, "configuration")
    source_video = _mapping(manifest, "source_video")
    detector = _mapping(manifest, "detector")
    schema_version = _required_string(manifest, "schema_version", "manifest")
    model_id = _required_string(detector, "model_id", "detector")
    revision = _required_string(detector, "revision", "detector")

    if "threshold" not in configuration:
        raise ValueError("configuration is missing required field threshold")
    raw_threshold = configuration["threshold"]
    if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
        raise ComparisonError("configuration.threshold must be a number")
    threshold = float(raw_threshold)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("configuration.threshold must be between 0 and 1")

    source_sha256 = _required_string(source_video, "sha256", "source_video")
    if fullmatch(r"[0-9a-fA-F]{64}", source_sha256) is None:
        raise ValueError("source_video.sha256 must be 64 hexadecimal characters")
    return (
        ("schema_version", schema_version),
        ("model_id", model_id),
        ("revision", revision),
        ("threshold", threshold),
        ("source_video_sha256", source_sha256),
        ("regions", _regions(configuration)),
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

    matched_detection_count = 0
    all_keys = sorted(set(left_index) | set(right_index), key=_sort_key)
    for key in all_keys:
        left_present = key in left_index
        right_present = key in right_index
        if not left_present or not right_present:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "kind": "semantic_key",
                    "key": _key_json(key),
                    "left_present": left_present,
                    "right_present": right_present,
                }
            continue
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
