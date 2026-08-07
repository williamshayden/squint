"""Timing- and hardware-neutral semantic comparison of checkpoint runs."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from re import fullmatch
from typing import cast

SCHEMA_VERSION = "0.1.0"

type SemanticKey = tuple[str, str, int, int, str | None]
type RegionKey = tuple[str, int, int, int, int]
type InputShape = tuple[int, int, int]
type InferenceKey = tuple[int, str, str, RegionKey, InputShape, float | None]


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
            raise ComparisonError(
                f"{path.name} row {line_number} must be a JSON object: {path}"
            )
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


def _artifact_identity(
    record: dict[str, object],
    context: str,
    *,
    expected_run_id: str | None = None,
) -> str:
    schema_version = _required_string(record, "schema_version", context)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{context}.schema_version must be {SCHEMA_VERSION}")
    run_id = _required_string(record, "run_id", context)
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"{context}.run_id must match manifest.json.run_id")
    return run_id


def _region_key(record: dict[str, object], context: str) -> RegionKey:
    required_fields = {"region_id", "x", "y", "width", "height"}
    if set(record) != required_fields:
        raise ValueError(f"{context} must contain exactly {sorted(required_fields)}")
    region_id = _required_string(record, "region_id", context)
    x = _required_integer(record, "x", context)
    y = _required_integer(record, "y", context)
    width = _required_integer(record, "width", context)
    height = _required_integer(record, "height", context)
    if x < 0 or y < 0:
        raise ValueError(f"{context} origin must not be negative")
    if width <= 0 or height <= 0:
        raise ValueError(f"{context} width and height must be positive")
    return (region_id, x, y, width, height)


def _region_json(region: RegionKey) -> dict[str, str | int]:
    region_id, x, y, width, height = region
    return {"region_id": region_id, "x": x, "y": y, "width": width, "height": height}


def _region_keys(configuration: dict[str, object]) -> tuple[RegionKey, ...]:
    if "regions" not in configuration:
        raise ValueError("configuration is missing required field regions")
    value = configuration["regions"]
    if not isinstance(value, list):
        raise ComparisonError("configuration.regions must be a list")
    normalized: list[RegionKey] = []
    region_ids: set[str] = set()
    for index, item in enumerate(value):
        context = f"configuration.regions[{index}]"
        if not isinstance(item, dict):
            raise ComparisonError(f"{context} must be an object")
        region = _region_key(cast(dict[str, object], item), context)
        region_id = region[0]
        if region_id in region_ids:
            raise ValueError(f"configuration.regions has duplicate region_id {region_id!r}")
        region_ids.add(region_id)
        normalized.append(region)
    return tuple(normalized)


def _regions(configuration: dict[str, object]) -> list[dict[str, str | int]]:
    return [_region_json(region) for region in _region_keys(configuration)]


def _region_catalog(manifest: dict[str, object]) -> dict[str, RegionKey]:
    configuration = _mapping(manifest, "configuration")
    return {region[0]: region for region in _region_keys(configuration)}


def _manifest_fields(manifest: dict[str, object]) -> tuple[tuple[str, object], ...]:
    configuration = _mapping(manifest, "configuration")
    source_video = _mapping(manifest, "source_video")
    detector = _mapping(manifest, "detector")
    schema_version = _required_string(manifest, "schema_version", "manifest")
    adapter = _required_string(detector, "adapter", "detector")
    model_id = _required_string(detector, "model_id", "detector")
    revision = _required_string(detector, "revision", "detector")
    weights_sha256 = _required_string(detector, "weights_sha256", "detector")
    if fullmatch(r"[0-9a-fA-F]{64}", weights_sha256) is None:
        raise ValueError("detector.weights_sha256 must be 64 hexadecimal characters")

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
        ("adapter", adapter),
        ("weights_sha256", weights_sha256),
    )


def _summary_fields(summary: dict[str, object]) -> tuple[tuple[str, object], ...]:
    status = _required_string(summary, "status", "summary.json")
    if status not in {"complete", "failed", "cancelled"}:
        raise ValueError("summary.json.status must be complete, failed, or cancelled")
    frames_processed = _required_integer(summary, "frames_processed", "summary.json")
    inference_count = _required_integer(summary, "inference_count", "summary.json")
    if frames_processed < 0:
        raise ValueError("summary.json.frames_processed must not be negative")
    if inference_count < 0:
        raise ValueError("summary.json.inference_count must not be negative")
    return (
        ("status", status),
        ("frames_processed", frames_processed),
        ("inference_count", inference_count),
    )


def _input_shape(record: dict[str, object], context: str) -> InputShape:
    if "input_shape" not in record:
        raise ValueError(f"{context} is missing required field input_shape")
    value = record["input_shape"]
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ComparisonError(f"{context}.input_shape must contain three integers")
    shape = cast(list[int], value)
    if any(item <= 0 for item in shape) or shape[2] != 3:
        raise ValueError(f"{context}.input_shape must contain positive RGB dimensions")
    return (shape[0], shape[1], shape[2])


def _source_time_ms(record: dict[str, object], context: str) -> float | None:
    if "source_time_ms" not in record:
        raise ValueError(f"{context} is missing required field source_time_ms")
    value = record["source_time_ms"]
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ComparisonError(f"{context}.source_time_ms must be a number or null")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{context}.source_time_ms must be finite")
    return normalized


def _inference_key(record: dict[str, object], context: str) -> InferenceKey:
    frame_index = _required_integer(record, "frame_index", context)
    if frame_index < 0:
        raise ValueError(f"{context}.frame_index must not be negative")
    frame_id = _required_string(record, "frame_id", context)
    region_id = _required_string(record, "region_id", context)
    if "region" not in record:
        raise ValueError(f"{context} is missing required field region")
    raw_region = record["region"]
    if not isinstance(raw_region, dict):
        raise ComparisonError(f"{context}.region must be an object")
    region = _region_key(cast(dict[str, object], raw_region), f"{context}.region")
    if region_id != region[0]:
        raise ValueError(f"{context}.region_id must match {context}.region.region_id")
    return (
        frame_index,
        frame_id,
        region_id,
        region,
        _input_shape(record, context),
        _source_time_ms(record, context),
    )


def _inference_sort_key(
    key: InferenceKey,
) -> tuple[int, str, str, RegionKey, InputShape, tuple[int, float]]:
    source_time = key[5]
    normalized_time = (0, 0.0) if source_time is None else (1, source_time)
    return (*key[:5], normalized_time)


def _inference_key_json(key: InferenceKey) -> list[object]:
    return [key[0], key[1], key[2], _region_json(key[3]), list(key[4]), key[5]]


def _inference_index(
    records: list[dict[str, object]],
    *,
    run_id: str,
    summary: dict[str, object],
    region_catalog: dict[str, RegionKey],
) -> tuple[dict[InferenceKey, dict[str, object]], dict[str, tuple[str, str]]]:
    indexed: dict[InferenceKey, dict[str, object]] = {}
    links: dict[str, tuple[str, str]] = {}
    logical_regions: set[tuple[int, str]] = set()
    frame_groups: dict[int, tuple[str, float | None]] = {}
    frame_ids: dict[str, int] = {}
    for row_number, record in enumerate(records, start=1):
        context = f"inferences.jsonl row {row_number}"
        _artifact_identity(record, context, expected_run_id=run_id)
        inference_id = _required_string(record, "inference_id", context)
        if inference_id in links:
            raise ValueError(f"duplicate inference_id: {inference_id!r}")
        key = _inference_key(record, context)
        frame_index, frame_id, region_id, region, input_shape, source_time_ms = key
        logical_region = (frame_index, region_id)
        if logical_region in logical_regions:
            raise ComparisonError(
                f"{context} has duplicate logical frame-region {logical_region!r}"
            )
        logical_regions.add(logical_region)

        if frame_index in frame_groups:
            expected_frame_id, expected_source_time_ms = frame_groups[frame_index]
            if frame_id != expected_frame_id:
                raise ComparisonError(
                    f"inferences.jsonl frame_index {frame_index} must use one frame_id"
                )
            if source_time_ms != expected_source_time_ms:
                raise ComparisonError(
                    f"inferences.jsonl frame_index {frame_index} must use one source_time_ms"
                )
        else:
            frame_groups[frame_index] = (frame_id, source_time_ms)

        if frame_id in frame_ids and frame_ids[frame_id] != frame_index:
            raise ComparisonError(
                f"inferences.jsonl frame_id {frame_id!r} must not reference multiple frame indices"
            )
        frame_ids[frame_id] = frame_index

        if region_id != "full-frame" and region_catalog.get(region_id) != region:
            raise ComparisonError(
                f"{context}.region must match a member of configuration.regions"
            )
        expected_input_shape = (region[4], region[3], 3)
        if input_shape != expected_input_shape:
            raise ComparisonError(
                f"{context}.input_shape must match region height, width, and RGB channels"
            )
        if key in indexed:
            raise ValueError(f"duplicate inference schedule key: {_inference_key_json(key)}")
        indexed[key] = record
        links[inference_id] = (key[1], key[2])

    inference_count = _required_integer(summary, "inference_count", "summary.json")
    if inference_count != len(records):
        raise ComparisonError(
            "summary.json.inference_count must equal the number of inferences.jsonl rows"
        )
    frames_processed = _required_integer(summary, "frames_processed", "summary.json")
    if frames_processed != len(frame_groups):
        raise ComparisonError(
            "summary.json.frames_processed must equal the number of distinct inference frame groups"
        )
    return indexed, links


def _semantic_key(record: dict[str, object]) -> SemanticKey:
    frame_id = record.get("frame_id")
    region_id = record.get("region_id")
    detection_index = record.get("detection_index")
    class_id = record.get("class_id")
    label = record.get("label")
    if not isinstance(frame_id, str):
        raise ComparisonError("detection frame_id must be a string")
    if not isinstance(region_id, str):
        raise ComparisonError("detection region_id must be a string")
    if not isinstance(detection_index, int) or isinstance(detection_index, bool):
        raise ComparisonError("detection_index must be an integer")
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        raise ComparisonError("detection class_id must be an integer")
    if label is not None and not isinstance(label, str):
        raise ComparisonError("detection label must be a string or null")
    return (frame_id, region_id, detection_index, class_id, label)


def _sort_key(key: SemanticKey) -> tuple[str, str, int, int, str]:
    return (*key[:4], "\0" if key[4] is None else key[4])


def _indexed(
    records: list[dict[str, object]],
    *,
    run_id: str,
    inference_links: dict[str, tuple[str, str]],
) -> dict[SemanticKey, dict[str, object]]:
    indexed: dict[SemanticKey, dict[str, object]] = {}
    for row_number, record in enumerate(records, start=1):
        context = f"detections.jsonl row {row_number}"
        _artifact_identity(record, context, expected_run_id=run_id)
        inference_id = _required_string(record, "inference_id", context)
        if inference_id not in inference_links:
            raise ValueError(f"{context}.inference_id must reference inferences.jsonl")
        key = _semantic_key(record)
        if (key[0], key[1]) != inference_links[inference_id]:
            raise ValueError(
                f"{context} frame_id and region_id must match its inferences.jsonl row"
            )
        if key in indexed:
            raise ValueError(f"duplicate semantic detection key: {key}")
        _box(record)
        _score(record)
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
        raise ComparisonError("detection score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ComparisonError("detection score must be a finite number") from error
    if not isfinite(score):
        raise ComparisonError("detection score must be a finite number")
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

    left_path = Path(left)
    right_path = Path(right)
    left_manifest = _json_object(left_path / "manifest.json")
    right_manifest = _json_object(right_path / "manifest.json")
    left_run_id = _artifact_identity(left_manifest, "manifest.json")
    right_run_id = _artifact_identity(right_manifest, "manifest.json")
    left_manifest_fields = _manifest_fields(left_manifest)
    right_manifest_fields = _manifest_fields(right_manifest)
    left_region_catalog = _region_catalog(left_manifest)
    right_region_catalog = _region_catalog(right_manifest)

    left_summary = _json_object(left_path / "summary.json")
    right_summary = _json_object(right_path / "summary.json")
    _artifact_identity(left_summary, "summary.json", expected_run_id=left_run_id)
    _artifact_identity(right_summary, "summary.json", expected_run_id=right_run_id)
    left_summary_fields = _summary_fields(left_summary)
    right_summary_fields = _summary_fields(right_summary)

    left_inference_records = _json_lines(left_path / "inferences.jsonl")
    right_inference_records = _json_lines(right_path / "inferences.jsonl")
    left_inference_index, left_inference_links = _inference_index(
        left_inference_records,
        run_id=left_run_id,
        summary=left_summary,
        region_catalog=left_region_catalog,
    )
    right_inference_index, right_inference_links = _inference_index(
        right_inference_records,
        run_id=right_run_id,
        summary=right_summary,
        region_catalog=right_region_catalog,
    )

    left_records = _json_lines(left_path / "detections.jsonl")
    right_records = _json_lines(right_path / "detections.jsonl")
    left_index = _indexed(
        left_records,
        run_id=left_run_id,
        inference_links=left_inference_links,
    )
    right_index = _indexed(
        right_records,
        run_id=right_run_id,
        inference_links=right_inference_links,
    )

    first_mismatch: dict[str, object] | None = None
    mismatch_count = 0
    for (field, left_value), (right_field, right_value) in zip(
        left_manifest_fields,
        right_manifest_fields,
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

    for (field, left_value), (right_field, right_value) in zip(
        left_summary_fields,
        right_summary_fields,
        strict=True,
    ):
        if field != right_field:
            raise AssertionError("summary comparison fields are inconsistent")
        if left_value != right_value:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "kind": "summary",
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                }

    all_inference_keys = sorted(
        set(left_inference_index) | set(right_inference_index),
        key=_inference_sort_key,
    )
    for inference_key in all_inference_keys:
        left_present = inference_key in left_inference_index
        right_present = inference_key in right_inference_index
        if not left_present or not right_present:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "kind": "inference_schedule",
                    "key": _inference_key_json(inference_key),
                    "left_present": left_present,
                    "right_present": right_present,
                }

    matched_detection_count = 0
    all_keys = sorted(set(left_index) | set(right_index), key=_sort_key)
    for detection_key in all_keys:
        left_present = detection_key in left_index
        right_present = detection_key in right_index
        if not left_present or not right_present:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "kind": "semantic_key",
                    "key": _key_json(detection_key),
                    "left_present": left_present,
                    "right_present": right_present,
                }
            continue
        left_record = left_index[detection_key]
        right_record = right_index[detection_key]
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
                        "key": _key_json(detection_key),
                        "left": list(left_box),
                        "right": list(right_box),
                        "box_atol": box_atol,
                    }
                else:
                    first_mismatch = {
                        "kind": "score",
                        "key": _key_json(detection_key),
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
