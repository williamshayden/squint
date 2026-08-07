"""Read-only projection of canonical completed-run artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, cast

from edge_perception.config import CaptureResult
from edge_perception.contracts import Region
from edge_perception.geometry import validate_region

_RUN_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class RunViewData:
    run_dir: Path
    status: Literal["complete", "failed", "cancelled"]
    frames_processed: int
    inference_count: int
    annotated_frame_count: int
    frame_p50_ms: float | None
    frame_p95_ms: float | None
    frame_p99_ms: float | None
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    detector_model_id: str
    detector_revision: str
    device: str
    threshold: float
    source_path: Path
    source_width: int
    source_height: int
    capture: CaptureResult | None
    regions: tuple[Region, ...]
    annotation_paths: tuple[Path, ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class _NonFiniteToken:
    value: str


class _NonFiniteJSON(ValueError):
    pass


def _reject_constant(value: str) -> object:
    raise _NonFiniteJSON(value)


def _find_non_finite(value: object, path: str) -> str | None:
    if isinstance(value, _NonFiniteToken):
        return path
    if isinstance(value, dict):
        for key, nested in value.items():
            found = _find_non_finite(nested, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _find_non_finite(nested, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _json_object(path: Path) -> dict[str, object]:
    artifact_name = path.name
    try:
        serialized = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{artifact_name}: {error}") from error
    try:
        value = json.loads(serialized, parse_constant=_reject_constant)
    except _NonFiniteJSON as error:
        try:
            diagnostic = json.loads(
                serialized,
                parse_constant=lambda token: _NonFiniteToken(token),
            )
        except json.JSONDecodeError:
            diagnostic = None
        field = _find_non_finite(diagnostic, artifact_name)
        raise ValueError(f"{field or artifact_name} must be finite") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"{artifact_name}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{artifact_name}: root must be an object")  # noqa: TRY004
    return cast(dict[str, object], value)


def _field(parent: dict[str, object], key: str, path: str) -> object:
    try:
        return parent[key]
    except KeyError as error:
        raise ValueError(f"{path}.{key} is required") from error


def _mapping(parent: dict[str, object], key: str, path: str) -> dict[str, object]:
    field_path = f"{path}.{key}"
    value = _field(parent, key, path)
    if not isinstance(value, dict):
        raise ValueError(f"{field_path} must be an object")  # noqa: TRY004
    return cast(dict[str, object], value)


def _string(
    parent: dict[str, object],
    key: str,
    path: str,
    *,
    nonempty: bool = True,
) -> str:
    field_path = f"{path}.{key}"
    value = _field(parent, key, path)
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise ValueError(f"{field_path} must be {qualifier}")
    return value


def _integer(
    parent: dict[str, object],
    key: str,
    path: str,
    *,
    positive: bool = False,
) -> int:
    field_path = f"{path}.{key}"
    value = _field(parent, key, path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_path} must be an integer")  # noqa: TRY004
    if positive and value <= 0:
        raise ValueError(f"{field_path} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_path} must not be negative")
    return value


def _sha256(parent: dict[str, object], key: str, path: str) -> str:
    field_path = f"{path}.{key}"
    value = _string(parent, key, path)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"{field_path} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _resolved_path(
    parent: dict[str, object],
    key: str,
    path: str,
    *,
    base_dir: Path,
) -> Path:
    field_path = f"{path}.{key}"
    value = Path(_string(parent, key, path))
    if not value.is_absolute():
        value = base_dir / value
    try:
        return value.resolve()
    except OSError as error:
        raise ValueError(f"{field_path}: {error}") from error


def _number(parent: dict[str, object], key: str, path: str) -> float:
    field_path = f"{path}.{key}"
    value = _field(parent, key, path)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_path} must be a number")  # noqa: TRY004
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_path} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{field_path} must not be negative")
    return normalized


def _nullable_number(parent: dict[str, object], key: str, path: str) -> float | None:
    if _field(parent, key, path) is None:
        return None
    return _number(parent, key, path)


def _nullable_integer(parent: dict[str, object], key: str, path: str) -> int | None:
    if _field(parent, key, path) is None:
        return None
    return _integer(parent, key, path)


def _latency_metrics(summary: dict[str, object]) -> tuple[float | None, float | None, float | None]:
    latency = _mapping(summary, "latency_ms", "summary.json")
    complete = _mapping(latency, "complete_frame", "summary.json.latency_ms")
    count = _integer(complete, "count", "summary.json.latency_ms.complete_frame")
    metrics = tuple(
        _nullable_number(complete, key, "summary.json.latency_ms.complete_frame")
        for key in ("p50_ms", "p95_ms", "p99_ms")
    )
    if count == 0 and any(metric is not None for metric in metrics):
        raise ValueError("summary.json.latency_ms.complete_frame metrics must be null when count is zero")
    if count > 0 and any(metric is None for metric in metrics):
        raise ValueError(
            "summary.json.latency_ms.complete_frame metrics must be numbers when count is positive"
        )
    return cast(tuple[float | None, float | None, float | None], metrics)


def _capture(source: dict[str, object], run_dir: Path) -> CaptureResult | None:
    value = _field(source, "capture", "manifest.json.source_video")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004
            "manifest.json.source_video.capture must be an object or null"
        )
    try:
        return CaptureResult.from_dict(cast(dict[str, object], value), base_dir=run_dir)
    except (TypeError, ValueError) as error:
        raise ValueError(f"manifest.json.source_video.capture: {error}") from error


def _regions(
    configuration: dict[str, object],
    source_width: int,
    source_height: int,
) -> tuple[Region, ...]:
    value = _field(configuration, "regions", "manifest.json.configuration")
    if not isinstance(value, list):
        raise ValueError(  # noqa: TRY004
            "manifest.json.configuration.regions must be an array"
        )
    parsed: list[Region] = []
    seen_ids: set[str] = set()
    expected_fields = {"region_id", "x", "y", "width", "height"}
    for index, item in enumerate(value):
        field_path = f"manifest.json.configuration.regions[{index}]"
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(f"{field_path} must contain exactly {sorted(expected_fields)}")
        region_values = cast(dict[str, object], item)
        region_id = _string(region_values, "region_id", field_path)
        if region_id in seen_ids:
            raise ValueError("manifest.json.configuration.regions contains duplicate region IDs")
        try:
            region = Region(
                region_id,
                _integer(region_values, "x", field_path),
                _integer(region_values, "y", field_path),
                _integer(region_values, "width", field_path, positive=True),
                _integer(region_values, "height", field_path, positive=True),
            )
            validate_region(region, source_width, source_height)
        except (TypeError, ValueError) as error:
            raise ValueError(f"manifest.json.configuration.regions: {error}") from error
        parsed.append(region)
        seen_ids.add(region_id)
    return tuple(parsed)


def _annotations(run_dir: Path) -> tuple[Path, ...]:
    annotated = run_dir / "annotated"
    if annotated.is_symlink() or annotated.is_junction() or not annotated.is_dir():
        raise ValueError("annotated must be a real directory")
    root = annotated.resolve()
    selected: list[Path] = []
    try:
        children = sorted(annotated.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise ValueError(f"annotated: {error}") from error
    for path in children:
        if path.suffix != ".png":
            continue
        field_path = f"annotated/{path.name}"
        if path.is_symlink():
            raise ValueError(f"{field_path} must not be a symlink")
        try:
            resolved = path.resolve()
        except OSError as error:
            raise ValueError(f"{field_path}: {error}") from error
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"{field_path} must be a contained regular file")
        selected.append(resolved)
    return tuple(selected)


def load_run_view(run_dir: Path) -> RunViewData:
    """Validate and project one canonical run without mutating it."""

    resolved_run_dir = Path(run_dir).resolve()
    manifest = _json_object(resolved_run_dir / "manifest.json")
    summary = _json_object(resolved_run_dir / "summary.json")
    manifest_schema = _string(manifest, "schema_version", "manifest.json")
    summary_schema = _string(summary, "schema_version", "summary.json")
    if manifest_schema != _RUN_SCHEMA_VERSION:
        raise ValueError(f"manifest.json.schema_version is unsupported: {manifest_schema}")
    if summary_schema != _RUN_SCHEMA_VERSION:
        raise ValueError(f"summary.json.schema_version is unsupported: {summary_schema}")
    manifest_run_id = _string(manifest, "run_id", "manifest.json")
    summary_run_id = _string(summary, "run_id", "summary.json")
    if manifest_run_id != summary_run_id:
        raise ValueError("manifest.json.run_id must match summary.json.run_id")

    status_value = _string(summary, "status", "summary.json")
    if status_value not in {"complete", "failed", "cancelled"}:
        raise ValueError("summary.json.status must be complete, failed, or cancelled")
    status = cast(Literal["complete", "failed", "cancelled"], status_value)
    if status == "failed":
        error_value = _string(summary, "error", "summary.json")
    else:
        if "error" in summary:
            raise ValueError(f"summary.json.error is not valid for {status} status")
        error_value = None

    configuration = _mapping(manifest, "configuration", "manifest.json")
    detector = _mapping(manifest, "detector", "manifest.json")
    source = _mapping(manifest, "source_video", "manifest.json")
    hardware = _mapping(summary, "hardware_peaks", "summary.json")
    source_width = _integer(source, "frame_width", "manifest.json.source_video", positive=True)
    source_height = _integer(source, "frame_height", "manifest.json.source_video", positive=True)
    source_path = _resolved_path(
        source,
        "path",
        "manifest.json.source_video",
        base_dir=resolved_run_dir,
    )
    source_sha256 = _sha256(source, "sha256", "manifest.json.source_video")
    capture = _capture(source, resolved_run_dir)
    if capture is not None:
        if capture.path.resolve() != source_path:
            raise ValueError(
                "manifest.json.source_video.capture.path must resolve to "
                "manifest.json.source_video.path"
            )
        if capture.sha256 != source_sha256:
            raise ValueError(
                "manifest.json.source_video.capture.sha256 must match "
                "manifest.json.source_video.sha256"
            )
    threshold = _number(configuration, "threshold", "manifest.json.configuration")
    if threshold > 1.0:
        raise ValueError("manifest.json.configuration.threshold must not exceed 1.0")
    annotation_paths = _annotations(resolved_run_dir)
    annotated_frame_count = _integer(summary, "annotated_frame_count", "summary.json")
    if annotated_frame_count != len(annotation_paths):
        raise ValueError(
            "summary.json.annotated_frame_count must equal the direct annotated PNG count"
        )
    p50, p95, p99 = _latency_metrics(summary)

    return RunViewData(
        run_dir=resolved_run_dir,
        status=status,
        frames_processed=_integer(summary, "frames_processed", "summary.json"),
        inference_count=_integer(summary, "inference_count", "summary.json"),
        annotated_frame_count=annotated_frame_count,
        frame_p50_ms=p50,
        frame_p95_ms=p95,
        frame_p99_ms=p99,
        peak_rss_bytes=_nullable_integer(hardware, "process_rss_bytes", "summary.json.hardware_peaks"),
        peak_vram_bytes=_nullable_integer(
            summary,
            "detector_peak_device_memory_bytes",
            "summary.json",
        ),
        detector_model_id=_string(detector, "model_id", "manifest.json.detector", nonempty=False),
        detector_revision=_string(detector, "revision", "manifest.json.detector", nonempty=False),
        device=_string(detector, "device", "manifest.json.detector", nonempty=False),
        threshold=threshold,
        source_path=source_path,
        source_width=source_width,
        source_height=source_height,
        capture=capture,
        regions=_regions(configuration, source_width, source_height),
        annotation_paths=annotation_paths,
        error=error_value,
    )
