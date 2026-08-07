"""Versioned, Qt-free configuration and capture provenance contracts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from edge_perception.contracts import Region

CONFIG_SCHEMA_VERSION = "0.1.0"
FPS_ABSOLUTE_TOLERANCE = 0.1
FPS_RELATIVE_TOLERANCE = 0.005


@dataclass(frozen=True, slots=True)
class ConfigPublicationResult:
    """Outcome of an exclusive config link and its owned-temp cleanup."""

    config_path: Path
    retained_temporary: Path | None
    error: OSError | None
    cleanup_diagnostic: str | None


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    device_id: str
    device_description: str
    requested_width: int | None
    requested_height: int | None
    requested_fps: float | None
    strict: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _require_nonempty_string(self.device_id, "device_id"))
        object.__setattr__(
            self,
            "device_description",
            _require_nonempty_string(self.device_description, "device_description"),
        )
        for field_name in ("requested_width", "requested_height"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _require_positive_integer(value, field_name))
        if self.requested_fps is not None:
            object.__setattr__(
                self,
                "requested_fps",
                _require_positive_float(self.requested_fps, "requested FPS"),
            )
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a boolean")

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "device_id": self.device_id,
            "device_description": self.device_description,
            "requested_width": self.requested_width,
            "requested_height": self.requested_height,
            "requested_fps": self.requested_fps,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CaptureRequest:
        values = _require_exact_fields(
            payload,
            {
                "device_id",
                "device_description",
                "requested_width",
                "requested_height",
                "requested_fps",
                "strict",
            },
            "capture request",
        )
        strict = values["strict"]
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        return cls(
            device_id=_require_json_string(values["device_id"], "device_id"),
            device_description=_require_json_string(
                values["device_description"], "device_description"
            ),
            requested_width=_require_optional_positive_integer(
                values["requested_width"], "requested_width"
            ),
            requested_height=_require_optional_positive_integer(
                values["requested_height"], "requested_height"
            ),
            requested_fps=_require_optional_positive_float(values["requested_fps"], "requested FPS"),
            strict=strict,
        )


@dataclass(frozen=True, slots=True)
class CaptureResult:
    request: CaptureRequest
    selected_width: int
    selected_height: int
    selected_min_fps: float
    selected_max_fps: float
    selected_pixel_format: str
    actual_width: int
    actual_height: int
    actual_fps: float
    container: str
    codec: str
    duration_seconds: float
    has_audio: bool
    file_size_bytes: int
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, CaptureRequest):
            raise TypeError("request must be a CaptureRequest")
        for field_name in ("selected_width", "selected_height", "actual_width", "actual_height"):
            object.__setattr__(
                self,
                field_name,
                _require_positive_integer(getattr(self, field_name), field_name),
            )
        for field_name in ("selected_min_fps", "selected_max_fps", "actual_fps"):
            object.__setattr__(
                self,
                field_name,
                _require_positive_float(getattr(self, field_name), field_name),
            )
        if self.selected_min_fps > self.selected_max_fps:
            raise ValueError("selected_min_fps must not exceed selected_max_fps")
        for field_name in ("selected_pixel_format", "container", "codec"):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "duration_seconds",
            _require_nonnegative_float(self.duration_seconds, "duration_seconds"),
        )
        if not isinstance(self.has_audio, bool):
            raise TypeError("has_audio must be a boolean")
        if self.has_audio:
            raise ValueError("GUI capture must not contain audio")
        object.__setattr__(
            self,
            "file_size_bytes",
            _require_positive_integer(self.file_size_bytes, "file_size_bytes"),
        )
        if not isinstance(self.path, (Path, str)):
            raise TypeError("path must be a path")
        object.__setattr__(self, "path", Path(self.path).resolve())
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "selected_width": self.selected_width,
            "selected_height": self.selected_height,
            "selected_min_fps": self.selected_min_fps,
            "selected_max_fps": self.selected_max_fps,
            "selected_pixel_format": self.selected_pixel_format,
            "actual_width": self.actual_width,
            "actual_height": self.actual_height,
            "actual_fps": self.actual_fps,
            "container": self.container,
            "codec": self.codec,
            "duration_seconds": self.duration_seconds,
            "has_audio": self.has_audio,
            "file_size_bytes": self.file_size_bytes,
            "path": str(self.path),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, base_dir: Path) -> CaptureResult:
        values = _require_exact_fields(
            payload,
            {
                "request",
                "selected_width",
                "selected_height",
                "selected_min_fps",
                "selected_max_fps",
                "selected_pixel_format",
                "actual_width",
                "actual_height",
                "actual_fps",
                "container",
                "codec",
                "duration_seconds",
                "has_audio",
                "file_size_bytes",
                "path",
                "sha256",
            },
            "capture result",
        )
        request_payload = _require_mapping(values["request"], "request")
        path = _resolve_config_path(_require_json_string(values["path"], "path"), base_dir)
        has_audio = values["has_audio"]
        if not isinstance(has_audio, bool):
            raise TypeError("has_audio must be a boolean")
        return cls(
            request=CaptureRequest.from_dict(request_payload),
            selected_width=_require_positive_integer(values["selected_width"], "selected_width"),
            selected_height=_require_positive_integer(values["selected_height"], "selected_height"),
            selected_min_fps=_require_positive_float(
                values["selected_min_fps"], "selected_min_fps"
            ),
            selected_max_fps=_require_positive_float(
                values["selected_max_fps"], "selected_max_fps"
            ),
            selected_pixel_format=_require_json_string(
                values["selected_pixel_format"], "selected_pixel_format"
            ),
            actual_width=_require_positive_integer(values["actual_width"], "actual_width"),
            actual_height=_require_positive_integer(values["actual_height"], "actual_height"),
            actual_fps=_require_positive_float(values["actual_fps"], "actual_fps"),
            container=_require_json_string(values["container"], "container"),
            codec=_require_json_string(values["codec"], "codec"),
            duration_seconds=_require_nonnegative_float(
                values["duration_seconds"], "duration_seconds"
            ),
            has_audio=has_audio,
            file_size_bytes=_require_positive_integer(values["file_size_bytes"], "file_size_bytes"),
            path=path,
            sha256=_require_json_string(values["sha256"], "sha256"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    regions: tuple[Region, ...]
    threshold: float
    max_frames: int | None
    warmup_runs: int
    annotate_every: int
    detector_id: str = "dfine-nano-coco"
    device: str = "auto"
    capture: CaptureResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        if not isinstance(self.regions, tuple) or not all(
            isinstance(region, Region) for region in self.regions
        ):
            raise TypeError("regions must be a tuple of Region values")
        region_ids = [region.region_id for region in self.regions]
        if "full-frame" in region_ids:
            raise ValueError("full-frame is a reserved region ID")
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")

        if isinstance(self.threshold, bool):
            raise TypeError("threshold must be a finite number")
        threshold = _require_finite_float(self.threshold, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        object.__setattr__(self, "threshold", threshold)

        if self.max_frames is not None:
            _require_nonnegative_integer(self.max_frames, "max_frames")
        _require_nonnegative_integer(self.warmup_runs, "warmup_runs")
        _require_nonnegative_integer(self.annotate_every, "annotate_every")
        if self.input_path.resolve() == self.output_dir.resolve():
            raise ValueError("output directory must differ from input path")
        object.__setattr__(self, "detector_id", _require_nonempty_string(self.detector_id, "detector_id"))
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of auto, cpu, cuda")
        if self.capture is not None:
            if not isinstance(self.capture, CaptureResult):
                raise TypeError("capture must be a CaptureResult or None")
            if self.capture.path.resolve() != self.input_path.resolve():
                raise ValueError("capture path must equal input_path")


def load_run_config(path: Path) -> RunConfig:
    """Load a versioned experiment document, resolving paths from its location."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=_reject_non_json_constant)
    values = _require_exact_fields(
        _require_mapping(payload, "config"),
        {"schema_version", "source", "output", "detector", "regions", "execution"},
        "config",
    )
    schema_version = _require_json_string(values["schema_version"], "schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    source = _require_exact_fields(
        _require_mapping(values["source"], "source"),
        {"path", "capture"},
        "source",
    )
    output = _require_exact_fields(
        _require_mapping(values["output"], "output"),
        {"directory"},
        "output",
    )
    detector = _require_exact_fields(
        _require_mapping(values["detector"], "detector"),
        {"id", "device", "threshold"},
        "detector",
    )
    execution = _require_exact_fields(
        _require_mapping(values["execution"], "execution"),
        {"max_frames", "warmup_runs", "annotate_every"},
        "execution",
    )
    regions_value = values["regions"]
    if not isinstance(regions_value, list):
        raise TypeError("regions must be an array")
    regions = tuple(_region_from_dict(region) for region in regions_value)
    base_dir = config_path.parent.resolve()
    capture_value = source["capture"]
    if capture_value is None:
        capture = None
    else:
        capture = CaptureResult.from_dict(_require_mapping(capture_value, "source.capture"), base_dir=base_dir)
    return RunConfig(
        input_path=_resolve_config_path(_require_json_string(source["path"], "source.path"), base_dir),
        output_dir=_resolve_config_path(
            _require_json_string(output["directory"], "output.directory"), base_dir
        ),
        regions=regions,
        threshold=_require_finite_float(detector["threshold"], "threshold"),
        max_frames=_require_optional_nonnegative_integer(execution["max_frames"], "max_frames"),
        warmup_runs=_require_nonnegative_integer(execution["warmup_runs"], "warmup_runs"),
        annotate_every=_require_nonnegative_integer(execution["annotate_every"], "annotate_every"),
        detector_id=_require_json_string(detector["id"], "detector.id"),
        device=_require_json_string(detector["device"], "detector.device"),
        capture=capture,
    )


def write_run_config(path: Path, config: RunConfig) -> None:
    """Atomically write a deterministic versioned experiment document."""

    encoded = _encoded_run_config(config)
    config_path = Path(path)
    temporary = config_path.with_name(f"{config_path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, config_path)


def publish_run_config(path: Path, config: RunConfig) -> ConfigPublicationResult:
    """Durably publish a new config without replacing an existing document."""

    encoded = _encoded_run_config(config)
    config_path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=config_path.parent,
        prefix=f".{config_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    stream_open = False
    publication_error: OSError | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream_open = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        publication_error = error
    finally:
        if not stream_open:
            os.close(descriptor)

    if publication_error is None:
        try:
            os.link(temporary, config_path)
        except FileExistsError as error:
            publication_error = FileExistsError(
                f"experiment config already exists: {config_path}"
            )
            publication_error.__cause__ = error
        except OSError as error:
            publication_error = OSError(
                f"exclusive config publication requires hard-link support: {config_path}"
            )
            publication_error.__cause__ = error

    retained_temporary: Path | None = None
    cleanup_diagnostic: str | None = None
    try:
        temporary.unlink(missing_ok=True)
    except OSError as error:
        retained_temporary = temporary
        cleanup_diagnostic = (
            f"config publication cleanup failed while removing {temporary}: {error}"
        )
    return ConfigPublicationResult(
        config_path=config_path,
        retained_temporary=retained_temporary,
        error=publication_error,
        cleanup_diagnostic=cleanup_diagnostic,
    )


def _encoded_run_config(config: RunConfig) -> str:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    capture: dict[str, object] | None = None
    capture_result = config.capture
    if capture_result is not None:
        capture = capture_result.to_dict()
        capture["path"] = str(capture_result.path.resolve())
    document: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "source": {"path": str(config.input_path.resolve()), "capture": capture},
        "output": {"directory": str(config.output_dir.resolve())},
        "detector": {
            "id": config.detector_id,
            "device": config.device,
            "threshold": config.threshold,
        },
        "regions": [region.to_dict() for region in config.regions],
        "execution": {
            "max_frames": config.max_frames,
            "warmup_runs": config.warmup_runs,
            "annotate_every": config.annotate_every,
        },
    }
    return json.dumps(document, allow_nan=False, sort_keys=True) + "\n"


def render_run_cli(config_path: Path, output_override: Path | None = None) -> tuple[str, ...]:
    """Return a shell-free command argument tuple for a configured run."""

    command = ("edge-perception", "run", "--config", str(Path(config_path).resolve()))
    if output_override is None:
        return command
    return (*command, "--output", str(Path(output_override).resolve()))


def _region_from_dict(payload: object) -> Region:
    values = _require_exact_fields(
        _require_mapping(payload, "region"),
        {"region_id", "x", "y", "width", "height"},
        "region",
    )
    return Region(
        _require_json_string(values["region_id"], "region_id"),
        _require_integer(values["x"], "x"),
        _require_integer(values["y"], "y"),
        _require_positive_integer(values["width"], "width"),
        _require_positive_integer(values["height"], "height"),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    payload: Mapping[str, object], expected: set[str], context: str
) -> Mapping[str, object]:
    actual = set(payload)
    unknown = actual - expected
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {', '.join(sorted(unknown))}")
    missing = expected - actual
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}")
    return payload


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_json_string(value: object, field_name: str) -> str:
    return _require_nonempty_string(value, field_name)


def _require_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_positive_integer(value: object, field_name: str) -> int:
    result = _require_integer(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _require_nonnegative_integer(value: object, field_name: str) -> int:
    result = _require_integer(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return result


def _require_optional_positive_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_integer(value, field_name)


def _require_optional_nonnegative_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_integer(value, field_name)


def _require_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _require_positive_float(value: object, field_name: str) -> float:
    result = _require_finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _require_nonnegative_float(value: object, field_name: str) -> float:
    result = _require_finite_float(value, field_name)
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _require_optional_positive_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_positive_float(value, field_name)


def _resolve_config_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")
