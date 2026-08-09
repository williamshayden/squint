from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
from numpy.typing import NDArray

from .tracker import DetectionBatch, GroundTruthBatch

SCHEMA_NAME = "squint.replay"
SCHEMA_VERSION = 1

_ARRAY_SPECS: dict[str, tuple[np.dtype[Any], tuple[int | str, ...]]] = {
    "timestamps_s": (np.dtype(np.float64), ("F",)),
    "detector_latency_ms": (np.dtype(np.float32), ("F",)),
    "scene_change": (np.dtype(np.float32), ("F", 3, 3)),
    "det_boxes_xyxy": (np.dtype(np.float32), ("D", 4)),
    "det_scores": (np.dtype(np.float32), ("D",)),
    "det_class_ids": (np.dtype(np.int64), ("D",)),
    "det_frame_offsets": (np.dtype(np.int64), ("F+1",)),
    "gt_boxes_xyxy": (np.dtype(np.float32), ("G", 4)),
    "gt_track_ids": (np.dtype(np.int64), ("G",)),
    "gt_class_ids": (np.dtype(np.int64), ("G",)),
    "gt_visibility": (np.dtype(np.float32), ("G",)),
    "gt_valid": (np.dtype(np.bool_), ("G",)),
    "gt_ignore": (np.dtype(np.bool_), ("G",)),
    "gt_frame_offsets": (np.dtype(np.int64), ("F+1",)),
}

_REQUIRED_MANIFEST_OBJECTS = (
    "schema",
    "episode",
    "source",
    "detector",
    "hardware",
    "cost_profile",
    "scene_feature",
    "normalization",
    "telemetry",
    "artifacts",
)


class EpisodeValidationError(ValueError):
    pass


def _immutable_array(value: NDArray[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EpisodeValidationError(f"manifest {name} must be an object")
    return cast(Mapping[str, object], value)


def _manifest_frame_count(manifest: Mapping[str, object]) -> int:
    source = _mapping(manifest["source"], "source")
    frame_count = source.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise EpisodeValidationError("manifest source.frame_count must be a positive integer")
    return frame_count


def _canonical_content_hash(manifest: Mapping[str, object]) -> str:
    normalized = cast(dict[str, object], _json_value(manifest))
    artifacts = cast(dict[str, object], normalized["artifacts"])
    content_hash = artifacts.pop("content_sha256", None)
    if not isinstance(content_hash, str):
        raise EpisodeValidationError("manifest artifacts.content_sha256 is required")
    arrays_hash = artifacts.get("arrays.npz_sha256")
    if not isinstance(arrays_hash, str):
        raise EpisodeValidationError("manifest artifacts.arrays.npz_sha256 is required")
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded + arrays_hash.encode()).hexdigest()


def _validate_manifest(manifest: Mapping[str, object], *, sealed: bool) -> None:
    schema = _mapping(manifest.get("schema"), "schema")
    if schema.get("name") != SCHEMA_NAME:
        raise EpisodeValidationError(f"unsupported schema name: {schema.get('name')!r}")
    if schema.get("version") != SCHEMA_VERSION:
        raise EpisodeValidationError(f"unsupported schema version: {schema.get('version')!r}")
    for name in _REQUIRED_MANIFEST_OBJECTS:
        if name not in manifest:
            raise EpisodeValidationError(f"manifest {name} object is required")
        _mapping(manifest[name], name)
    source = _mapping(manifest["source"], "source")
    _manifest_frame_count(manifest)
    fps = source.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise EpisodeValidationError("manifest source.fps must be finite and positive")
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    if sealed:
        for name in ("arrays.npz_sha256", "content_sha256"):
            value = artifacts.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise EpisodeValidationError(f"manifest artifacts.{name} is required")


def _validate_arrays(manifest: Mapping[str, object], arrays: Mapping[str, NDArray[Any]]) -> None:
    expected_names = set(_ARRAY_SPECS)
    actual_names = set(arrays)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise EpisodeValidationError(
            f"arrays.npz array set mismatch (missing={missing}, unexpected={unexpected})"
        )
    frame_count = _manifest_frame_count(manifest)
    values = {
        "F": frame_count,
        "D": len(arrays["det_boxes_xyxy"]),
        "G": len(arrays["gt_boxes_xyxy"]),
        "F+1": frame_count + 1,
    }
    for name, (dtype, shape_spec) in _ARRAY_SPECS.items():
        array = arrays[name]
        expected_shape = tuple(values[item] if isinstance(item, str) else item for item in shape_spec)
        if array.dtype != dtype:
            raise EpisodeValidationError(f"{name} must have dtype {dtype}")
        if array.shape != expected_shape:
            raise EpisodeValidationError(f"{name} must have shape {expected_shape}")
    validate_offsets(
        "det_frame_offsets",
        cast(NDArray[np.int64], arrays["det_frame_offsets"]),
        frame_count=frame_count,
        value_count=values["D"],
    )
    validate_offsets(
        "gt_frame_offsets",
        cast(NDArray[np.int64], arrays["gt_frame_offsets"]),
        frame_count=frame_count,
        value_count=values["G"],
    )
    timestamps = arrays["timestamps_s"]
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise EpisodeValidationError("timestamps_s must be finite and strictly increasing")
    latency = arrays["detector_latency_ms"]
    if not np.all(np.isfinite(latency)) or np.any(latency < 0):
        raise EpisodeValidationError("detector_latency_ms must be finite and nonnegative")
    scene_change = arrays["scene_change"]
    if not np.all(np.isfinite(scene_change)) or np.any((scene_change < 0) | (scene_change > 1)):
        raise EpisodeValidationError("scene_change must contain finite values in [0, 1]")
    det_offsets = arrays["det_frame_offsets"]
    gt_offsets = arrays["gt_frame_offsets"]
    for index in range(frame_count):
        det_start, det_stop = (int(value) for value in det_offsets[index : index + 2])
        gt_start, gt_stop = (int(value) for value in gt_offsets[index : index + 2])
        try:
            DetectionBatch(
                arrays["det_boxes_xyxy"][det_start:det_stop],
                arrays["det_scores"][det_start:det_stop],
                arrays["det_class_ids"][det_start:det_stop],
            )
            GroundTruthBatch(
                arrays["gt_boxes_xyxy"][gt_start:gt_stop],
                arrays["gt_track_ids"][gt_start:gt_stop],
                arrays["gt_class_ids"][gt_start:gt_stop],
                arrays["gt_visibility"][gt_start:gt_stop],
                arrays["gt_valid"][gt_start:gt_stop],
                arrays["gt_ignore"][gt_start:gt_stop],
            )
        except ValueError as error:
            raise EpisodeValidationError(f"frame {index} records are invalid: {error}") from error


def _validate_content_hash(manifest: Mapping[str, object]) -> None:
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    expected = artifacts.get("content_sha256")
    actual = _canonical_content_hash(manifest)
    if actual != expected:
        raise EpisodeValidationError("content_sha256 mismatch")


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    index: int
    timestamp_s: float
    detector_latency_ms: float
    scene_change: NDArray[np.float32]
    detections: DetectionBatch
    ground_truth: GroundTruthBatch

    def __post_init__(self) -> None:
        scene_change = np.asarray(self.scene_change)
        if scene_change.dtype != np.dtype(np.float32) or scene_change.shape != (3, 3):
            raise ValueError("scene_change must have dtype float32 and shape (3, 3)")
        object.__setattr__(self, "scene_change", cast(NDArray[np.float32], _immutable_array(scene_change)))


@dataclass(frozen=True, slots=True)
class Episode:
    path: Path
    manifest: Mapping[str, object]
    arrays: Mapping[str, NDArray[Any]]
    content_sha256: str

    @classmethod
    def open(cls, path: str | Path) -> Episode:
        episode_path = Path(path).resolve()
        manifest_path = episode_path / "manifest.json"
        arrays_path = episode_path / "arrays.npz"
        if not episode_path.is_dir():
            raise EpisodeValidationError(f"{episode_path}: episode directory is required")
        if not manifest_path.is_file():
            raise EpisodeValidationError(f"{manifest_path}: manifest.json is required")
        if not arrays_path.is_file():
            raise EpisodeValidationError(f"{arrays_path}: arrays.npz is required")
        if {item.name for item in episode_path.iterdir()} != {"manifest.json", "arrays.npz"}:
            raise EpisodeValidationError("episode must contain only manifest.json and arrays.npz")
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EpisodeValidationError(f"{manifest_path}: invalid JSON") from error
        manifest = _mapping(loaded_manifest, "root")
        _validate_manifest(manifest, sealed=True)
        expected_arrays_hash = _mapping(manifest["artifacts"], "artifacts")["arrays.npz_sha256"]
        actual_arrays_hash = sha256(arrays_path.read_bytes()).hexdigest()
        if actual_arrays_hash != expected_arrays_hash:
            raise EpisodeValidationError("arrays.npz sha256 mismatch")
        try:
            with np.load(arrays_path, allow_pickle=False) as stored:
                arrays = {name: _immutable_array(np.array(stored[name], copy=True)) for name in stored.files}
        except (OSError, ValueError) as error:
            raise EpisodeValidationError(f"{arrays_path}: invalid arrays.npz") from error
        frozen_manifest = cast(Mapping[str, object], _freeze_json(manifest))
        episode = cls(
            episode_path,
            frozen_manifest,
            MappingProxyType(arrays),
            cast(str, _mapping(manifest["artifacts"], "artifacts")["content_sha256"]),
        )
        episode.validate()
        return episode

    @property
    def frame_count(self) -> int:
        return _manifest_frame_count(self.manifest)

    @property
    def fps(self) -> float:
        return float(cast(float, _mapping(self.manifest["source"], "source")["fps"]))

    def frame(self, index: int) -> ReplayFrame:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        det_offsets = self.arrays["det_frame_offsets"]
        gt_offsets = self.arrays["gt_frame_offsets"]
        det_start, det_stop = (int(value) for value in det_offsets[index : index + 2])
        gt_start, gt_stop = (int(value) for value in gt_offsets[index : index + 2])
        return ReplayFrame(
            index=index,
            timestamp_s=float(self.arrays["timestamps_s"][index]),
            detector_latency_ms=float(self.arrays["detector_latency_ms"][index]),
            scene_change=cast(NDArray[np.float32], self.arrays["scene_change"][index]),
            detections=DetectionBatch(
                cast(NDArray[np.float32], self.arrays["det_boxes_xyxy"][det_start:det_stop]),
                cast(NDArray[np.float32], self.arrays["det_scores"][det_start:det_stop]),
                cast(NDArray[np.int64], self.arrays["det_class_ids"][det_start:det_stop]),
            ),
            ground_truth=GroundTruthBatch(
                cast(NDArray[np.float32], self.arrays["gt_boxes_xyxy"][gt_start:gt_stop]),
                cast(NDArray[np.int64], self.arrays["gt_track_ids"][gt_start:gt_stop]),
                cast(NDArray[np.int64], self.arrays["gt_class_ids"][gt_start:gt_stop]),
                cast(NDArray[np.float32], self.arrays["gt_visibility"][gt_start:gt_stop]),
                cast(NDArray[np.bool_], self.arrays["gt_valid"][gt_start:gt_stop]),
                cast(NDArray[np.bool_], self.arrays["gt_ignore"][gt_start:gt_stop]),
            ),
        )

    def slice(self, start: int, stop: int) -> EpisodeView:
        return EpisodeView(self, start, stop)

    def validate(self) -> None:
        _validate_manifest(self.manifest, sealed=True)
        _validate_arrays(self.manifest, self.arrays)
        _validate_content_hash(self.manifest)


@dataclass(frozen=True, slots=True)
class EpisodeView:
    parent: Episode
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop > self.parent.frame_count or self.start >= self.stop:
            raise ValueError("episode slice must be a nonempty in-range interval")

    @property
    def frame_count(self) -> int:
        return self.stop - self.start

    @property
    def fps(self) -> float:
        return self.parent.fps

    @property
    def content_sha256(self) -> str:
        return sha256(f"{self.parent.content_sha256}:{self.start}:{self.stop}".encode()).hexdigest()

    def frame(self, index: int) -> ReplayFrame:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        return replace(self.parent.frame(self.start + index), index=index)


def validate_offsets(
    name: str,
    offsets: NDArray[np.int64],
    *,
    frame_count: int,
    value_count: int,
) -> None:
    if offsets.shape != (frame_count + 1,):
        raise EpisodeValidationError(f"{name} offset length must equal frame_count + 1")
    if offsets[0] != 0:
        raise EpisodeValidationError(f"{name} offsets must start at zero")
    if np.any(np.diff(offsets) < 0):
        raise EpisodeValidationError(f"{name} offsets must be monotonic")
    if offsets[-1] != value_count:
        raise EpisodeValidationError(f"{name} final offset must equal value count")


def write_deterministic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(  # type: ignore[no-untyped-call]
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            entry = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_STORED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())


def seal_episode(
    path: str | Path,
    *,
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray[Any]],
) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    _validate_manifest(manifest, sealed=False)
    _validate_arrays(manifest, arrays)
    working = destination.parent / f".{destination.name}.{uuid4().hex}.incomplete"
    working.mkdir(parents=True)
    arrays_path = working / "arrays.npz"
    write_deterministic_npz(arrays_path, arrays)
    arrays_hash = sha256(arrays_path.read_bytes()).hexdigest()
    sealed = copy.deepcopy(cast(dict[str, object], _json_value(manifest)))
    artifacts = cast(dict[str, object], sealed["artifacts"])
    artifacts["arrays.npz_sha256"] = arrays_hash
    artifacts.pop("content_sha256", None)
    sealed["artifacts"] = artifacts
    normalized = json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    artifacts["content_sha256"] = sha256(normalized + arrays_hash.encode()).hexdigest()
    manifest_path = working / "manifest.json"
    manifest_path.write_text(
        json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for item in (arrays_path, manifest_path):
        with item.open("rb") as stream:
            os.fsync(stream.fileno())
    os.replace(working, destination)
    return destination
