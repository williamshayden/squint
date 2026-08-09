"""CPU-only construction of immutable MOT17 replay traces."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from squint_rl.reference.dfine import scene_change_grid
from squint_rl.reference.mot17 import Mot17Sequence
from squint_rl.tracker import DetectionBatch

_ARRAY_NAMES = (
    "timestamps_s", "detector_latency_ms", "scene_change", "det_boxes_xyxy",
    "det_scores", "det_class_ids", "det_frame_offsets", "gt_boxes_xyxy",
    "gt_track_ids", "gt_class_ids", "gt_visibility", "gt_valid", "gt_ignore",
    "gt_frame_offsets",
)
_CAUSAL_NAMES = (
    "timestamps_s", "detector_latency_ms", "scene_change", "det_boxes_xyxy",
    "det_scores", "det_class_ids", "det_frame_offsets",
)
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
_SOURCE_HASH_CHUNK_BYTES = 1024 * 1024


class Detector(Protocol):
    def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]: ...


def _framed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def _array(name: str, value: object, dtype: np.dtype[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    result = np.asarray(value, dtype=dtype)
    if result.size == 0 and 0 in shape:
        result = np.empty(shape, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return np.array(result, dtype=dtype, copy=True)


def _validate_offsets(name: str, offsets: NDArray[Any], count: int) -> None:
    if int(offsets[0]) != 0:
        raise ValueError(f"{name} must start at zero")
    if any(int(right) < int(left) for left, right in pairwise(offsets)):
        raise ValueError(f"{name} must be monotonic")
    if int(offsets[-1]) != count:
        raise ValueError(f"{name} final offset must equal value count {count}")


def _validate_boxes(name: str, boxes: NDArray[Any]) -> None:
    if not np.all(np.isfinite(boxes)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError(f"{name} must have positive-area xyxy boxes")


def _validate_unit_interval(name: str, values: NDArray[Any]) -> None:
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")


def _validate_arrays(arrays: Mapping[str, NDArray[Any]]) -> None:
    expected = set(_ARRAY_NAMES)
    actual = set(arrays)
    if actual != expected:
        raise ValueError(
            "replay-v1 array set mismatch "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    for name, (dtype, _) in _ARRAY_SPECS.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")  # noqa: TRY004
        if value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
    timestamps = arrays["timestamps_s"]
    det_boxes = arrays["det_boxes_xyxy"]
    gt_boxes = arrays["gt_boxes_xyxy"]
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("timestamps_s must have nonempty shape (F,)")
    if det_boxes.ndim != 2 or det_boxes.shape[1:] != (4,):
        raise ValueError("det_boxes_xyxy must have shape (D, 4)")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1:] != (4,):
        raise ValueError("gt_boxes_xyxy must have shape (G, 4)")
    dimensions = {
        "F": len(timestamps),
        "D": len(det_boxes),
        "G": len(gt_boxes),
        "F+1": len(timestamps) + 1,
    }
    for name, (_, shape_spec) in _ARRAY_SPECS.items():
        expected_shape = tuple(
            dimensions[item] if isinstance(item, str) else item for item in shape_spec
        )
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    latency = arrays["detector_latency_ms"]
    if not np.all(np.isfinite(latency)) or np.any(latency < 0):
        raise ValueError("detector_latency_ms must be finite and nonnegative")
    _validate_unit_interval("scene_change", arrays["scene_change"])
    _validate_boxes("det_boxes_xyxy", det_boxes)
    _validate_unit_interval("det_scores", arrays["det_scores"])
    _validate_boxes("gt_boxes_xyxy", gt_boxes)
    _validate_unit_interval("gt_visibility", arrays["gt_visibility"])
    if np.any(arrays["gt_valid"] & arrays["gt_ignore"]):
        raise ValueError("gt_valid and gt_ignore cannot both be true")
    frame_count = dimensions["F"]
    det_offsets = arrays["det_frame_offsets"]
    gt_offsets = arrays["gt_frame_offsets"]
    _validate_offsets("det_frame_offsets", det_offsets, dimensions["D"])
    _validate_offsets("gt_frame_offsets", gt_offsets, dimensions["G"])
    for index in range(frame_count):
        start, stop = int(gt_offsets[index]), int(gt_offsets[index + 1])
        track_ids = arrays["gt_track_ids"][start:stop]
        if len(np.unique(track_ids)) != len(track_ids):
            raise ValueError(f"gt_track_ids must be unique within frame {index}")


def _immutable_array(value: NDArray[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def pack_episode_arrays(
    sequence: Mot17Sequence,
    detections: Sequence[DetectionBatch],
    scene_features: Sequence[NDArray[Any]],
    latencies: Sequence[float],
) -> dict[str, NDArray[Any]]:
    frame_count = len(sequence.image_paths)
    if not np.isfinite(sequence.fps) or sequence.fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not (len(detections) == len(scene_features) == len(latencies) == frame_count):
        raise ValueError("detections, scene_features, and latencies cardinality must equal frames")
    det_boxes: list[NDArray[Any]] = []
    det_scores: list[NDArray[Any]] = []
    det_classes: list[NDArray[Any]] = []
    det_offsets = [0]
    for batch in detections:
        det_boxes.append(np.array(batch.boxes_xyxy, dtype=np.float32, copy=True))
        det_scores.append(np.array(batch.scores, dtype=np.float32, copy=True))
        det_classes.append(np.array(batch.class_ids, dtype=np.int64, copy=True))
        det_offsets.append(det_offsets[-1] + len(batch))
    gt_boxes: list[NDArray[Any]] = []
    gt_tracks: list[NDArray[Any]] = []
    gt_classes: list[NDArray[Any]] = []
    gt_visibility: list[NDArray[Any]] = []
    gt_valid: list[NDArray[Any]] = []
    gt_ignore: list[NDArray[Any]] = []
    gt_offsets = [0]
    for gt_batch in sequence.ground_truth:
        gt_boxes.append(np.array(gt_batch.boxes_xyxy, dtype=np.float32, copy=True))
        gt_tracks.append(np.array(gt_batch.track_ids, dtype=np.int64, copy=True))
        gt_classes.append(np.array(gt_batch.class_ids, dtype=np.int64, copy=True))
        gt_visibility.append(np.array(gt_batch.visibility, dtype=np.float32, copy=True))
        gt_valid.append(np.array(gt_batch.valid, dtype=np.bool_, copy=True))
        gt_ignore.append(np.array(gt_batch.ignore, dtype=np.bool_, copy=True))
        gt_offsets.append(gt_offsets[-1] + len(gt_batch))
    if len(sequence.ground_truth) != frame_count:
        raise ValueError("ground_truth cardinality must equal frames")

    def flat(items: list[NDArray[Any]], tail: tuple[int, ...], dtype: np.dtype[Any]) -> NDArray[Any]:
        return np.concatenate(items, axis=0).astype(dtype, copy=False) if items else np.empty((0, *tail), dtype=dtype)

    arrays = {
        "timestamps_s": np.arange(frame_count, dtype=np.float64) / float(sequence.fps),
        "detector_latency_ms": _array("latencies", latencies, np.dtype(np.float32), (frame_count,)),
        "scene_change": _array("scene_features", scene_features, np.dtype(np.float32), (frame_count, 3, 3)),
        "det_boxes_xyxy": flat(det_boxes, (4,), np.dtype(np.float32)),
        "det_scores": flat(det_scores, (), np.dtype(np.float32)).reshape(-1),
        "det_class_ids": flat(det_classes, (), np.dtype(np.int64)).reshape(-1),
        "det_frame_offsets": np.asarray(det_offsets, dtype=np.int64),
        "gt_boxes_xyxy": flat(gt_boxes, (4,), np.dtype(np.float32)),
        "gt_track_ids": flat(gt_tracks, (), np.dtype(np.int64)).reshape(-1),
        "gt_class_ids": flat(gt_classes, (), np.dtype(np.int64)).reshape(-1),
        "gt_visibility": flat(gt_visibility, (), np.dtype(np.float32)).reshape(-1),
        "gt_valid": flat(gt_valid, (), np.dtype(np.bool_)).reshape(-1),
        "gt_ignore": flat(gt_ignore, (), np.dtype(np.bool_)).reshape(-1),
        "gt_frame_offsets": np.asarray(gt_offsets, dtype=np.int64),
    }
    ordered = {name: arrays[name] for name in _ARRAY_NAMES}
    _validate_arrays(ordered)
    return ordered


def causal_trace_sha256(arrays: Mapping[str, NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    _framed(digest, b"squint.replay\x00causal-v1")
    for name in _CAUSAL_NAMES:
        if name not in arrays or not isinstance(arrays[name], np.ndarray):
            raise ValueError(f"causal trace requires numpy array {name}")
        value = np.ascontiguousarray(arrays[name])
        _framed(digest, name.encode("utf-8"))
        _framed(digest, value.dtype.str.encode("ascii"))
        _framed(digest, struct.pack("<Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", dimension))
        _framed(digest, value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_source_sha256(sequence: Mot17Sequence) -> str:
    files = [*sequence.image_paths, sequence.source_dir / "seqinfo.ini", sequence.source_dir / "gt" / "gt.txt"]
    entries: list[tuple[str, Path]] = []
    for path in files:
        try:
            relative = path.relative_to(sequence.source_dir).as_posix()
        except ValueError as error:
            raise ValueError(f"canonical source path is outside source_dir: {path}") from error
        entries.append((relative, path))
    digest = hashlib.sha256()
    _framed(digest, b"squint.mot17-source-v1")
    for name, path in sorted(entries):
        file_digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(_SOURCE_HASH_CHUNK_BYTES), b""):
                    file_digest.update(block)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"unable to read canonical source file {name} ({path})") from error
        _framed(digest, name.encode("utf-8"))
        _framed(digest, file_digest.digest())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawTrace:
    sequence_id: str
    arrays: Mapping[str, NDArray[Any]]
    manifest_fields: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.arrays, Mapping):
            raise ValueError("arrays must be a mapping")  # noqa: TRY004
        _validate_arrays(self.arrays)
        if not isinstance(self.manifest_fields, Mapping):
            raise ValueError("manifest_fields must be a mapping")  # noqa: TRY004
        manifest = dict(self.manifest_fields)
        expected_hash = causal_trace_sha256(self.arrays)
        if manifest.get("causal_trace_sha256") != expected_hash:
            raise ValueError("manifest_fields causal_trace_sha256 does not match arrays")
        frozen_arrays = {
            name: _immutable_array(self.arrays[name]) for name in _ARRAY_NAMES
        }
        object.__setattr__(self, "arrays", MappingProxyType(frozen_arrays))
        object.__setattr__(self, "manifest_fields", MappingProxyType(manifest))


def build_sequence(
    sequence: Mot17Sequence,
    detector: Detector,
    warmup_frames: int = 10,
) -> RawTrace:
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int):
        raise TypeError("warmup_frames must be an integer")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be nonnegative")
    for image_path in sequence.image_paths[:warmup_frames]:
        with Image.open(image_path) as opened:
            detector.predict(opened.convert("RGB"))

    previous: Image.Image | None = None
    detections: list[DetectionBatch] = []
    scenes: list[NDArray[Any]] = []
    latencies: list[float] = []
    for image_path in sequence.image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            scenes.append(scene_change_grid(previous, image))
            batch, latency = detector.predict(image)
            detections.append(batch)
            latencies.append(float(latency))
            previous = image.copy()
    packed = pack_episode_arrays(sequence, detections, scenes, latencies)
    manifest = {
        "schema": "squint.replay",
        "schema_version": 1,
        "sequence_id": sequence.identifier,
        "frame_count": len(sequence.image_paths),
        "fps": float(sequence.fps),
        "width": sequence.width,
        "height": sequence.height,
        "warmup_frames": warmup_frames,
        "source_sha256": canonical_source_sha256(sequence),
        "causal_trace_sha256": causal_trace_sha256(packed),
    }
    return RawTrace(sequence.identifier, packed, manifest)
