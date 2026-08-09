"""CPU-only construction of immutable MOT17 replay traces."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
    return {name: arrays[name] for name in _ARRAY_NAMES}


def causal_trace_sha256(arrays: Mapping[str, NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    _framed(digest, b"squint.replay\x00causal-v1")
    for name in _CAUSAL_NAMES:
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
    entries: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(sequence.source_dir).as_posix().encode("utf-8")
        entries.append((relative.decode("utf-8"), path.read_bytes()))
    digest = hashlib.sha256()
    _framed(digest, b"squint.mot17-source-v1")
    for name, content in sorted(entries):
        _framed(digest, name.encode("utf-8"))
        _framed(digest, hashlib.sha256(content).digest())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawTrace:
    sequence_id: str
    arrays: Mapping[str, NDArray[Any]]
    manifest_fields: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen_arrays: dict[str, NDArray[Any]] = {}
        for name, value in self.arrays.items():
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            frozen_arrays[name] = frozen
        object.__setattr__(self, "arrays", MappingProxyType(frozen_arrays))
        object.__setattr__(self, "manifest_fields", MappingProxyType(dict(self.manifest_fields)))


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

    measured_sequence = replace(
        sequence,
        image_paths=sequence.image_paths[warmup_frames:],
        ground_truth=sequence.ground_truth[warmup_frames:],
    )
    previous: Image.Image | None = None
    detections: list[DetectionBatch] = []
    scenes: list[NDArray[Any]] = []
    latencies: list[float] = []
    for image_path in measured_sequence.image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            scenes.append(scene_change_grid(previous, image))
            batch, latency = detector.predict(image)
            detections.append(batch)
            latencies.append(float(latency))
            previous = image.copy()
    packed = pack_episode_arrays(measured_sequence, detections, scenes, latencies)
    frozen_arrays: dict[str, NDArray[Any]] = {}
    for name, value in packed.items():
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        frozen_arrays[name] = frozen
    manifest = {
        "schema": "squint.replay",
        "schema_version": 1,
        "sequence_id": sequence.identifier,
        "frame_count": len(measured_sequence.image_paths),
        "fps": float(sequence.fps),
        "width": sequence.width,
        "height": sequence.height,
        "warmup_frames": warmup_frames,
        "source_sha256": canonical_source_sha256(sequence),
        "causal_trace_sha256": causal_trace_sha256(frozen_arrays),
    }
    return RawTrace(sequence.identifier, MappingProxyType(frozen_arrays), MappingProxyType(manifest))
