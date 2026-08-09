from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Observation = dict[str, FloatArray]


def _array(
    value: object,
    dtype: np.dtype[Any],
    shape_tail: tuple[int, ...],
    name: str,
) -> FloatArray:
    result = cast(FloatArray, np.array(value, dtype=dtype, copy=True))
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        dimensions = ", ".join(map(str, shape_tail))
        raise ValueError(f"{name} must have shape (N, {dimensions})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result[:, 2] <= result[:, 0]) or np.any(result[:, 3] <= result[:, 1]):
        raise ValueError(f"{name} must have x2 > x1 and y2 > y1")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    boxes_xyxy: FloatArray
    scores: FloatArray
    class_ids: IntArray

    def __post_init__(self) -> None:
        boxes = _array(self.boxes_xyxy, np.dtype(np.float32), (4,), "boxes_xyxy")
        scores = np.array(self.scores, dtype=np.float32, copy=True)
        classes = np.array(self.class_ids, dtype=np.int64, copy=True)
        if scores.shape != (len(boxes),) or classes.shape != (len(boxes),):
            raise ValueError("scores and class_ids must have shape (N,)")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must be finite values in [0, 1]")
        scores.setflags(write=False)
        classes.setflags(write=False)
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_ids", classes)

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    @classmethod
    def empty(cls) -> DetectionBatch:
        return cls(
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.int64),
        )


@dataclass(frozen=True, slots=True)
class GroundTruthBatch:
    boxes_xyxy: FloatArray
    track_ids: IntArray
    class_ids: IntArray
    visibility: FloatArray
    valid: BoolArray
    ignore: BoolArray

    def __post_init__(self) -> None:
        boxes = _array(self.boxes_xyxy, np.dtype(np.float32), (4,), "boxes_xyxy")
        vectors: dict[str, NDArray[Any]] = {
            "track_ids": np.array(self.track_ids, dtype=np.int64, copy=True),
            "class_ids": np.array(self.class_ids, dtype=np.int64, copy=True),
            "visibility": np.array(self.visibility, dtype=np.float32, copy=True),
            "valid": np.array(self.valid, dtype=np.bool_, copy=True),
            "ignore": np.array(self.ignore, dtype=np.bool_, copy=True),
        }
        if any(value.shape != (len(boxes),) for value in vectors.values()):
            raise ValueError("ground-truth vectors must have shape (N,)")
        if not np.all(np.isfinite(vectors["visibility"])) or np.any(
            (vectors["visibility"] < 0) | (vectors["visibility"] > 1)
        ):
            raise ValueError("visibility must be finite values in [0, 1]")
        if np.any(vectors["valid"] & vectors["ignore"]):
            raise ValueError("ground truth cannot be both valid and ignored")
        if len(np.unique(vectors["track_ids"])) != len(boxes):
            raise ValueError("ground-truth track_ids must be unique within a frame")
        for value in vectors.values():
            value.setflags(write=False)
        object.__setattr__(self, "boxes_xyxy", boxes)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    @classmethod
    def empty(cls) -> GroundTruthBatch:
        return cls(
            np.empty((0, 4), np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            np.empty(0, np.bool_),
            np.empty(0, np.bool_),
        )


@dataclass(frozen=True, slots=True)
class TrackBatch:
    boxes_xyxy: FloatArray
    track_ids: IntArray
    class_ids: IntArray
    scores: FloatArray

    def __post_init__(self) -> None:
        boxes = _array(self.boxes_xyxy, np.dtype(np.float32), (4,), "boxes_xyxy")
        track_ids = np.array(self.track_ids, dtype=np.int64, copy=True)
        class_ids = np.array(self.class_ids, dtype=np.int64, copy=True)
        scores = np.array(self.scores, dtype=np.float32, copy=True)
        if any(value.shape != (len(boxes),) for value in (track_ids, class_ids, scores)):
            raise ValueError("track vectors must have shape (N,)")
        if np.any(track_ids < 0) or len(np.unique(track_ids)) != len(boxes):
            raise ValueError("emitted track_ids must be unique and nonnegative")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("track scores must be finite values in [0, 1]")
        for value in (track_ids, class_ids, scores):
            value.setflags(write=False)
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "scores", scores)

    def __len__(self) -> int:
        return len(self.boxes_xyxy)

    @classmethod
    def empty(cls) -> TrackBatch:
        return cls(
            np.empty((0, 4), np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.float32),
        )


@dataclass(frozen=True, slots=True)
class TrackerSummary:
    active_tracks: int
    confirmed_tracks: int
    stale_tracks: int
    mean_age_s: float
    mean_motion_px_s: float
    mean_confidence: float

    def __post_init__(self) -> None:
        if min(self.active_tracks, self.confirmed_tracks, self.stale_tracks) < 0:
            raise ValueError("tracker counts must be nonnegative")
        if self.confirmed_tracks > self.active_tracks or self.stale_tracks > self.active_tracks:
            raise ValueError("confirmed and stale counts cannot exceed active tracks")
        if not all(
            np.isfinite(value) and value >= 0
            for value in (self.mean_age_s, self.mean_motion_px_s)
        ):
            raise ValueError("tracker means must be finite and nonnegative")
        if not np.isfinite(self.mean_confidence) or not 0 <= self.mean_confidence <= 1:
            raise ValueError("mean_confidence must be in [0, 1]")

    @classmethod
    def empty(cls) -> TrackerSummary:
        return cls(0, 0, 0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ObservationScales:
    active_tracks: float
    age_s: float
    motion_px_s: float
    time_since_detector_s: float

    def __post_init__(self) -> None:
        values = (self.active_tracks, self.age_s, self.motion_px_s, self.time_since_detector_s)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("observation scales must be finite and positive")


@dataclass(frozen=True, slots=True)
class PolicyContext:
    nominal_rate: float
    source_fps: float
    reserve_ms: float
    seed: int


@runtime_checkable
class Tracker(Protocol):
    def reset(self) -> None:
        pass

    def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
        pass

    def summary(self) -> TrackerSummary:
        pass
