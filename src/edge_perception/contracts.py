"""Backend-neutral immutable records shared across the checkpoint."""

from dataclasses import dataclass
from math import isfinite


def _normalize_finite_float(value: float, field_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        message = f"{field_name} must be a finite number"
        raise TypeError(message) from error
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return str(value)


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x1", _normalize_finite_float(self.x1, "x1"))
        object.__setattr__(self, "y1", _normalize_finite_float(self.y1, "y1"))
        object.__setattr__(self, "x2", _normalize_finite_float(self.x2, "x2"))
        object.__setattr__(self, "y2", _normalize_finite_float(self.y2, "y2"))
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _require_string(self.region_id, "region_id"))
        for field_name, value in (("x", self.x), ("y", self.y), ("width", self.width), ("height", self.height)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("region width and height must be positive")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "region_id": self.region_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    class_id: int
    score: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise TypeError("box must be a Box")
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool):
            raise TypeError("class_id must be an integer")
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "score", _normalize_finite_float(self.score, "score"))
        if self.label is not None:
            object.__setattr__(self, "label", _require_string(self.label, "label"))
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, object]:
        return {
            "box": self.box.to_list(),
            "class_id": self.class_id,
            "score": self.score,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class StageTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("preprocess_ms", self.preprocess_ms),
            ("inference_ms", self.inference_ms),
            ("postprocess_ms", self.postprocess_ms),
            ("total_ms", self.total_ms),
        ):
            normalized = _normalize_finite_float(value, field_name)
            object.__setattr__(self, field_name, normalized)
            if normalized < 0.0:
                raise ValueError(f"{field_name} must not be negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True, slots=True)
class BatchPrediction:
    detections: tuple[tuple[Detection, ...], ...]
    timing: StageTiming

    def __post_init__(self) -> None:
        if not isinstance(self.detections, tuple):
            raise TypeError("detections must be a tuple of detection tuples")
        for image_detections in self.detections:
            if not isinstance(image_detections, tuple):
                raise TypeError("detections must be a tuple of detection tuples")
            if not all(isinstance(detection, Detection) for detection in image_detections):
                raise TypeError("detections must contain Detection values")
        if not isinstance(self.timing, StageTiming):
            raise TypeError("timing must be a StageTiming")

    def to_dict(self) -> dict[str, object]:
        return {
            "detections": [
                [detection.to_dict() for detection in image_detections]
                for image_detections in self.detections
            ],
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DetectorIdentity:
    adapter: str
    model_id: str
    revision: str
    weights_sha256: str
    backend: str
    backend_version: str
    device: str
    dtype: str

    def __post_init__(self) -> None:
        for field_name in (
            "adapter",
            "model_id",
            "revision",
            "weights_sha256",
            "backend",
            "backend_version",
            "device",
            "dtype",
        ):
            object.__setattr__(self, field_name, _require_string(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter": self.adapter,
            "model_id": self.model_id,
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "device": self.device,
            "dtype": self.dtype,
        }
