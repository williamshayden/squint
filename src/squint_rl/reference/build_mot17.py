"""CPU-only construction of immutable MOT17 replay traces."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from squint_rl.budget import BudgetConfig, TokenBucket
from squint_rl.reference.bytetrack import ByteTrackAdapter
from squint_rl.reference.dfine import (
    MODEL_ID,
    MODEL_REVISION,
    THRESHOLD,
    WEIGHTS_SHA256,
    scene_change_grid,
)
from squint_rl.reference.mot17 import Mot17Sequence
from squint_rl.tracker import DetectionBatch, TrackBatch, Tracker, TrackerSummary

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
_TRACE_SCHEMA_NAME = "squint.replay"
_TRACE_SCHEMA_VERSION = 1
_PROFILE_SCHEMA_NAME = "squint.reference-profile"
_PROFILE_SCHEMA_VERSION = 1
_PROFILE_HASH_DOMAIN = "squint.reference-profile/v1"
_PROFILE_TRAINING_IDS = ("02", "04", "05", "10")
_PROFILE_RATES = (0.10, 0.25, 0.50, 0.75, 1.00)
_PROFILE_COST_FIELDS = ("unit", "p95_ms", "reserve_ms", "capacity_ms", "profile_sha256")
_PROFILE_NORMALIZATION_FIELDS = (
    "active_tracks", "age_s", "motion_px_s", "time_since_detector_s"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DETECTOR_FIELDS = {
    "model_id", "revision", "weights", "preprocessor", "threshold",
    "class_mapping", "precision", "timing",
}
_HARDWARE_FIELDS = {"platform", "runtime", "device"}
_PINNED_DETECTOR_BASE: dict[str, object] = {
    "model_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "weights": {"model.safetensors": WEIGHTS_SHA256},
    "preprocessor": {
        "class": "RTDetrImageProcessor", "height": 640, "width": 640,
        "do_pad": False, "use_fast": False,
    },
    "threshold": THRESHOLD,
    "class_mapping": {
        "source_label": "person", "source_label_id": 0, "output_class_id": 1,
    },
    "timing": {
        "protocol": "synchronized-forward-only-v1", "unit": "ms",
        "includes": ["model_forward"],
        "excludes": ["preprocess", "postprocess", "telemetry"],
    },
}
_CUDA_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
_PCI_BDF_RE = re.compile(
    r"(?:[0-9a-f]{4}|[0-9a-f]{8}):[0-9a-f]{2}:[01][0-9a-f]\.[0-7]\Z"
)


class Detector(Protocol):
    def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]: ...


class _TrackerFactory(Protocol):
    def __call__(self, *, frame_rate: float) -> Tracker: ...


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


def _freeze_json(value: object, *, path: str = "manifest") -> object:
    """Validate and recursively freeze a JSON value without retaining aliases."""
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings for JSON")  # noqa: TRY004
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        try:
            if not math.isfinite(float(value)):
                raise ValueError(f"{path} contains an overflowing JSON number")
        except OverflowError as error:
            raise ValueError(f"{path} contains an overflowing JSON number") from error
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return value
    raise ValueError(f"{path} must contain JSON values")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("value must have a finite JSON representation") from error


def _lowercase_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hash")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")  # noqa: TRY004
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return normalized


def _identity_object(
    value: object, name: str, fields: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} identity must be a JSON object")  # noqa: TRY004
    frozen = _freeze_json(value, path=name)
    identity = cast(Mapping[str, object], frozen)
    if set(identity) != fields:
        raise ValueError(f"{name} identity has missing or unexpected fields")
    return identity


def _nested_object(
    value: object, name: str, fields: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")  # noqa: TRY004
    nested = cast(Mapping[str, object], value)
    if set(nested) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")
    return nested


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _pinned_detector_identity(precision: object) -> Mapping[str, object]:
    if precision not in ("float32", "float16"):
        raise ValueError("detector.precision must be float32 or float16")
    return cast(
        Mapping[str, object],
        _freeze_json({**_PINNED_DETECTOR_BASE, "precision": precision}, path="detector"),
    )


def _validate_detector_identity(value: object) -> Mapping[str, object]:
    detector = _identity_object(value, "detector", _DETECTOR_FIELDS)
    expected = _pinned_detector_identity(detector["precision"])
    if _canonical_json(detector) != _canonical_json(expected):
        raise ValueError("detector identity does not match the pinned D-FINE contract")
    return detector


def _validate_hardware_identity(value: object) -> Mapping[str, object]:
    hardware = _identity_object(value, "hardware", _HARDWARE_FIELDS)
    platform = _nested_object(
        hardware["platform"], "hardware.platform", {"system", "machine", "python"}
    )
    runtime = _nested_object(
        hardware["runtime"],
        "hardware.runtime",
        {"torch", "transformers", "cuda", "driver"},
    )
    device = _nested_object(
        hardware["device"],
        "hardware.device",
        {"type", "name", "uuid", "pci_bus_id"},
    )
    for field in ("system", "machine", "python"):
        _nonempty_string(platform[field], f"hardware.platform.{field}")
    for field in ("torch", "transformers"):
        _nonempty_string(runtime[field], f"hardware.runtime.{field}")
    for field in ("cuda", "driver"):
        _nullable_string(runtime[field], f"hardware.runtime.{field}")
    device_type = _nonempty_string(device["type"], "hardware.device.type")
    if device_type not in ("cpu", "cuda"):
        raise ValueError("hardware.device.type must be cpu or cuda")
    _nonempty_string(device["name"], "hardware.device.name")
    for field in ("uuid", "pci_bus_id"):
        _nullable_string(device[field], f"hardware.device.{field}")
    optional_identity = (
        runtime["cuda"], runtime["driver"], device["uuid"], device["pci_bus_id"]
    )
    if device_type == "cpu" and any(item is not None for item in optional_identity):
        raise ValueError("CPU hardware identity requires null CUDA, driver, UUID, and PCI bus ID")
    if device_type == "cuda" and any(item is None for item in optional_identity):
        raise ValueError("CUDA hardware identity requires CUDA, driver, UUID, and PCI bus ID")
    if device_type == "cuda":
        uuid = cast(str, device["uuid"])
        pci_bus_id = cast(str, device["pci_bus_id"])
        if (
            not uuid.startswith("GPU-")
            or _CUDA_UUID_RE.fullmatch(uuid[4:]) is None
            or uuid[4:] != uuid[4:].lower()
        ):
            raise ValueError("hardware.device.uuid must be a canonical lowercase CUDA UUID")
        if _PCI_BDF_RE.fullmatch(pci_bus_id) is None:
            raise ValueError("hardware.device.pci_bus_id must be a canonical lowercase PCI BDF")
    return hardware


def _validate_identity_matrix(
    detector: Mapping[str, object], hardware: Mapping[str, object]
) -> None:
    device = cast(Mapping[str, object], hardware["device"])
    if device["type"] == "cpu" and detector["precision"] != "float32":
        raise ValueError("CPU detector.precision must be float32")


def _runtime_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} must be UTF-8 bytes or a string") from error
    if not isinstance(value, str):
        raise ValueError(f"{name} must be bytes or a string")  # noqa: TRY004
    normalized = value.strip().strip("\x00").strip()
    if not normalized or "\x00" in normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _cuda_uuid(value: object, *, nvml: bool) -> str:
    if not isinstance(value, (str, bytes)):
        value = str(value)
    text = _runtime_text(value, "NVML CUDA UUID" if nvml else "Torch CUDA UUID")
    if nvml:
        if not text.startswith("GPU-"):
            raise ValueError("NVML CUDA UUID must have exactly one GPU- prefix")
        text = text[4:]
    if _CUDA_UUID_RE.fullmatch(text) is None:
        raise ValueError("CUDA UUID must contain one canonical bare UUID")
    return text.lower()


def _pci_bus_id(value: object) -> str:
    normalized = _runtime_text(value, "NVML PCI bus ID").lower()
    if _PCI_BDF_RE.fullmatch(normalized) is None:
        raise ValueError("NVML PCI bus ID must be a canonical PCI BDF")
    return normalized


@dataclass(slots=True)
class _HardwareSession:
    hardware_identity: Mapping[str, object]
    _nvml: Any | None = None
    _handle: object | None = None
    available: bool = False
    error_code: str | None = None
    closed: bool = False

    @classmethod
    def create(
        cls,
        device: str,
        *,
        load: Callable[[str], Any] = import_module,
    ) -> _HardwareSession:
        cuda_match = (
            re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
            if isinstance(device, str)
            else None
        )
        if device not in ("cpu", "cuda") and cuda_match is None:
            raise ValueError("device must be cpu, cuda, or cuda:N")
        try:
            platform_runtime = load("platform")
            torch_runtime = load("torch")
            transformers_runtime = load("transformers")
            system = _runtime_text(platform_runtime.system(), "platform system")
            machine = _runtime_text(platform_runtime.machine(), "platform machine")
            python = _runtime_text(platform_runtime.python_version(), "Python version")
            processor = platform_runtime.processor()
            cpu_name = _runtime_text(processor or machine, "CPU device name")
            torch_version = _runtime_text(torch_runtime.__version__, "Torch version")
            transformers_version = _runtime_text(
                transformers_runtime.__version__, "Transformers version"
            )
        except Exception as error:
            raise RuntimeError("unable to establish runtime identity") from error
        identity: dict[str, object] = {
            "platform": {"system": system, "machine": machine, "python": python},
            "runtime": {
                "torch": torch_version, "transformers": transformers_version,
                "cuda": None, "driver": None,
            },
            "device": {"type": "cpu", "name": cpu_name, "uuid": None, "pci_bus_id": None},
        }
        if device == "cpu":
            return cls(_validate_hardware_identity(identity))

        initialized = False
        try:
            if not torch_runtime.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            index = torch_runtime.cuda.current_device() if device == "cuda" else int(device[5:])
            if type(index) is not int or index < 0:
                raise ValueError("selected CUDA ordinal must be a nonnegative integer")
            properties = torch_runtime.cuda.get_device_properties(index)
            name = _runtime_text(properties.name, "CUDA device name")
            torch_uuid = _cuda_uuid(properties.uuid, nvml=False)
            canonical_uuid = f"GPU-{torch_uuid}"
            cuda_version = _runtime_text(torch_runtime.version.cuda, "CUDA version")
            nvml_runtime = load("pynvml")
            nvml_runtime.nvmlInit()
            initialized = True
            handle = nvml_runtime.nvmlDeviceGetHandleByUUID(
                canonical_uuid.encode("ascii")
            )
            nvml_uuid = _cuda_uuid(nvml_runtime.nvmlDeviceGetUUID(handle), nvml=True)
            if nvml_uuid != torch_uuid:
                raise ValueError("Torch and NVML CUDA UUIDs do not match")
            pci_bus_id = _pci_bus_id(nvml_runtime.nvmlDeviceGetPciInfo(handle).busId)
            driver = _runtime_text(
                nvml_runtime.nvmlSystemGetDriverVersion(), "NVIDIA driver version"
            )
            cast(dict[str, object], identity["runtime"]).update(
                cuda=cuda_version, driver=driver
            )
            cast(dict[str, object], identity["device"]).update(
                type="cuda", name=name, uuid=canonical_uuid, pci_bus_id=pci_bus_id
            )
            return cls(
                _validate_hardware_identity(identity), nvml_runtime, handle, available=True
            )
        except BaseException as error:
            if initialized:
                with suppress(Exception):
                    nvml_runtime.nvmlShutdown()
            if isinstance(error, Exception):
                raise RuntimeError(  # noqa: TRY004
                    "unable to establish CUDA runtime identity"
                ) from error
            raise

    def sample(self) -> tuple[float, int] | None:
        nvml, handle = self._nvml, self._handle
        if nvml is None or handle is None or self.error_code is not None or self.closed:
            return None
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            used_vram = nvml.nvmlDeviceGetMemoryInfo(handle).used
        except Exception:  # noqa: BLE001 - dynamic telemetry is deliberately nonfatal
            self.error_code = "sample_failed"
            return None
        if (
            isinstance(utilization, (bool, str, bytes))
            or type(used_vram) is not int
            or used_vram < 0
        ):
            self.error_code = "invalid_sample"
            return None
        try:
            normalized_utilization = float(utilization)
            normalized_vram = float(used_vram)
        except Exception:  # noqa: BLE001 - malformed telemetry is nonfatal
            self.error_code = "invalid_sample"
            return None
        if (
            not math.isfinite(normalized_utilization)
            or not 0.0 <= normalized_utilization <= 100.0
            or not math.isfinite(normalized_vram)
        ):
            self.error_code = "invalid_sample"
            return None
        return normalized_utilization, used_vram

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._nvml is not None:
            self._nvml.nvmlShutdown()


def _validate_trace_manifest(
    sequence_id: str,
    arrays: Mapping[str, NDArray[Any]],
    manifest: Mapping[str, object],
) -> None:
    manifest_sequence_id = manifest.get("sequence_id")
    if not isinstance(manifest_sequence_id, str) or manifest_sequence_id != sequence_id:
        raise ValueError("manifest_fields sequence_id must match RawTrace sequence_id")
    frame_count = manifest.get("frame_count")
    if type(frame_count) is not int or frame_count != len(arrays["timestamps_s"]):
        raise ValueError("manifest_fields frame_count must exactly match array frame count")
    fps_value = manifest.get("fps")
    if (
        isinstance(fps_value, bool)
        or not isinstance(fps_value, (int, float))
        or not math.isfinite(float(fps_value))
        or float(fps_value) <= 0.0
    ):
        raise ValueError("manifest_fields fps must be a finite positive number")
    expected_timestamps = np.arange(frame_count, dtype=np.float64) / float(fps_value)
    if not np.array_equal(arrays["timestamps_s"], expected_timestamps):
        raise ValueError("timestamps_s must exactly equal arange(frame_count) / manifest fps")
    if "schema" in manifest and manifest["schema"] != _TRACE_SCHEMA_NAME:
        raise ValueError("manifest_fields schema must be squint.replay")
    if "schema_version" in manifest and (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != _TRACE_SCHEMA_VERSION
    ):
        raise ValueError("manifest_fields schema_version must be integer 1")


def _profile_hash_payload(
    detector: Mapping[str, object],
    hardware: Mapping[str, object],
    cost_profile: Mapping[str, object],
    normalization: Mapping[str, object],
    training_traces: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cost_without_hash = {
        key: value for key, value in cost_profile.items() if key != "profile_sha256"
    }
    return {
        "hash_domain": _PROFILE_HASH_DOMAIN,
        "schema": {"name": _PROFILE_SCHEMA_NAME, "version": _PROFILE_SCHEMA_VERSION},
        "detector": _jsonable(detector),
        "hardware": _jsonable(hardware),
        "cost_profile": _jsonable(cost_without_hash),
        "normalization": _jsonable(normalization),
        "training_traces": _jsonable(tuple(training_traces)),
    }


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
        _validate_trace_manifest(self.sequence_id, self.arrays, manifest)
        expected_hash = causal_trace_sha256(self.arrays)
        if manifest.get("causal_trace_sha256") != expected_hash:
            raise ValueError("manifest_fields causal_trace_sha256 does not match arrays")
        frozen_arrays = {
            name: _immutable_array(self.arrays[name]) for name in _ARRAY_NAMES
        }
        object.__setattr__(self, "arrays", MappingProxyType(frozen_arrays))
        object.__setattr__(self, "manifest_fields", _freeze_json(manifest))


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Immutable training-only detector, hardware, cost, and scale profile."""
    detector: Mapping[str, object]
    hardware: Mapping[str, object]
    cost_profile: Mapping[str, object]
    normalization: Mapping[str, object]
    training_traces: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        detector = _validate_detector_identity(self.detector)
        hardware = _validate_hardware_identity(self.hardware)
        _validate_identity_matrix(detector, hardware)
        cost = _freeze_json(self.cost_profile, path="cost_profile")
        normalization = _freeze_json(self.normalization, path="normalization")
        traces = tuple(_freeze_json(trace, path="training_traces") for trace in self.training_traces)
        if not isinstance(cost, Mapping):
            raise ValueError("cost_profile must be a JSON object")  # noqa: TRY004
        if not isinstance(normalization, Mapping):
            raise ValueError("normalization must be a JSON object")  # noqa: TRY004
        if any(not isinstance(trace, Mapping) for trace in traces):
            raise ValueError("training_traces must contain JSON objects")
        cost_mapping = cast(Mapping[str, object], cost)
        normalization_mapping = cast(Mapping[str, object], normalization)
        if set(cost_mapping) != set(_PROFILE_COST_FIELDS):
            raise ValueError("cost_profile has missing or unexpected fields")
        if cost_mapping["unit"] != "detector_ms":
            raise ValueError("cost_profile.unit must be detector_ms")
        p95 = _positive_number(cost_mapping["p95_ms"], "cost_profile.p95_ms")
        reserve = _positive_number(cost_mapping["reserve_ms"], "cost_profile.reserve_ms")
        capacity = _positive_number(cost_mapping["capacity_ms"], "cost_profile.capacity_ms")
        if p95 != reserve:
            raise ValueError("cost_profile.p95_ms must equal reserve_ms")
        if capacity != 2.0 * reserve:
            raise ValueError("cost_profile.capacity_ms must equal 2 * reserve_ms")
        profile_hash = _lowercase_sha256(
            cost_mapping["profile_sha256"], "cost_profile.profile_sha256"
        )
        if set(normalization_mapping) != set(_PROFILE_NORMALIZATION_FIELDS):
            raise ValueError("normalization has missing or unexpected fields")
        active_tracks = normalization_mapping["active_tracks"]
        if isinstance(active_tracks, bool) or not isinstance(active_tracks, int):
            raise ValueError("normalization.active_tracks must be a positive integer")  # noqa: TRY004
        if active_tracks <= 0:
            raise ValueError("normalization.active_tracks must be a positive integer")
        for name in _PROFILE_NORMALIZATION_FIELDS[1:]:
            _positive_number(normalization_mapping[name], f"normalization.{name}")
        if len(traces) != len(_PROFILE_TRAINING_IDS):
            raise ValueError("training_traces must contain exactly 02, 04, 05, and 10")
        expected_ids = list(_PROFILE_TRAINING_IDS)
        actual_ids: list[str] = []
        for index, trace in enumerate(traces):
            trace_mapping = cast(Mapping[str, object], trace)
            if set(trace_mapping) != {"sequence_id", "causal_trace_sha256"}:
                raise ValueError("training_traces have missing or unexpected fields")
            identifier = trace_mapping["sequence_id"]
            if not isinstance(identifier, str):
                raise ValueError("training trace sequence_id must be a string")  # noqa: TRY004
            actual_ids.append(identifier)
            _lowercase_sha256(
                trace_mapping["causal_trace_sha256"],
                f"training_traces[{index}].causal_trace_sha256",
            )
        if actual_ids != expected_ids:
            raise ValueError("training_traces must be ordered exactly as 02, 04, 05, and 10")

        hash_payload = _profile_hash_payload(
            detector, hardware, cost_mapping, normalization_mapping,
            cast(tuple[Mapping[str, object], ...], traces),
        )
        expected_profile_hash = hashlib.sha256(_canonical_json(hash_payload).encode()).hexdigest()
        if profile_hash != expected_profile_hash:
            raise ValueError("cost_profile.profile_sha256 does not match canonical profile")
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "hardware", hardware)
        object.__setattr__(self, "cost_profile", cost)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "training_traces", cast(tuple[Mapping[str, object], ...], traces))

    @property
    def profile_sha256(self) -> str:
        return cast(str, self.cost_profile["profile_sha256"])

    @property
    def detector_identity(self) -> Mapping[str, object]:
        return self.detector

    @property
    def hardware_identity(self) -> Mapping[str, object]:
        return self.hardware

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": {"name": _PROFILE_SCHEMA_NAME, "version": _PROFILE_SCHEMA_VERSION},
            "detector": _jsonable(self.detector),
            "hardware": _jsonable(self.hardware),
            "cost_profile": _jsonable(self.cost_profile),
            "normalization": _jsonable(self.normalization),
            "training_traces": _jsonable(self.training_traces),
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_json(self) -> str:
        return self.canonical_json

    def episode_manifest_fields(self) -> dict[str, object]:
        return {
            "cost_profile": _jsonable(self.cost_profile),
            "normalization": _jsonable(self.normalization),
        }

    def cost_profile_manifest(self) -> dict[str, object]:
        return cast(dict[str, object], _jsonable(self.cost_profile))

    def normalization_manifest(self) -> dict[str, object]:
        return cast(dict[str, object], _jsonable(self.normalization))

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(self.canonical_json)

    @classmethod
    def load(cls, path: str | Path) -> ReferenceProfile:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to load reference profile {source}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("reference profile must be a JSON object")  # noqa: TRY004
        expected_fields = {
            "schema", "detector", "hardware", "cost_profile", "normalization", "training_traces"
        }
        if set(payload) != expected_fields:
            raise ValueError("reference profile has missing or unexpected fields")
        schema = payload["schema"]
        if not isinstance(schema, Mapping) or set(schema) != {"name", "version"}:
            raise ValueError("reference profile schema must contain name and version")
        version = schema.get("version")
        if schema.get("name") != _PROFILE_SCHEMA_NAME:
            raise ValueError("unsupported reference profile schema name")
        if type(version) is not int or version != _PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported reference profile schema version")
        training_traces = payload["training_traces"]
        if not isinstance(training_traces, list):
            raise ValueError("reference profile training_traces must be a JSON array")  # noqa: TRY004
        profile = cls(
            cast(Mapping[str, object], payload["detector"]),
            cast(Mapping[str, object], payload["hardware"]),
            cast(Mapping[str, object], payload["cost_profile"]),
            cast(Mapping[str, object], payload["normalization"]),
            tuple(cast(Sequence[Mapping[str, object]], training_traces)),
        )
        if text != profile.canonical_json:
            raise ValueError("reference profile JSON is not canonical")
        return profile


def _profile_percentile(
    values: Sequence[object],
    percentile: float,
    *,
    expected_count: int,
    field: str,
    positive: bool = False,
) -> float:
    try:
        domain = np.asarray(values, dtype=np.float64)
        if domain.shape != (expected_count,) or expected_count == 0:
            raise ValueError("wrong domain cardinality")
        if not np.all(np.isfinite(domain)) or np.any(domain < 0.0):
            raise ValueError("nonfinite or negative domain value")
        result = float(np.percentile(domain, percentile, method="linear"))
    except (FloatingPointError, IndexError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{field} requires exactly {expected_count} finite nonnegative samples"
        ) from error
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "strictly positive" if positive else "nonnegative"
        raise ValueError(f"{field} percentile must be finite and {qualifier}")
    return result


def _summary_sample(value: object, field: str, *, integer: bool = False) -> float:
    valid_type = isinstance(value, (int, np.integer)) if integer else isinstance(
        value, (int, float, np.integer, np.floating)
    )
    if isinstance(value, (bool, np.bool_)) or not valid_type:
        raise ValueError(f"normalization.{field} requires numeric tracker summaries")
    try:
        result = float(cast(Any, value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"normalization.{field} requires numeric tracker summaries") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"normalization.{field} requires finite nonnegative samples")
    return result


def _frame_detections(trace: RawTrace, frame_index: int) -> DetectionBatch:
    arrays = trace.arrays
    try:
        offsets = arrays["det_frame_offsets"]
        start, stop = int(offsets[frame_index]), int(offsets[frame_index + 1])
        return DetectionBatch(
            arrays["det_boxes_xyxy"][start:stop],
            arrays["det_scores"][start:stop],
            arrays["det_class_ids"][start:stop],
        )
    except (IndexError, OverflowError, TypeError, ValueError) as error:
        raise ValueError("normalization.active_tracks detector slices are invalid") from error


def _run_profile_schedule(
    trace: RawTrace,
    tracker: Tracker,
    *,
    budget: BudgetConfig | None,
    context: str,
    samples: dict[str, list[float]],
) -> None:
    timestamps = trace.arrays["timestamps_s"]
    bucket = TokenBucket(budget) if budget is not None else None
    if bucket is not None:
        bucket.reset(timestamp_s=float(timestamps[0]))
    last_detector_timestamp_s: float | None = None
    for frame_index in range(len(timestamps)):
        try:
            timestamp_s = float(timestamps[frame_index])
        except (IndexError, OverflowError, TypeError, ValueError) as error:
            raise ValueError("normalization.time_since_detector_s timestamps are invalid") from error
        if frame_index > 0 and bucket is not None:
            bucket.refill(timestamp_s=timestamp_s)
        try:
            summary = tracker.summary()
        except Exception as error:
            raise ValueError(f"{context} summary failed at frame {frame_index}") from error
        if not isinstance(summary, TrackerSummary):
            raise ValueError(  # noqa: TRY004
                f"{context} summary returned invalid TrackerSummary"
            )
        samples["active_tracks"].append(
            _summary_sample(summary.active_tracks, "active_tracks", integer=True)
        )
        samples["age_s"].append(_summary_sample(summary.mean_age_s, "age_s"))
        samples["motion_px_s"].append(
            _summary_sample(summary.mean_motion_px_s, "motion_px_s")
        )
        if last_detector_timestamp_s is not None:
            samples["time_since_detector_s"].append(
                timestamp_s - last_detector_timestamp_s
            )
        run_detector = bucket is None or bucket.affordable
        detections = _frame_detections(trace, frame_index) if run_detector else None
        try:
            tracks = tracker.step(detections, timestamp_s)
        except Exception as error:
            raise ValueError(f"{context} step failed at frame {frame_index}") from error
        if not isinstance(tracks, TrackBatch):
            raise ValueError(  # noqa: TRY004
                f"{context} step returned invalid TrackBatch"
            )
        if run_detector:
            if bucket is not None:
                try:
                    bucket.charge(
                        float(trace.arrays["detector_latency_ms"][frame_index])
                    )
                except (IndexError, OverflowError, TypeError, ValueError) as error:
                    raise ValueError("cost_profile.p95_ms latency domain is invalid") from error
            last_detector_timestamp_s = timestamp_s


def profile_training_traces(
    traces: Sequence[RawTrace],
    *,
    tracker_factory: _TrackerFactory = ByteTrackAdapter,
) -> ReferenceProfile:
    """Derive one causal detector-budget profile from canonical training traces."""
    if len(traces) != len(_PROFILE_TRAINING_IDS):
        raise ValueError("profile requires exactly training traces 02, 04, 05, and 10")
    if len({trace.sequence_id for trace in traces}) != len(traces):
        raise ValueError("profile training traces must not contain duplicates")
    if {trace.sequence_id for trace in traces} != set(_PROFILE_TRAINING_IDS):
        raise ValueError("profile requires exactly training traces 02, 04, 05, and 10")
    ordered = sorted(traces, key=lambda trace: _PROFILE_TRAINING_IDS.index(trace.sequence_id))
    first_manifest = ordered[0].manifest_fields
    detector = _validate_detector_identity(first_manifest.get("detector"))
    hardware = _validate_hardware_identity(first_manifest.get("hardware"))
    _validate_identity_matrix(detector, hardware)
    detector_json = _canonical_json(detector)
    hardware_json = _canonical_json(hardware)
    profile_traces: list[Mapping[str, object]] = []
    frame_counts: list[int] = []
    frame_rates: list[float] = []
    latency_domain: list[object] = []
    for trace in ordered:
        current_detector = _validate_detector_identity(trace.manifest_fields.get("detector"))
        current_hardware = _validate_hardware_identity(trace.manifest_fields.get("hardware"))
        _validate_identity_matrix(current_detector, current_hardware)
        if _canonical_json(current_detector) != detector_json:
            raise ValueError("training traces must share one detector identity")
        if _canonical_json(current_hardware) != hardware_json:
            raise ValueError("training traces must share one hardware identity")
        causal_hash = _lowercase_sha256(
            trace.manifest_fields.get("causal_trace_sha256"),
            f"trace {trace.sequence_id}.causal_trace_sha256",
        )
        profile_traces.append(
            {"sequence_id": trace.sequence_id, "causal_trace_sha256": causal_hash}
        )
        fps = _positive_number(
            trace.manifest_fields.get("fps"), f"trace {trace.sequence_id}.fps"
        )
        timestamps = trace.arrays["timestamps_s"]
        latencies = trace.arrays["detector_latency_ms"]
        frame_counts.append(len(timestamps))
        frame_rates.append(fps)
        latency_domain.extend(latencies)

    total_frames = sum(frame_counts)
    expected_summary_count = (len(_PROFILE_RATES) + 1) * total_frames
    expected_time_count = (len(_PROFILE_RATES) + 1) * sum(
        frame_count - 1 for frame_count in frame_counts
    )
    if expected_time_count == 0:
        raise ValueError("normalization.time_since_detector_s domain must be nonempty")
    reserve = _profile_percentile(
        latency_domain,
        95.0,
        expected_count=total_frames,
        field="cost_profile.p95_ms",
        positive=True,
    )
    budgets = [
        [
            BudgetConfig.for_rate(
                reserve_ms=reserve, source_fps=fps, nominal_rate=rate
            )
            for rate in _PROFILE_RATES
        ]
        for fps in frame_rates
    ]
    samples: dict[str, list[float]] = {
        name: [] for name in _PROFILE_NORMALIZATION_FIELDS
    }
    tracker_instances: list[Tracker] = []
    schedule_names = ("all-frame", *(f"greedy-{rate:.2f}" for rate in _PROFILE_RATES))
    for trace, fps, trace_budgets in zip(
        ordered, frame_rates, budgets, strict=True
    ):
        for schedule, budget in zip(
            schedule_names, (None, *trace_budgets), strict=True
        ):
            context = f"trace {trace.sequence_id} schedule {schedule}"
            try:
                tracker = tracker_factory(frame_rate=fps)
            except Exception as error:
                raise ValueError(f"{context} factory failed") from error
            if not isinstance(tracker, Tracker):
                raise ValueError(  # noqa: TRY004
                    f"{context} factory returned invalid Tracker"
                )
            if any(tracker is previous for previous in tracker_instances):
                raise ValueError(f"{context} factory reused a Tracker")
            tracker_instances.append(tracker)
            try:
                tracker.reset()
            except Exception as error:
                raise ValueError(f"{context} reset failed") from error
            _run_profile_schedule(
                trace,
                tracker,
                budget=budget,
                context=context,
                samples=samples,
            )

    active_tracks = _profile_percentile(
        samples["active_tracks"],
        99.0,
        expected_count=expected_summary_count,
        field="normalization.active_tracks",
    )
    age_s = _profile_percentile(
        samples["age_s"],
        99.0,
        expected_count=expected_summary_count,
        field="normalization.age_s",
    )
    motion_px_s = _profile_percentile(
        samples["motion_px_s"],
        99.0,
        expected_count=expected_summary_count,
        field="normalization.motion_px_s",
    )
    time_since_detector_s = _profile_percentile(
        samples["time_since_detector_s"],
        99.0,
        expected_count=expected_time_count,
        field="normalization.time_since_detector_s",
    )
    largest_frame_interval = max(1.0 / fps for fps in frame_rates)
    normalization: Mapping[str, object] = {
        "active_tracks": max(1, math.ceil(active_tracks)),
        "age_s": float(max(largest_frame_interval, age_s)),
        "motion_px_s": float(max(1.0, motion_px_s)),
        "time_since_detector_s": float(
            max(largest_frame_interval, time_since_detector_s)
        ),
    }
    cost_without_hash: dict[str, object] = {
        "unit": "detector_ms",
        "p95_ms": reserve,
        "reserve_ms": reserve,
        "capacity_ms": 2.0 * reserve,
    }
    hash_payload = _profile_hash_payload(
        detector, hardware, cost_without_hash, normalization, profile_traces
    )
    cost = dict(cost_without_hash)
    cost["profile_sha256"] = hashlib.sha256(_canonical_json(hash_payload).encode()).hexdigest()
    return ReferenceProfile(detector, hardware, cost, normalization, tuple(profile_traces))


def _telemetry_statistics(values: Sequence[float | int]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.sum(array / len(array), dtype=np.float64)),
        "p95": float(np.percentile(array, 95.0, method="linear")),
        "max": float(np.max(array)),
    }


def build_sequence(
    sequence: Mot17Sequence,
    detector: Detector,
    warmup_frames: int = 10,
    *,
    detector_identity: Mapping[str, object],
    hardware_identity: Mapping[str, object],
    telemetry_session: _HardwareSession | None = None,
) -> RawTrace:
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int):
        raise TypeError("warmup_frames must be an integer")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be nonnegative")
    validated_detector = _validate_detector_identity(detector_identity)
    validated_hardware = _validate_hardware_identity(hardware_identity)
    _validate_identity_matrix(validated_detector, validated_hardware)
    if telemetry_session is not None:
        if not isinstance(telemetry_session, _HardwareSession):
            raise ValueError("telemetry session must be a HardwareSession")
        if telemetry_session.closed:
            raise ValueError("telemetry session is closed")
        if _canonical_json(telemetry_session.hardware_identity) != _canonical_json(
            validated_hardware
        ):
            raise ValueError("telemetry session hardware identity does not match trace hardware")
    for image_path in sequence.image_paths[:warmup_frames]:
        with Image.open(image_path) as opened:
            detector.predict(opened.convert("RGB"))

    previous: Image.Image | None = None
    detections: list[DetectionBatch] = []
    scenes: list[NDArray[Any]] = []
    latencies: list[float] = []
    utilization_samples: list[float] = []
    vram_samples: list[int] = []
    for image_path in sequence.image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            scenes.append(scene_change_grid(previous, image))
            batch, latency = detector.predict(image)
            sample = telemetry_session.sample() if telemetry_session is not None else None
            if sample is not None:
                utilization, used_vram = sample
                utilization_samples.append(utilization)
                vram_samples.append(used_vram)
            detections.append(batch)
            latencies.append(float(latency))
            previous = image.copy()
    packed = pack_episode_arrays(sequence, detections, scenes, latencies)
    telemetry = {
        "nvml": {
            "available": telemetry_session.available if telemetry_session is not None else False,
            "error": telemetry_session.error_code if telemetry_session is not None else None,
        },
        "sample_count": len(utilization_samples),
        "gpu_utilization_percent": _telemetry_statistics(utilization_samples),
        "used_vram_bytes": _telemetry_statistics(vram_samples),
    }
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
        "detector": validated_detector,
        "hardware": validated_hardware,
        "telemetry": telemetry,
    }
    return RawTrace(sequence.identifier, packed, manifest)
