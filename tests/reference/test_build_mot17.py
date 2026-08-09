from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from math import nan
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from squint_rl.reference import build_mot17 as build_mot17_module
from squint_rl.reference.build_mot17 import (
    RawTrace,
    ReferenceProfile,
    build_sequence,
    canonical_source_sha256,
    causal_trace_sha256,
    pack_episode_arrays,
    profile_training_traces,
)
from squint_rl.reference.mot17 import Mot17Sequence
from squint_rl.tracker import (
    DetectionBatch,
    GroundTruthBatch,
    TrackBatch,
    TrackerSummary,
)

_FAKE_UUID_BARE = "01234567-89ab-cdef-0123-456789abcdef"


def _sequence(
    tmp_path: Path, frame_count: int = 3, identifier: str = "02"
) -> Mot17Sequence:
    source = tmp_path / f"MOT17-{identifier}-FRCNN"
    image_dir = source / "img1"
    (source / "gt").mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    (source / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-02-FRCNN\n", encoding="utf-8")
    (source / "gt" / "gt.txt").write_text("canonical gt\n", encoding="utf-8")
    paths = []
    for index in range(frame_count):
        path = image_dir / f"{index + 1:06d}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(path, format="PNG")
        paths.append(path)
    ground_truth = tuple(
        GroundTruthBatch(
            np.array([[index, 1, index + 2, 3]], np.float32),
            np.array([index + 1], np.int64),
            np.array([1], np.int64),
            np.array([0.5], np.float32),
            np.array([True]),
            np.array([False]),
        )
        for index in range(frame_count)
    )
    return Mot17Sequence(identifier, source, tuple(paths), 4, 4, 25.0, ground_truth)


def _detections(count: int) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            np.array([[index, 0, index + 1, 2]], np.float32),
            np.array([min(0.99, 0.25 + index / 10)], np.float32),
            np.array([1], np.int64),
        )
        for index in range(count)
    )


def _detector_identity() -> dict[str, object]:
    return {
        "model_id": "ustc-community/dfine-nano-coco",
        "revision": "066438d3d8f0da137a37b38fdf3368fd4afceced",
        "weights": {
            "model.safetensors": (
                "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
            )
        },
        "preprocessor": {
            "class": "RTDetrImageProcessor",
            "height": 640,
            "width": 640,
            "do_pad": False,
            "use_fast": False,
        },
        "threshold": 0.10,
        "class_mapping": {
            "source_label": "person",
            "source_label_id": 0,
            "output_class_id": 1,
        },
        "precision": "float32",
        "timing": {
            "protocol": "synchronized-forward-only-v1",
            "unit": "ms",
            "includes": ["model_forward"],
            "excludes": ["preprocess", "postprocess", "telemetry"],
        },
    }


def _hardware_identity(*, device_type: str = "cpu") -> dict[str, object]:
    accelerated = device_type != "cpu"
    return {
        "platform": {"system": "Linux", "machine": "x86_64", "python": "3.10.20"},
        "runtime": {
            "torch": "2.6.0",
            "transformers": "4.57.6",
            "cuda": "12.4" if accelerated else None,
            "driver": "550.54" if accelerated else None,
        },
        "device": {
            "type": device_type,
            "name": "Fixture accelerator" if accelerated else "Fixture CPU",
            "uuid": f"GPU-{_FAKE_UUID_BARE}" if accelerated else None,
            "pci_bus_id": "0000:01:00.0" if accelerated else None,
        },
    }


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


class _FakePlatformRuntime:
    @staticmethod
    def system() -> str:
        return "Linux"

    @staticmethod
    def machine() -> str:
        return "x86_64"

    @staticmethod
    def python_version() -> str:
        return "3.10.20"

    @staticmethod
    def processor() -> str:
        return "Fixture CPU"


class _FakeCudaRuntime:
    def __init__(
        self,
        *,
        uuid: object | None = None,
        available: bool = True,
        current_device: int = 1,
    ) -> None:
        self.uuid = _FakeTorchUuid(_FAKE_UUID_BARE) if uuid is None else uuid
        self.available = available
        self.selected = current_device
        self.property_requests: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return self.selected

    def get_device_properties(self, index: int) -> SimpleNamespace:
        self.property_requests.append(index)
        return SimpleNamespace(name="Fixture accelerator", uuid=self.uuid)


class _FakeTorchUuid:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _ExplodingFloat:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error or RuntimeError("private conversion text")

    def __float__(self) -> float:
        raise self.error


class _FakeTorchRuntime:
    __version__ = "2.6.0+cu124"

    def __init__(self, cuda: _FakeCudaRuntime | None = None) -> None:
        self.cuda = _FakeCudaRuntime() if cuda is None else cuda
        self.version = SimpleNamespace(cuda="12.4")


class _FakeNvmlRuntime:
    def __init__(
        self,
        readings: Sequence[tuple[object, object] | Exception] = (),
        *,
        uuid: object | None = None,
        pci_bus_id: object = b"0000:01:00.0",
        driver: object = b"573.44",
        events: list[str] | None = None,
    ) -> None:
        self.readings = list(readings)
        self.uuid = f"GPU-{_FAKE_UUID_BARE}".encode("ascii") if uuid is None else uuid
        self.pci_bus_id = pci_bus_id
        self.driver = driver
        self.events = [] if events is None else events
        self.handle = object()
        self.initialized = 0
        self.shutdowns = 0
        self.handle_uuids: list[object] = []
        self.utilization_reads = 0
        self.memory_reads = 0
        self._reading_index = 0
        self._pending_vram: object | None = None

    def nvmlInit(self) -> None:
        self.initialized += 1

    def nvmlShutdown(self) -> None:
        self.shutdowns += 1

    def nvmlDeviceGetHandleByUUID(self, uuid: object) -> object:
        self.handle_uuids.append(uuid)
        return self.handle

    def nvmlDeviceGetUUID(self, handle: object) -> object:
        assert handle is self.handle
        return self.uuid

    def nvmlDeviceGetPciInfo(self, handle: object) -> SimpleNamespace:
        assert handle is self.handle
        return SimpleNamespace(busId=self.pci_bus_id)

    def nvmlSystemGetDriverVersion(self) -> object:
        return self.driver

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> SimpleNamespace:
        assert handle is self.handle
        self.utilization_reads += 1
        self.events.append("sample-utilization")
        reading = self.readings[self._reading_index]
        if isinstance(reading, Exception):
            self._reading_index += 1
            raise reading
        utilization, self._pending_vram = reading
        return SimpleNamespace(gpu=utilization)

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> SimpleNamespace:
        assert handle is self.handle
        self.memory_reads += 1
        self.events.append("sample-memory")
        used = self._pending_vram
        self._pending_vram = None
        self._reading_index += 1
        if isinstance(used, Exception):
            raise used
        return SimpleNamespace(used=used)


def _runtime_loader(
    nvml: _FakeNvmlRuntime,
    *,
    cuda: _FakeCudaRuntime | None = None,
    calls: list[str] | None = None,
) -> Any:
    modules = {
        "platform": _FakePlatformRuntime(),
        "torch": _FakeTorchRuntime(cuda),
        "transformers": SimpleNamespace(__version__="4.57.6"),
        "pynvml": nvml,
    }

    def load(name: str) -> object:
        if calls is not None:
            calls.append(name)
        return modules[name]

    return load


def _hardware_session(
    nvml: _FakeNvmlRuntime,
    *,
    device: str = "cuda:1",
    cuda: _FakeCudaRuntime | None = None,
) -> Any:
    return build_mot17_module._HardwareSession.create(
        device,
        load=_runtime_loader(nvml, cuda=cuda),
    )


class _ConstantDetector:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls = 0
        self.failure = failure

    def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]:
        del image
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return _detections(1)[0], 5.0


def _trace_with_session(
    tmp_path: Path,
    session: Any,
    *,
    frame_count: int = 3,
) -> RawTrace:
    return build_sequence(
        _sequence(tmp_path, frame_count=frame_count),
        _ConstantDetector(),
        warmup_frames=0,
        detector_identity=_detector_identity(),
        hardware_identity=session.hardware_identity,
        telemetry_session=session,
    )


def _trace_manifest(
    arrays: dict[str, np.ndarray[Any, Any]],
    *,
    identifier: str = "02",
    detector: dict[str, object] | None = None,
    hardware: dict[str, object] | None = None,
    fps: float = 25.0,
) -> dict[str, object]:
    return {
        "schema": "squint.replay",
        "schema_version": 1,
        "sequence_id": identifier,
        "frame_count": len(arrays["timestamps_s"]),
        "fps": fps,
        "source_sha256": "a" * 64,
        "causal_trace_sha256": causal_trace_sha256(arrays),
        "detector": _detector_identity() if detector is None else detector,
        "hardware": _hardware_identity() if hardware is None else hardware,
        "telemetry": {"gpu_utilization": 1.0},
    }


_DELETE = object()


def _mutate_identity(
    identity: dict[str, object], path: tuple[str, ...], value: object
) -> dict[str, object]:
    target = identity
    for name in path[:-1]:
        child = target[name]
        assert isinstance(child, dict)
        target = child
    if value is _DELETE:
        target.pop(path[-1])
    else:
        target[path[-1]] = value
    return identity


def test_pack_episode_arrays_uses_replay_v1_schema_and_offsets(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    arrays = pack_episode_arrays(
        sequence,
        _detections(3),
        [np.full((3, 3), index / 10, np.float64) for index in range(3)],
        [1, 2, 3],
    )
    assert tuple(arrays) == (
        "timestamps_s", "detector_latency_ms", "scene_change", "det_boxes_xyxy",
        "det_scores", "det_class_ids", "det_frame_offsets", "gt_boxes_xyxy",
        "gt_track_ids", "gt_class_ids", "gt_visibility", "gt_valid", "gt_ignore",
        "gt_frame_offsets",
    )
    assert arrays["timestamps_s"].dtype == np.float64
    assert arrays["timestamps_s"].tolist() == [0.0, 1 / 25, 2 / 25]
    assert arrays["detector_latency_ms"].dtype == np.float32
    assert arrays["scene_change"].dtype == np.float32
    assert arrays["det_frame_offsets"].dtype == np.int64
    assert arrays["gt_frame_offsets"].dtype == np.int64
    np.testing.assert_array_equal(arrays["det_frame_offsets"], [0, 1, 2, 3])
    np.testing.assert_array_equal(arrays["gt_frame_offsets"], [0, 1, 2, 3])


def test_hashes_have_causal_and_source_boundaries(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    arrays = pack_episode_arrays(sequence, _detections(3), [np.zeros((3, 3))] * 3, [1, 2, 3])
    causal = causal_trace_sha256(arrays)
    arrays["gt_boxes_xyxy"][0, 0] += 1
    assert causal_trace_sha256(arrays) == causal
    arrays["detector_latency_ms"][0] += 1
    assert causal_trace_sha256(arrays) != causal
    source = canonical_source_sha256(sequence)
    (sequence.source_dir / "gt" / "gt.txt").write_text("changed\n", encoding="utf-8")
    assert canonical_source_sha256(sequence) != source


def test_warmups_are_extra_and_the_complete_sequence_is_measured(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path, 4)

    class Detector:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]:
            self.calls.append(int(image.getpixel((0, 0))[0]))  # type: ignore[index]
            return _detections(1)[0], float(len(self.calls))

    detector = Detector()
    trace = build_sequence(
        sequence,
        detector,
        warmup_frames=2,
        detector_identity=_detector_identity(),
        hardware_identity=_hardware_identity(),
    )
    assert isinstance(trace, RawTrace)
    assert detector.calls == [0, 1, 0, 1, 2, 3]
    assert trace.arrays["timestamps_s"].tolist() == [0.0, 0.04, 0.08, 0.12]
    assert trace.arrays["detector_latency_ms"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert trace.arrays["scene_change"].shape == (4, 3, 3)
    assert trace.arrays["gt_track_ids"].tolist() == [1, 2, 3, 4]
    assert trace.manifest_fields["frame_count"] == 4
    assert _plain_json(trace.manifest_fields["detector"]) == _detector_identity()
    assert _plain_json(trace.manifest_fields["hardware"]) == _hardware_identity()
    assert _plain_json(trace.manifest_fields["telemetry"]) == {
        "nvml": {"available": False, "error": None},
        "sample_count": 0,
        "gpu_utilization_percent": {"mean": None, "p95": None, "max": None},
        "used_vram_bytes": {"mean": None, "p95": None, "max": None},
    }
    assert trace.manifest_fields["causal_trace_sha256"] == causal_trace_sha256(trace.arrays)
    with pytest.raises(TypeError):
        build_sequence(
            sequence,
            detector,
            warmup_frames=True,
            detector_identity=_detector_identity(),
            hardware_identity=_hardware_identity(),
        )
    with pytest.raises(ValueError):
        build_sequence(
            sequence,
            detector,
            warmup_frames=-1,
            detector_identity=_detector_identity(),
            hardware_identity=_hardware_identity(),
        )


@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_pinned_detector_identity_serializes_the_exact_reference_contract(
    precision: str,
) -> None:
    expected = _detector_identity()
    expected["precision"] = precision
    identity = build_mot17_module._pinned_detector_identity(precision)
    assert _plain_json(identity) == expected


def test_cpu_session_never_calls_nvml_and_has_a_complete_identity() -> None:
    nvml = _FakeNvmlRuntime()
    imports: list[str] = []
    session = build_mot17_module._HardwareSession.create(
        "cpu",
        load=_runtime_loader(nvml, calls=imports),
    )
    expected = _hardware_identity()
    runtime = expected["runtime"]
    assert isinstance(runtime, dict)
    runtime["torch"] = "2.6.0+cu124"
    assert _plain_json(session.hardware_identity) == expected
    assert nvml.initialized == 0
    assert nvml.handle_uuids == []
    assert imports == ["platform", "torch", "transformers"]
    session.close()
    assert nvml.shutdowns == 0


def test_cuda_session_binds_nvml_by_normalized_torch_uuid() -> None:
    cuda = _FakeCudaRuntime(uuid=f" {_FAKE_UUID_BARE.upper()}\x00 ".encode())
    nvml = _FakeNvmlRuntime(
        uuid=f"GPU-{_FAKE_UUID_BARE}".encode(),
        pci_bus_id=b"0000:01:00.0\x00",
        driver="573.44",
    )
    session = _hardware_session(nvml, device="cuda", cuda=cuda)
    assert cuda.property_requests == [1]
    assert nvml.handle_uuids == [f"GPU-{_FAKE_UUID_BARE}".encode("ascii")]
    assert _plain_json(session.hardware_identity) == _hardware_identity(
        device_type="cuda"
    ) | {
        "runtime": {
            "torch": "2.6.0+cu124",
            "transformers": "4.57.6",
            "cuda": "12.4",
            "driver": "573.44",
        }
    }


def test_cuda_session_normalizes_eight_digit_nvml_pci_domain() -> None:
    nvml = _FakeNvmlRuntime(pci_bus_id=b"00000000:0A:0B.3\x00")
    session = _hardware_session(nvml)
    hardware = _plain_json(session.hardware_identity)
    assert isinstance(hardware, dict)
    device = hardware["device"]
    assert isinstance(device, dict)
    assert device["pci_bus_id"] == "00000000:0a:0b.3"


def test_cuda_session_converts_torch_cuuuid_to_nvml_ascii_bytes() -> None:
    bare = "01234567-89ab-cdef-0123-456789abcdef"
    cuda = _FakeCudaRuntime(uuid=_FakeTorchUuid(bare.upper()))
    nvml = _FakeNvmlRuntime(uuid=f"GPU-{bare}".encode("ascii"))
    session = _hardware_session(nvml, cuda=cuda)
    assert nvml.handle_uuids == [f"GPU-{bare}".encode("ascii")]
    hardware = _plain_json(session.hardware_identity)
    assert isinstance(hardware, dict)
    device = hardware["device"]
    assert isinstance(device, dict)
    assert device["uuid"] == f"GPU-{bare}"


@pytest.mark.parametrize(
    "nvml_uuid",
    [
        b"01234567-89ab-cdef-0123-456789abcdef",
        b"GPU-GPU-01234567-89ab-cdef-0123-456789abcdef",
        b"GPU-fedcba98-7654-3210-fedc-ba9876543210",
    ],
)
def test_cuda_session_rejects_malformed_or_mismatched_nvml_uuid_and_shuts_down(
    nvml_uuid: bytes,
) -> None:
    bare = "01234567-89ab-cdef-0123-456789abcdef"
    nvml = _FakeNvmlRuntime(uuid=nvml_uuid)
    with pytest.raises(RuntimeError, match="runtime identity"):
        _hardware_session(nvml, cuda=_FakeCudaRuntime(uuid=_FakeTorchUuid(bare)))
    assert nvml.handle_uuids == [f"GPU-{bare}".encode("ascii")]
    assert nvml.shutdowns == 1


def test_cuda_session_rejects_malformed_torch_uuid_before_nvml_init() -> None:
    nvml = _FakeNvmlRuntime()
    with pytest.raises(RuntimeError, match="runtime identity"):
        _hardware_session(nvml, cuda=_FakeCudaRuntime(uuid=_FakeTorchUuid("not-a-uuid")))
    assert nvml.initialized == 0


def test_build_sequence_samples_after_each_measured_prediction_and_aggregates(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    nvml = _FakeNvmlRuntime(
        [(10, 100), (20, 200), (30, 300), (40, 400)], events=events
    )
    session = _hardware_session(nvml)

    class Detector:
        def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]:
            events.append(f"predict-{int(image.getpixel((0, 0))[0])}")  # type: ignore[index]
            return _detections(1)[0], 5.0

    trace = build_sequence(
        _sequence(tmp_path, frame_count=4),
        Detector(),
        warmup_frames=2,
        detector_identity=_detector_identity(),
        hardware_identity=session.hardware_identity,
        telemetry_session=session,
    )
    assert events == [
        "predict-0", "predict-1",
        "predict-0", "sample-utilization", "sample-memory",
        "predict-1", "sample-utilization", "sample-memory",
        "predict-2", "sample-utilization", "sample-memory",
        "predict-3", "sample-utilization", "sample-memory",
    ]
    telemetry = _plain_json(trace.manifest_fields["telemetry"])
    assert telemetry == {
        "nvml": {"available": True, "error": None},
        "sample_count": 4,
        "gpu_utilization_percent": {"mean": 25.0, "p95": 38.5, "max": 40.0},
        "used_vram_bytes": {
            "mean": 250.0,
            "p95": pytest.approx(385.0),
            "max": 400.0,
        },
    }
    assert type(telemetry["sample_count"]) is int  # type: ignore[index]
    for field in ("gpu_utilization_percent", "used_vram_bytes"):
        statistics = telemetry[field]  # type: ignore[index]
        assert isinstance(statistics, dict)
        assert all(type(value) is float for value in statistics.values())


def test_representable_vram_cannot_overflow_telemetry_aggregates(
    tmp_path: Path,
) -> None:
    used_vram = 10**308
    session = _hardware_session(_FakeNvmlRuntime([(10, used_vram)] * 3))
    trace = _trace_with_session(tmp_path, session)
    telemetry = _plain_json(trace.manifest_fields["telemetry"])
    assert isinstance(telemetry, dict)
    assert telemetry["used_vram_bytes"] == {
        "mean": float(used_vram),
        "p95": float(used_vram),
        "max": float(used_vram),
    }


@pytest.mark.parametrize("precision", ["bfloat16", "", True, None])
def test_pinned_detector_identity_rejects_invalid_precision(precision: object) -> None:
    with pytest.raises(ValueError, match="precision"):
        build_mot17_module._pinned_detector_identity(precision)


def test_build_sequence_rejects_invalid_identity_before_prediction(
    tmp_path: Path,
) -> None:
    detector = _ConstantDetector()
    float16 = _detector_identity()
    float16["precision"] = "float16"
    with pytest.raises(ValueError, match=r"CPU.*float32|float32.*CPU"):
        build_sequence(
            _sequence(tmp_path / "matrix"),
            detector,
            warmup_frames=0,
            detector_identity=float16,
            hardware_identity=_hardware_identity(),
        )
    with pytest.raises(ValueError, match="detector"):
        build_sequence(
            _sequence(tmp_path / "malformed"),
            detector,
            warmup_frames=0,
            detector_identity={},
            hardware_identity=_hardware_identity(),
        )
    assert detector.calls == 0


@pytest.mark.parametrize(
    ("uuid", "pci_bus_id", "driver"),
    [
        (b"GPU-fedcba98-7654-3210-fedc-ba9876543210", "0000:01:00.0", "573.44"),
        (f"GPU-{_FAKE_UUID_BARE}".encode(), "", "573.44"),
        (f"GPU-{_FAKE_UUID_BARE}".encode(), "0000:01:20.0", "573.44"),
        (f"GPU-{_FAKE_UUID_BARE}".encode(), "00000000:01:ff.0", "573.44"),
        (f"GPU-{_FAKE_UUID_BARE}".encode(), "0000:01:00.0", b"\x00"),
    ],
)
def test_cuda_static_identity_failure_is_contextual_and_shuts_down_once(
    uuid: object,
    pci_bus_id: object,
    driver: object,
) -> None:
    nvml = _FakeNvmlRuntime(uuid=uuid, pci_bus_id=pci_bus_id, driver=driver)
    with pytest.raises(RuntimeError, match="runtime identity"):
        _hardware_session(nvml)
    assert nvml.initialized == 1
    assert nvml.shutdowns == 1


@pytest.mark.parametrize("device", ["cuda:", "cuda:-1", "cuda:one", "cuda:1:2", "gpu"])
def test_hardware_session_rejects_unsupported_device_before_nvml(device: str) -> None:
    nvml = _FakeNvmlRuntime()
    with pytest.raises(ValueError, match="device"):
        _hardware_session(nvml, device=device)
    assert nvml.initialized == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Other GPU"),
        ("uuid", "GPU-fedcba98-7654-3210-fedc-ba9876543210"),
        ("pci_bus_id", "0000:02:00.0"),
    ],
)
def test_build_sequence_rejects_session_hardware_mismatch_before_prediction(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    nvml = _FakeNvmlRuntime([(10, 100)])
    session = _hardware_session(nvml)
    hardware = _plain_json(session.hardware_identity)
    assert isinstance(hardware, dict)
    device = hardware["device"]
    assert isinstance(device, dict)
    device[field] = value
    detector = _ConstantDetector()
    with pytest.raises(ValueError, match=r"telemetry session.*hardware"):
        build_sequence(
            _sequence(tmp_path),
            detector,
            warmup_frames=0,
            detector_identity=_detector_identity(),
            hardware_identity=hardware,
            telemetry_session=session,
        )
    assert detector.calls == 0
    assert nvml.utilization_reads == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uuid", _FAKE_UUID_BARE),
        ("uuid", f"GPU-GPU-{_FAKE_UUID_BARE}"),
        ("uuid", f"GPU-{_FAKE_UUID_BARE.upper()}"),
        ("uuid", "GPU-not-a-uuid"),
        ("pci_bus_id", "000:01:00.0"),
        ("pci_bus_id", "00000:01:00.0"),
        ("pci_bus_id", "0000:1:00.0"),
        ("pci_bus_id", "0000:01:0.0"),
        ("pci_bus_id", "0000:01:20.0"),
        ("pci_bus_id", "00000000:01:ff.0"),
        ("pci_bus_id", "0000:01:00.8"),
        ("pci_bus_id", "0000:0A:00.0"),
        ("pci_bus_id", " 0000:01:00.0"),
    ],
)
def test_build_sequence_rejects_noncanonical_cuda_identity_before_image_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    hardware = _hardware_identity(device_type="cuda")
    device = hardware["device"]
    assert isinstance(device, dict)
    device[field] = value
    detector = _ConstantDetector()

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity validation must precede image I/O")

    monkeypatch.setattr(Image, "open", fail_open)
    with pytest.raises(ValueError, match="hardware"):
        build_sequence(
            _sequence(tmp_path),
            detector,
            warmup_frames=0,
            detector_identity=_detector_identity(),
            hardware_identity=hardware,
        )
    assert detector.calls == 0


@pytest.mark.parametrize(
    "pci_bus_id",
    ["0000:0a:00.3", "0000:0a:1f.3", "00000000:0a:00.3", "00000000:0a:1f.3"],
)
def test_build_sequence_accepts_canonical_cuda_pci_domain_widths(
    tmp_path: Path,
    pci_bus_id: str,
) -> None:
    hardware = _hardware_identity(device_type="cuda")
    device = hardware["device"]
    assert isinstance(device, dict)
    device["pci_bus_id"] = pci_bus_id
    trace = build_sequence(
        _sequence(tmp_path),
        _ConstantDetector(),
        warmup_frames=0,
        detector_identity=_detector_identity(),
        hardware_identity=hardware,
    )
    manifest_hardware = _plain_json(trace.manifest_fields["hardware"])
    assert isinstance(manifest_hardware, dict)
    manifest_device = manifest_hardware["device"]
    assert isinstance(manifest_device, dict)
    assert manifest_device["pci_bus_id"] == pci_bus_id


@pytest.mark.parametrize(
    ("reading", "error_code", "memory_reads"),
    [
        (RuntimeError("private driver text"), "sample_failed", 0),
        ((nan, 100), "invalid_sample", 1),
        ((-1, 100), "invalid_sample", 1),
        ((101, 100), "invalid_sample", 1),
        ((True, 100), "invalid_sample", 1),
        (("10", 100), "invalid_sample", 1),
        ((10, -1), "invalid_sample", 1),
        ((10, True), "invalid_sample", 1),
        ((10, np.int64(100)), "invalid_sample", 1),
        ((10, RuntimeError("private memory text")), "sample_failed", 1),
        pytest.param(
            (_ExplodingFloat(), 100), "invalid_sample", 1,
            id="exceptional-utilization-conversion",
        ),
        pytest.param(
            (10, 10**10000), "invalid_sample", 1,
            id="unrepresentable-vram",
        ),
    ],
)
def test_invalid_dynamic_sample_is_discarded_and_disables_session(
    tmp_path: Path,
    reading: tuple[object, object] | Exception,
    error_code: str,
    memory_reads: int,
) -> None:
    nvml = _FakeNvmlRuntime([reading, (50, 500)])
    session = _hardware_session(nvml)
    trace = _trace_with_session(tmp_path, session)
    telemetry = _plain_json(trace.manifest_fields["telemetry"])
    assert telemetry == {
        "nvml": {"available": True, "error": error_code},
        "sample_count": 0,
        "gpu_utilization_percent": {"mean": None, "p95": None, "max": None},
        "used_vram_bytes": {"mean": None, "p95": None, "max": None},
    }
    assert nvml.utilization_reads == 1
    assert nvml.memory_reads == memory_reads
    assert "private" not in json.dumps(telemetry)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_dynamic_sample_does_not_swallow_base_exceptions(
    exception_type: type[BaseException],
) -> None:
    session = _hardware_session(
        _FakeNvmlRuntime([(_ExplodingFloat(exception_type()), 100)])
    )
    with pytest.raises(exception_type):
        session.sample()
    assert session.error_code is None


def test_partial_telemetry_is_retained_and_error_disables_later_traces(
    tmp_path: Path,
) -> None:
    nvml = _FakeNvmlRuntime(
        [(12, 120), RuntimeError("private driver text"), (99, 999)]
    )
    session = _hardware_session(nvml)
    first = _trace_with_session(tmp_path / "first", session, frame_count=4)
    first_telemetry = _plain_json(first.manifest_fields["telemetry"])
    assert first_telemetry == {
        "nvml": {"available": True, "error": "sample_failed"},
        "sample_count": 1,
        "gpu_utilization_percent": {"mean": 12.0, "p95": 12.0, "max": 12.0},
        "used_vram_bytes": {"mean": 120.0, "p95": 120.0, "max": 120.0},
    }
    assert (nvml.utilization_reads, nvml.memory_reads) == (2, 1)
    second = _trace_with_session(tmp_path / "second", session, frame_count=2)
    second_telemetry = _plain_json(second.manifest_fields["telemetry"])
    assert second_telemetry["nvml"] == {
        "available": True,
        "error": "sample_failed",
    }
    assert second_telemetry["sample_count"] == 0
    assert (nvml.utilization_reads, nvml.memory_reads) == (2, 1)


def test_hardware_session_closes_once_on_normal_and_error_exit() -> None:
    normal_nvml = _FakeNvmlRuntime()
    normal = _hardware_session(normal_nvml)
    normal.close()
    normal.close()
    assert normal_nvml.shutdowns == 1

    error_nvml = _FakeNvmlRuntime()
    failing = _hardware_session(error_nvml)
    with pytest.raises(LookupError, match="experiment failed"):
        try:
            raise LookupError("experiment failed")
        finally:
            failing.close()
    failing.close()
    assert error_nvml.shutdowns == 1


def test_pack_rejects_cardinality_and_nonfinite_inputs(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    with pytest.raises(ValueError, match="cardinality"):
        pack_episode_arrays(sequence, _detections(2), [np.zeros((3, 3))] * 3, [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        pack_episode_arrays(sequence, _detections(3), [np.zeros((3, 3))] * 3, [1, np.nan, 3])


@pytest.mark.parametrize(
    ("latencies", "scenes", "match"),
    [
        ([1.0, -0.1, 3.0], [np.zeros((3, 3))] * 3, "detector_latency_ms"),
        ([1.0, 2.0, 3.0], [np.full((3, 3), 1.01)] * 3, "scene_change"),
        ([1.0, 2.0, 3.0], [np.full((3, 3), -0.01)] * 3, "scene_change"),
    ],
)
def test_pack_rejects_negative_latency_and_scene_values_outside_unit_interval(
    tmp_path: Path,
    latencies: list[float],
    scenes: list[np.ndarray[Any, Any]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pack_episode_arrays(_sequence(tmp_path), _detections(3), scenes, latencies)


def test_empty_detection_and_ground_truth_offsets_cover_every_frame(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    sequence = replace(
        sequence,
        ground_truth=tuple(GroundTruthBatch.empty() for _ in sequence.image_paths),
    )
    arrays = pack_episode_arrays(
        sequence,
        tuple(DetectionBatch.empty() for _ in sequence.image_paths),
        [np.zeros((3, 3), np.float32)] * 3,
        [0.0, 0.0, 0.0],
    )
    assert arrays["det_boxes_xyxy"].shape == (0, 4)
    assert arrays["gt_boxes_xyxy"].shape == (0, 4)
    np.testing.assert_array_equal(arrays["det_frame_offsets"], [0, 0, 0, 0])
    np.testing.assert_array_equal(arrays["gt_frame_offsets"], [0, 0, 0, 0])


def test_raw_trace_arrays_are_non_bypassably_immutable_and_detached(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    source = pack_episode_arrays(
        sequence,
        _detections(3),
        [np.zeros((3, 3), np.float32)] * 3,
        [1.0, 2.0, 3.0],
    )
    expected_scores = source["det_scores"].copy()
    trace = RawTrace(
        sequence.identifier,
        source,
        _trace_manifest(source),
    )
    source["det_scores"][0] = 0.99
    np.testing.assert_array_equal(trace.arrays["det_scores"], expected_scores)
    for value in trace.arrays.values():
        assert not value.flags.owndata
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        trace.manifest_fields["new"] = "value"  # type: ignore[index]


def _valid_raw_trace_inputs(tmp_path: Path) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, object]]:
    arrays = pack_episode_arrays(
        _sequence(tmp_path),
        _detections(3),
        [np.zeros((3, 3), np.float32)] * 3,
        [1.0, 2.0, 3.0],
    )
    return arrays, _trace_manifest(arrays)


def _profile_trace(
    tmp_path: Path,
    identifier: str,
    *,
    detector: dict[str, object] | None = None,
    hardware: dict[str, object] | None = None,
    latency: float | Sequence[float] = 1.0,
    frame_count: int = 3,
    fps: float = 25.0,
) -> RawTrace:
    sequence = replace(
        _sequence(tmp_path / identifier, frame_count, identifier), fps=fps
    )
    latencies = (
        [float(latency)] * frame_count
        if isinstance(latency, (int, float))
        else [float(value) for value in latency]
    )
    arrays = pack_episode_arrays(
        sequence,
        _detections(frame_count),
        [np.zeros((3, 3), np.float32)] * frame_count,
        latencies,
    )
    manifest = _trace_manifest(
        arrays,
        identifier=identifier,
        detector=detector,
        hardware=hardware,
        fps=fps,
    )
    return RawTrace(identifier, arrays, manifest)


class _EmptyTracker:
    def reset(self) -> None:
        pass

    def step(
        self, detections: DetectionBatch | None, timestamp_s: float
    ) -> TrackBatch:
        del detections, timestamp_s
        return TrackBatch.empty()

    def summary(self) -> TrackerSummary:
        return TrackerSummary.empty()


def _empty_tracker_factory(*, frame_rate: float) -> _EmptyTracker:
    del frame_rate
    return _EmptyTracker()


def _profile(traces: Sequence[RawTrace]) -> ReferenceProfile:
    return profile_training_traces(traces, tracker_factory=_empty_tracker_factory)


class _RecordingTracker(_EmptyTracker):
    def __init__(self) -> None:
        self.reset_count = 0
        self.actions: list[int] = []
        self.events: list[str] = []

    def reset(self) -> None:
        self.reset_count += 1
        self.events.append("reset")

    def step(
        self, detections: DetectionBatch | None, timestamp_s: float
    ) -> TrackBatch:
        del timestamp_s
        action = int(detections is not None)
        self.actions.append(action)
        self.events.append("detect" if action else "skip")
        return TrackBatch.empty()

    def summary(self) -> TrackerSummary:
        self.events.append("summary")
        return TrackerSummary.empty()


class _RecordingFactory:
    def __init__(self) -> None:
        self.frame_rates: list[float] = []
        self.trackers: list[_RecordingTracker] = []

    def __call__(self, *, frame_rate: float) -> _RecordingTracker:
        tracker = _RecordingTracker()
        self.frame_rates.append(frame_rate)
        self.trackers.append(tracker)
        return tracker


class _SummaryFactory:
    def __init__(self) -> None:
        self.sample_count = 0

    def __call__(self, *, frame_rate: float) -> _EmptyTracker:
        del frame_rate
        owner = self

        class SummaryTracker(_EmptyTracker):
            def summary(self) -> TrackerSummary:
                value = owner.sample_count
                owner.sample_count += 1
                return TrackerSummary(value, 0, 0, value / 10, value * 2.0, 0.0)

        return SummaryTracker()


class _AdversarialTracker(_EmptyTracker):
    def __init__(self, case: str) -> None:
        self.case = case

    def reset(self) -> None:
        if self.case == "reset_raises":
            raise RuntimeError("reset exploded")

    def summary(self) -> Any:
        if self.case == "summary_raises":
            raise RuntimeError("summary exploded")
        if self.case == "summary_none":
            return None
        if self.case == "summary_wrong_type":
            return TrackBatch.empty()
        return super().summary()

    def step(
        self, detections: DetectionBatch | None, timestamp_s: float
    ) -> Any:
        if self.case == "step_raises":
            raise IndexError("step exploded")
        if self.case == "step_none":
            return None
        if self.case == "step_wrong_type":
            return TrackerSummary.empty()
        return super().step(detections, timestamp_s)


class _AdversarialFactory:
    def __init__(self, case: str) -> None:
        self.case = case

    def __call__(self, *, frame_rate: float) -> Any:
        del frame_rate
        if self.case == "factory_raises":
            raise RuntimeError("factory exploded")
        if self.case == "missing_protocol":
            return object()
        return _AdversarialTracker(self.case)


class _WrongSignatureFactory:
    def __call__(self) -> _EmptyTracker:
        return _EmptyTracker()


class _AccessTrap(Mapping[str, object]):
    def __init__(self, values: Mapping[str, object], allowed: set[str]) -> None:
        self._values = values
        self._allowed = allowed

    def __getitem__(self, key: str) -> object:
        if key not in self._allowed:
            raise AssertionError(f"forbidden profile access: {key}")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("profile construction must not iterate trace mappings")

    def __len__(self) -> int:
        raise AssertionError("profile construction must not size trace mappings")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sequence_id", _DELETE, "sequence_id"),
        ("sequence_id", "04", "sequence_id"),
        ("sequence_id", True, "sequence_id"),
        ("frame_count", _DELETE, "frame_count"),
        ("frame_count", 2, "frame_count"),
        ("frame_count", True, "frame_count"),
        ("fps", _DELETE, "fps"),
        ("fps", 30.0, "timestamps_s"),
        ("fps", 0.0, "fps"),
        ("fps", nan, "fps"),
        ("fps", True, "fps"),
        ("schema", "other.replay", "schema"),
        ("schema", True, "schema"),
        ("schema_version", 2, "schema_version"),
        ("schema_version", True, "schema_version"),
    ],
)
def test_raw_trace_rejects_incoherent_manifest_metadata(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    arrays, manifest = _valid_raw_trace_inputs(tmp_path)
    if value is _DELETE:
        manifest.pop(field)
    else:
        manifest[field] = value
    with pytest.raises(ValueError, match=match):
        RawTrace("02", arrays, manifest)


def test_raw_trace_requires_timestamps_exactly_derived_from_manifest_fps(
    tmp_path: Path,
) -> None:
    arrays, manifest = _valid_raw_trace_inputs(tmp_path)
    arrays["timestamps_s"] = arrays["timestamps_s"].copy()
    arrays["timestamps_s"][1] += 0.000001
    manifest["causal_trace_sha256"] = causal_trace_sha256(arrays)
    with pytest.raises(ValueError, match="timestamps_s"):
        RawTrace("02", arrays, manifest)


def test_raw_trace_schema_fields_are_optional_as_a_pair(tmp_path: Path) -> None:
    arrays, manifest = _valid_raw_trace_inputs(tmp_path)
    manifest.pop("schema")
    manifest.pop("schema_version")
    trace = RawTrace("02", arrays, manifest)
    assert "schema" not in trace.manifest_fields


def test_reference_profile_uses_explicit_canonical_schema_and_hash(tmp_path: Path) -> None:
    traces = [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    profile = _profile(traces)

    assert isinstance(profile, ReferenceProfile)
    assert profile.profile_sha256 == profile.cost_profile["profile_sha256"]
    assert profile.to_dict()["schema"] == {
        "name": "squint.reference-profile",
        "version": 1,
    }
    assert profile.canonical_json == profile.to_json()
    destination = tmp_path / "reference-profile.json"
    profile.write(destination)
    assert ReferenceProfile.load(destination).canonical_json == profile.canonical_json


def test_profile_derives_linear_p95_reserve_and_capacity(tmp_path: Path) -> None:
    traces = [
        _profile_trace(tmp_path, identifier, latency=latencies)
        for identifier, latencies in zip(
            ("02", "04", "05", "10"),
            ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)),
            strict=True,
        )
    ]

    profile = profile_training_traces(
        traces, tracker_factory=_empty_tracker_factory
    )

    assert profile.cost_profile["p95_ms"] == pytest.approx(11.45)
    assert profile.cost_profile["reserve_ms"] == pytest.approx(11.45)
    assert profile.cost_profile["capacity_ms"] == pytest.approx(22.9)


def test_profile_runs_exact_canonical_tracker_lifecycle_and_actions(
    tmp_path: Path,
) -> None:
    frame_count = 11
    traces = {
        identifier: _profile_trace(
            tmp_path,
            identifier,
            latency=([20.0] + [10.0] * 10 if identifier == "02" else 10.0),
            frame_count=frame_count,
            fps=fps,
        )
        for identifier, fps in zip(
            ("02", "04", "05", "10"), (8.0, 16.0, 32.0, 64.0), strict=True
        )
    }
    factory = _RecordingFactory()

    profile = profile_training_traces(
        [traces["10"], traces["02"], traces["05"], traces["04"]],
        tracker_factory=factory,
    )

    assert profile.cost_profile["reserve_ms"] == 10.0
    assert factory.frame_rates == [8.0] * 6 + [16.0] * 6 + [32.0] * 6 + [64.0] * 6
    assert len(factory.trackers) == len({id(tracker) for tracker in factory.trackers}) == 24
    first_trace_actions = [
        [1] * 11,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    normal_actions = [
        [1] * 11,
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1] * 11,
    ]
    expected_actions = first_trace_actions + normal_actions * 3
    assert [tracker.actions for tracker in factory.trackers] == expected_actions
    for tracker, actions in zip(factory.trackers, expected_actions, strict=True):
        expected_events = ["reset"]
        for action in actions:
            expected_events.extend(("summary", "detect" if action else "skip"))
        assert tracker.reset_count == 1
        assert tracker.events == expected_events


def test_profile_derives_p99_scales_from_exact_schedule_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import squint_rl.reference.build_mot17 as build_mot17_module

    calls: list[tuple[float, int, str]] = []
    real_percentile = np.percentile

    def recording_percentile(
        values: object, q: float, *, method: str
    ) -> np.floating[Any]:
        calls.append((float(q), int(np.asarray(values).size), method))
        return real_percentile(values, q, method=method)

    monkeypatch.setattr(build_mot17_module.np, "percentile", recording_percentile)
    factory = _SummaryFactory()
    profile = profile_training_traces(
        [
            _profile_trace(tmp_path, identifier, latency=100.0)
            for identifier in ("02", "04", "05", "10")
        ],
        tracker_factory=factory,
    )

    assert factory.sample_count == 72
    assert calls == [
        (95.0, 12, "linear"),
        (99.0, 72, "linear"),
        (99.0, 72, "linear"),
        (99.0, 72, "linear"),
        (99.0, 48, "linear"),
    ]
    assert profile.normalization == {
        "active_tracks": 71,
        "age_s": pytest.approx(7.029),
        "motion_px_s": pytest.approx(140.58),
        "time_since_detector_s": pytest.approx(0.08),
    }


def test_profile_applies_all_normalization_floors(tmp_path: Path) -> None:
    profile = _profile(
        [
            _profile_trace(tmp_path, identifier, latency=10.0, frame_count=2)
            for identifier in ("02", "04", "05", "10")
        ]
    )

    assert profile.normalization == {
        "active_tracks": 1,
        "age_s": pytest.approx(0.04),
        "motion_px_s": 1.0,
        "time_since_detector_s": pytest.approx(0.04),
    }


@pytest.mark.parametrize(
    ("frame_count", "latency", "match"),
    [
        (3, 0.0, r"cost_profile\.p95_ms"),
        (1, 1.0, r"normalization\.time_since_detector_s"),
    ],
)
def test_profile_rejects_invalid_domains_before_tracker_creation(
    tmp_path: Path, frame_count: int, latency: float, match: str
) -> None:
    factory = _RecordingFactory()

    with pytest.raises(ValueError, match=match):
        profile_training_traces(
            [
                _profile_trace(
                    tmp_path,
                    identifier,
                    latency=latency,
                    frame_count=frame_count,
                )
                for identifier in ("02", "04", "05", "10")
            ],
            tracker_factory=factory,
        )

    assert factory.trackers == []


@pytest.mark.parametrize(
    ("case", "operation"),
    [
        ("wrong_signature", "factory"),
        ("factory_raises", "factory"),
        ("missing_protocol", "factory"),
        ("reset_raises", "reset"),
        ("summary_raises", "summary"),
        ("summary_none", "summary"),
        ("summary_wrong_type", "summary"),
        ("step_raises", "step"),
        ("step_none", "step"),
        ("step_wrong_type", "step"),
    ],
)
def test_profile_wraps_malformed_tracker_boundaries_contextually(
    tmp_path: Path, case: str, operation: str
) -> None:
    factory: Any = (
        _WrongSignatureFactory()
        if case == "wrong_signature"
        else _AdversarialFactory(case)
    )
    traces = [
        _profile_trace(tmp_path, identifier, latency=10.0, frame_count=2)
        for identifier in ("02", "04", "05", "10")
    ]

    with pytest.raises(
        ValueError, match=rf"trace 02.*all-frame.*{operation}"
    ):
        profile_training_traces(traces, tracker_factory=factory)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_profile_does_not_wrap_process_control_exceptions(
    tmp_path: Path, error_type: type[BaseException]
) -> None:
    def factory(*, frame_rate: float) -> _EmptyTracker:
        del frame_rate
        raise error_type()

    with pytest.raises(error_type):
        profile_training_traces(
            [
                _profile_trace(tmp_path, identifier, latency=10.0, frame_count=2)
                for identifier in ("02", "04", "05", "10")
            ],
            tracker_factory=factory,
        )


def test_profile_default_bytetrack_factory_succeeds(tmp_path: Path) -> None:
    profile = profile_training_traces(
        [
            _profile_trace(tmp_path, identifier, latency=10.0, frame_count=2)
            for identifier in ("02", "04", "05", "10")
        ]
    )

    assert profile.cost_profile["reserve_ms"] == 10.0


def test_profile_is_order_independent_and_avoids_noncausal_trace_access(
    tmp_path: Path,
) -> None:
    traces = [
        _profile_trace(tmp_path, identifier, latency=10.0)
        for identifier in ("02", "04", "05", "10")
    ]
    expected = _profile(traces)
    allowed_arrays = {
        "timestamps_s",
        "detector_latency_ms",
        "det_boxes_xyxy",
        "det_scores",
        "det_class_ids",
        "det_frame_offsets",
    }
    allowed_manifest = {
        "sequence_id",
        "detector",
        "hardware",
        "fps",
        "causal_trace_sha256",
    }
    for trace in traces:
        object.__setattr__(trace, "arrays", _AccessTrap(trace.arrays, allowed_arrays))
        object.__setattr__(
            trace,
            "manifest_fields",
            _AccessTrap(trace.manifest_fields, allowed_manifest),
        )

    actual = profile_training_traces(
        list(reversed(traces)), tracker_factory=_empty_tracker_factory
    )

    assert actual.canonical_json == expected.canonical_json


def test_profile_hash_binds_domain_schema_and_only_nonrecursive_profile_fields(
    tmp_path: Path,
) -> None:
    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    serialized = profile.to_dict()
    cost_profile = deepcopy(serialized["cost_profile"])
    assert isinstance(cost_profile, dict)
    cost_profile.pop("profile_sha256")
    payload = {
        "hash_domain": "squint.reference-profile/v1",
        "schema": {"name": "squint.reference-profile", "version": 1},
        "detector": serialized["detector"],
        "hardware": serialized["hardware"],
        "cost_profile": cost_profile,
        "normalization": serialized["normalization"],
        "training_traces": serialized["training_traces"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert profile.profile_sha256 == sha256(encoded).hexdigest()
    assert "profile_sha256" not in json.dumps(payload["detector"])
    assert "profile_sha256" not in json.dumps(payload["hardware"])

    changed_domain = deepcopy(payload)
    changed_domain["hash_domain"] = "squint.reference-profile/v2"
    changed_schema = deepcopy(payload)
    schema = changed_schema["schema"]
    assert isinstance(schema, dict)
    schema["version"] = 2
    assert sha256(
        json.dumps(changed_domain, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != profile.profile_sha256
    assert sha256(
        json.dumps(changed_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != profile.profile_sha256


def test_raw_trace_deep_freezes_json_manifest_and_rejects_non_json_values(
    tmp_path: Path,
) -> None:
    arrays, _manifest = _valid_raw_trace_inputs(tmp_path)
    manifest = _trace_manifest(arrays)
    trace = RawTrace("02", arrays, manifest)
    manifest["detector"] = {"precision": "float16"}
    detector = trace.manifest_fields["detector"]
    assert isinstance(detector, Mapping)
    assert detector["precision"] == "float32"
    timing = detector["timing"]
    assert isinstance(timing, Mapping)
    assert timing["includes"] == ("model_forward",)
    preprocessor = detector["preprocessor"]
    assert isinstance(preprocessor, Mapping)
    with pytest.raises(TypeError):
        preprocessor["width"] = 320  # type: ignore[index]
    with pytest.raises(ValueError, match="JSON"):
        RawTrace(
            "02",
            arrays,
            {**_trace_manifest(arrays), "detector": {"bad": object()}},
        )
    with pytest.raises(ValueError, match="finite"):
        RawTrace(
            "02",
            arrays,
            {**_trace_manifest(arrays), "detector": {"bad": nan}},
        )


def test_profile_requires_exact_training_ids_and_matching_nested_identities(
    tmp_path: Path,
) -> None:
    factory = _RecordingFactory()
    base = {
        identifier: _profile_trace(tmp_path, identifier)
        for identifier in ("02", "04", "05", "10", "11")
    }
    for traces in (
        [base[identifier] for identifier in ("02", "04", "05")],
        [base[identifier] for identifier in ("02", "04", "05", "05")],
        [base[identifier] for identifier in ("02", "04", "05", "11")],
    ):
        with pytest.raises(ValueError):
            profile_training_traces(traces, tracker_factory=factory)
    mixed_detector_identity = _detector_identity()
    mixed_detector_identity["precision"] = "float16"
    mixed_detector = _profile_trace(
        tmp_path, "04", detector=mixed_detector_identity
    )
    with pytest.raises(ValueError, match="detector"):
        profile_training_traces(
            [_profile_trace(tmp_path, "02"), mixed_detector]
            + [_profile_trace(tmp_path, identifier) for identifier in ("05", "10")],
            tracker_factory=factory,
        )
    mixed_hardware_identity = _hardware_identity()
    platform = mixed_hardware_identity["platform"]
    assert isinstance(platform, dict)
    platform["machine"] = "aarch64"
    mixed_hardware = _profile_trace(
        tmp_path, "05", hardware=mixed_hardware_identity
    )
    with pytest.raises(ValueError, match="hardware"):
        profile_training_traces(
            [_profile_trace(tmp_path, "02"), _profile_trace(tmp_path, "04"), mixed_hardware, _profile_trace(tmp_path, "10")],
            tracker_factory=factory,
        )
    assert factory.trackers == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model_id",), _DELETE),
        (("revision",), _DELETE),
        (("weights",), _DELETE),
        (("preprocessor",), _DELETE),
        (("threshold",), _DELETE),
        (("class_mapping",), _DELETE),
        (("precision",), _DELETE),
        (("timing",), _DELETE),
        (("unexpected",), "value"),
        (("model_id",), "other/model"),
        (("model_id",), True),
        (("revision",), "main"),
        (("revision",), None),
        (("weights",), {}),
        (("weights",), {"alternate.safetensors": "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"}),
        (("weights", "model.safetensors"), "0" * 64),
        (("weights", "model.safetensors"), True),
        (("preprocessor",), {}),
        (("preprocessor", "class"), _DELETE),
        (("preprocessor", "class"), "OtherProcessor"),
        (("preprocessor", "height"), 320),
        (("preprocessor", "height"), True),
        (("preprocessor", "width"), 320),
        (("preprocessor", "width"), 640.0),
        (("preprocessor", "do_pad"), True),
        (("preprocessor", "do_pad"), 0),
        (("preprocessor", "use_fast"), True),
        (("preprocessor", "use_fast"), 0),
        (("preprocessor", "extra"), False),
        (("threshold",), 0.2),
        (("threshold",), True),
        (("threshold",), nan),
        (("threshold",), "0.10"),
        (("class_mapping",), {}),
        (("class_mapping", "source_label"), _DELETE),
        (("class_mapping", "source_label"), "car"),
        (("class_mapping", "source_label_id"), _DELETE),
        (("class_mapping", "source_label_id"), 1),
        (("class_mapping", "source_label_id"), 999),
        (("class_mapping", "source_label_id"), -1),
        (("class_mapping", "source_label_id"), True),
        (("class_mapping", "output_class_id"), 2),
        (("class_mapping", "output_class_id"), True),
        (("class_mapping", "extra"), 1),
        (("precision",), "bfloat16"),
        (("precision",), True),
        (("timing",), {}),
        (("timing", "protocol"), _DELETE),
        (("timing", "protocol"), "wall-clock-v1"),
        (("timing", "unit"), "seconds"),
        (("timing", "includes"), ["preprocess", "model_forward"]),
        (("timing", "includes"), "model_forward"),
        (("timing", "excludes"), ["telemetry", "postprocess", "preprocess"]),
        (("timing", "extra"), []),
    ],
)
def test_profile_rejects_invalid_detector_identity(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    detector = _mutate_identity(_detector_identity(), path, value)
    with pytest.raises(ValueError, match="detector"):
        traces = [
            _profile_trace(tmp_path, identifier, detector=detector)
            for identifier in ("02", "04", "05", "10")
        ]
        _profile(traces)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("platform",), _DELETE),
        (("runtime",), _DELETE),
        (("device",), _DELETE),
        (("unexpected",), {}),
        (("platform",), {}),
        (("platform", "system"), _DELETE),
        (("platform", "machine"), _DELETE),
        (("platform", "python"), _DELETE),
        (("platform", "system"), ""),
        (("platform", "machine"), True),
        (("platform", "python"), None),
        (("platform", "extra"), "value"),
        (("runtime",), {}),
        (("runtime", "torch"), _DELETE),
        (("runtime", "transformers"), _DELETE),
        (("runtime", "cuda"), _DELETE),
        (("runtime", "driver"), _DELETE),
        (("runtime", "torch"), ""),
        (("runtime", "transformers"), True),
        (("runtime", "cuda"), ""),
        (("runtime", "driver"), False),
        (("runtime", "extra"), None),
        (("device",), {}),
        (("device", "type"), _DELETE),
        (("device", "name"), _DELETE),
        (("device", "uuid"), _DELETE),
        (("device", "pci_bus_id"), _DELETE),
        (("device", "type"), ""),
        (("device", "type"), "tpu"),
        (("device", "name"), True),
        (("device", "uuid"), ""),
        (("device", "pci_bus_id"), False),
        (("device", "extra"), None),
        (("runtime", "cuda"), "12.4"),
        (("runtime", "driver"), "550.54"),
        (("device", "uuid"), "CPU-fixture"),
        (("device", "pci_bus_id"), "0000:00:00.0"),
    ],
)
def test_profile_rejects_invalid_hardware_identity(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    hardware = _hardware_identity(
        device_type="cuda" if path == ("device", "type") and value == "tpu" else "cpu"
    )
    hardware = _mutate_identity(hardware, path, value)
    with pytest.raises(ValueError, match="hardware"):
        traces = [
            _profile_trace(tmp_path, identifier, hardware=hardware)
            for identifier in ("02", "04", "05", "10")
        ]
        _profile(traces)


@pytest.mark.parametrize(
    "path",
    [
        ("runtime", "cuda"),
        ("runtime", "driver"),
        ("device", "uuid"),
        ("device", "pci_bus_id"),
    ],
)
def test_accelerated_hardware_requires_complete_runtime_and_device_identity(
    tmp_path: Path, path: tuple[str, ...]
) -> None:
    hardware = _mutate_identity(_hardware_identity(device_type="cuda"), path, None)
    traces = [
        _profile_trace(tmp_path, identifier, hardware=hardware)
        for identifier in ("02", "04", "05", "10")
    ]
    with pytest.raises(ValueError, match="hardware"):
        _profile(traces)


def test_complete_accelerated_hardware_identity_is_supported(tmp_path: Path) -> None:
    hardware = _hardware_identity(device_type="cuda")
    profile = _profile(
        [
            _profile_trace(tmp_path, identifier, hardware=hardware)
            for identifier in ("02", "04", "05", "10")
        ],
    )
    assert profile.to_dict()["hardware"] == hardware


def test_cuda_detector_allows_float16_with_pinned_person_label_id(
    tmp_path: Path,
) -> None:
    detector = _detector_identity()
    detector["precision"] = "float16"
    hardware = _hardware_identity(device_type="cuda")
    profile = _profile(
        [
            _profile_trace(
                tmp_path, identifier, detector=detector, hardware=hardware
            )
            for identifier in ("02", "04", "05", "10")
        ],
    )
    assert profile.to_dict()["detector"] == detector


@pytest.mark.parametrize("entry_point", ["training", "direct", "load"])
def test_cpu_profile_rejects_float16_through_every_entry_point(
    tmp_path: Path, entry_point: str
) -> None:
    detector = _detector_identity()
    detector["precision"] = "float16"
    if entry_point == "training":
        with pytest.raises(ValueError, match=r"CPU.*float32|float32.*CPU"):
            _profile(
                [
                    _profile_trace(tmp_path, identifier, detector=detector)
                    for identifier in ("02", "04", "05", "10")
                ],
            )
        return

    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    if entry_point == "direct":
        with pytest.raises(ValueError, match=r"CPU.*float32|float32.*CPU"):
            ReferenceProfile(
                detector,
                profile.hardware,
                profile.cost_profile,
                profile.normalization,
                profile.training_traces,
            )
        return

    payload = profile.to_dict()
    payload["detector"] = detector
    path = tmp_path / "cpu-float16.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"CPU.*float32|float32.*CPU"):
        ReferenceProfile.load(path)


def test_profile_hash_excludes_gt_source_and_telemetry_but_includes_causal_trace(
    tmp_path: Path,
) -> None:
    traces = [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    original = _profile(traces)
    arrays = dict(traces[0].arrays)
    arrays["gt_boxes_xyxy"] = arrays["gt_boxes_xyxy"].copy()
    arrays["gt_boxes_xyxy"][0, 0] += 1.0
    changed_manifest = dict(traces[0].manifest_fields)
    changed_manifest["source_sha256"] = "c" * 64
    changed_manifest["telemetry"] = {"gpu_utilization": 99.0}
    changed = RawTrace("02", arrays, changed_manifest)
    unchanged = _profile([changed, *traces[1:]])
    assert unchanged.canonical_json == original.canonical_json
    causal_arrays = dict(traces[0].arrays)
    causal_arrays["scene_change"] = causal_arrays["scene_change"].copy()
    causal_arrays["scene_change"][0, 0, 0] = 0.5
    causal_manifest = dict(traces[0].manifest_fields)
    causal_manifest["causal_trace_sha256"] = causal_trace_sha256(causal_arrays)
    causal_changed = RawTrace("02", causal_arrays, causal_manifest)
    changed_profile = _profile([causal_changed, *traces[1:]])
    assert changed_profile.canonical_json != original.canonical_json
    assert changed_profile.normalization == original.normalization
    assert {
        key: value
        for key, value in changed_profile.cost_profile.items()
        if key != "profile_sha256"
    } == {
        key: value
        for key, value in original.cost_profile.items()
        if key != "profile_sha256"
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": {"name": "wrong", "version": 1}},
        {"schema": {"name": "squint.reference-profile", "version": 2}},
        {"schema": {"name": "squint.reference-profile", "version": True}},
        {"cost_profile": {"reserve_ms": True}},
        {"cost_profile": {"profile_sha256": "A" * 64}},
        {"cost_profile": {"profile_sha256": "0" * 63}},
        {"normalization": {"age_s": 0.0}},
    ],
)
def test_profile_load_rejects_schema_numeric_and_hash_mutations(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    payload = profile.to_dict()
    for key, value in mutation.items():
        if isinstance(value, dict):
            base = payload[key]
            assert isinstance(base, dict)
            payload[key] = {**base, **value}
        else:
            payload[key] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError):
        ReferenceProfile.load(path)


def test_profile_load_rejects_bool_schema_version_contextually(tmp_path: Path) -> None:
    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    payload = profile.to_dict()
    payload["schema"] = {"name": "squint.reference-profile", "version": True}
    path = tmp_path / "bool-version.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version"):
        ReferenceProfile.load(path)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("schema", []),
        ("detector", []),
        ("hardware", False),
        ("cost_profile", "bad"),
        ("normalization", 1),
        ("training_traces", None),
        ("training_traces", True),
        ("training_traces", 1),
        ("training_traces", "02"),
        ("training_traces", {}),
        ("training_traces", ["02"]),
    ],
)
def test_profile_load_wraps_every_malformed_top_level_type_as_value_error(
    tmp_path: Path, field: str, malformed: object
) -> None:
    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    payload = profile.to_dict()
    payload[field] = malformed
    path = tmp_path / f"malformed-{field}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        ReferenceProfile.load(path)


def test_profile_write_requires_new_destination_and_load_requires_canonical_json(
    tmp_path: Path,
) -> None:
    profile = _profile(
        [_profile_trace(tmp_path, identifier) for identifier in ("02", "04", "05", "10")]
    )
    destination = tmp_path / "profile.json"
    profile.write(destination)
    with pytest.raises(FileExistsError):
        profile.write(destination)
    destination.write_text(" \n" + profile.to_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        ReferenceProfile.load(destination)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "array set"),
        ("extra", "array set"),
        ("dtype", "det_scores"),
        ("shape", "scene_change"),
        ("offset-start", "det_frame_offsets"),
        ("offset-order", "det_frame_offsets"),
        ("offset-final", "det_frame_offsets"),
        ("cardinality", "gt_track_ids"),
        ("timestamp-order", "timestamps_s"),
        ("det-score-range", "det_scores"),
        ("gt-visibility-range", "gt_visibility"),
    ],
)
def test_raw_trace_direct_construction_rejects_malformed_arrays_contextually(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    arrays, manifest = _valid_raw_trace_inputs(tmp_path)
    if mutation == "missing":
        arrays.pop("gt_ignore")
    elif mutation == "extra":
        arrays["surprise"] = np.empty(0, np.float32)
    elif mutation == "dtype":
        arrays["det_scores"] = arrays["det_scores"].astype(np.float64)
    elif mutation == "shape":
        arrays["scene_change"] = arrays["scene_change"][:, :, :2]
    elif mutation == "offset-start":
        arrays["det_frame_offsets"] = np.array([1, 1, 2, 3], np.int64)
    elif mutation == "offset-order":
        arrays["det_frame_offsets"] = np.array([0, 2, 1, 3], np.int64)
    elif mutation == "offset-final":
        arrays["det_frame_offsets"] = np.array([0, 1, 2, 2], np.int64)
    elif mutation == "cardinality":
        arrays["gt_track_ids"] = arrays["gt_track_ids"][:-1]
    elif mutation == "timestamp-order":
        arrays["timestamps_s"] = np.array([0.0, 0.04, 0.04], np.float64)
    elif mutation == "det-score-range":
        arrays["det_scores"][0] = 1.01
    else:
        arrays["gt_visibility"][0] = -0.01
    with pytest.raises(ValueError, match=match):
        RawTrace("02", arrays, manifest)


def test_raw_trace_direct_construction_requires_coherent_causal_hash(tmp_path: Path) -> None:
    arrays, manifest = _valid_raw_trace_inputs(tmp_path)
    manifest["causal_trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="causal_trace_sha256"):
        RawTrace("02", arrays, manifest)
    manifest = _trace_manifest(arrays)
    manifest.pop("causal_trace_sha256")
    with pytest.raises(ValueError, match="causal_trace_sha256"):
        RawTrace("02", arrays, manifest)


def test_source_hash_streams_sorted_files_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = _sequence(tmp_path)
    expected = canonical_source_sha256(sequence)
    reversed_sequence = replace(sequence, image_paths=tuple(reversed(sequence.image_paths)))
    assert canonical_source_sha256(reversed_sequence) == expected

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("canonical source hashing must stream files")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    assert canonical_source_sha256(sequence) == expected


def test_source_hash_is_path_sensitive_and_wraps_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = _sequence(tmp_path)
    original = canonical_source_sha256(sequence)
    renamed = sequence.image_paths[0].with_name("renamed.png")
    sequence.image_paths[0].rename(renamed)
    renamed_sequence = replace(sequence, image_paths=(renamed, *sequence.image_paths[1:]))
    assert canonical_source_sha256(renamed_sequence) != original

    real_open = Path.open

    def fail_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == renamed:
            raise OSError("unreadable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(ValueError, match=r"renamed\.png.*read|read.*renamed\.png"):
        canonical_source_sha256(renamed_sequence)


def test_import_does_not_load_heavy_or_hardware_modules() -> None:
    root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    script = (
        "import sys; import squint_rl.reference.build_mot17; "
        "forbidden={'torch','transformers','pynvml'}; "
        "loaded={name.split('.')[0] for name in sys.modules}; "
        "assert forbidden.isdisjoint(loaded), forbidden & loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
