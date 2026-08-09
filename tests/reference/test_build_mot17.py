from __future__ import annotations

import inspect
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

from squint_rl.artifacts import AtomicRun
from squint_rl.episode import Episode
from squint_rl.episode import seal_episode as real_seal_episode
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
from squint_rl.reference.mot17 import Mot17FormatError, Mot17Sequence
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


def _telemetry(
    *, available: bool = False, sample_count: int = 0
) -> dict[str, object]:
    populated = sample_count > 0
    return {
        "nvml": {"available": available, "error": None},
        "sample_count": sample_count,
        "gpu_utilization_percent": {
            "mean": 10.0 if populated else None,
            "p95": 12.0 if populated else None,
            "max": 15.0 if populated else None,
        },
        "used_vram_bytes": {
            "mean": 100.0 if populated else None,
            "p95": 120.0 if populated else None,
            "max": 150.0 if populated else None,
        },
    }


def _sealable_trace(
    tmp_path: Path,
    identifier: str,
    *,
    detector: dict[str, object] | None = None,
    hardware: dict[str, object] | None = None,
    telemetry: dict[str, object] | None = None,
) -> RawTrace:
    trace = _profile_trace(
        tmp_path,
        identifier,
        detector=detector,
        hardware=hardware,
    )
    manifest = dict(trace.manifest_fields)
    manifest.update(
        {
            "width": 4,
            "height": 4,
            "telemetry": _telemetry() if telemetry is None else telemetry,
        }
    )
    return RawTrace(identifier, trace.arrays, manifest)


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


def test_seal_trace_train_round_trips_exact_manifest_and_arrays(tmp_path: Path) -> None:
    traces = [
        _sealable_trace(tmp_path, "02"),
        *[_profile_trace(tmp_path, identifier) for identifier in ("04", "05", "10")],
    ]
    trace = traces[0]
    profile = _profile(traces)
    profile_fields = profile.episode_manifest_fields()
    expected = {
        "schema": {"name": "squint.replay", "version": 1},
        "episode": {"id": "MOT17-02-FRCNN"},
        "source": {
            "id": "02",
            "sha256": "a" * 64,
            "width": 4,
            "height": 4,
            "frame_count": 3,
            "fps": 25.0,
            "duration_s": 0.08,
            "dataset": "MOT17",
            "split": "train",
            "class_mapping": {"1": "pedestrian"},
            "ignore_region_rules": (
                "valid iff mark=1 and class_id=1; ignored iff mark=1 and class_id "
                "in {2,7,8,12}; otherwise neither valid nor ignored"
            ),
        },
        "detector": _detector_identity(),
        "hardware": _hardware_identity(),
        "cost_profile": profile_fields["cost_profile"],
        "scene_feature": {
            "name": "mean_absolute_grayscale_change",
            "shape": [3, 3],
        },
        "normalization": profile_fields["normalization"],
        "telemetry": _telemetry(),
        "artifacts": {},
    }

    manifest = build_mot17_module._episode_manifest(
        trace, profile, partition="train"
    )
    assert _plain_json(manifest) == expected
    destination = tmp_path / "sealed-train"
    assert build_mot17_module._seal_trace(
        destination,
        trace=trace,
        profile=profile,
        partition="train",
    ) == destination.resolve()

    episode = Episode.open(destination)
    sealed_manifest = _plain_json(episode.manifest)
    assert isinstance(sealed_manifest, dict)
    artifacts = sealed_manifest.pop("artifacts")
    assert sealed_manifest == {key: value for key, value in expected.items() if key != "artifacts"}
    assert isinstance(artifacts, dict)
    assert set(artifacts) == {"arrays.npz_sha256", "content_sha256"}
    assert episode.content_sha256 == artifacts["content_sha256"]
    assert episode.manifest["cost_profile"]["profile_sha256"] == profile.profile_sha256  # type: ignore[index]
    assert set(destination.iterdir()) == {
        destination / "manifest.json",
        destination / "arrays.npz",
    }
    for name, expected_array in trace.arrays.items():
        np.testing.assert_array_equal(episode.arrays[name], expected_array)


@pytest.mark.parametrize(
    ("identifier", "partition"),
    [("09", "validation"), ("11", "test"), ("13", "test")],
)
def test_synthetic_heldout_trace_seals_without_recomputing_training_profile(
    tmp_path: Path, identifier: str, partition: str
) -> None:
    training = [
        _sealable_trace(tmp_path / "training", "02"),
        *[
            _profile_trace(tmp_path / "training", name)
            for name in ("04", "05", "10")
        ],
    ]
    profile = _profile(training)
    profile_bytes = profile.canonical_json.encode()
    heldout = _sealable_trace(tmp_path / "synthetic-heldout", identifier)
    assert all(
        item["causal_trace_sha256"]
        != heldout.manifest_fields["causal_trace_sha256"]
        or item["sequence_id"] != identifier
        for item in profile.training_traces
    )

    episode = Episode.open(
        build_mot17_module._seal_trace(
            tmp_path / f"heldout-{identifier}",
            trace=heldout,
            profile=profile,
            partition=partition,
        )
    )

    assert profile.canonical_json.encode() == profile_bytes
    assert episode.manifest["episode"] == {"id": f"MOT17-{identifier}-FRCNN"}
    assert episode.manifest["source"]["split"] == partition  # type: ignore[index]


@pytest.mark.parametrize(
    ("identifier", "partition"),
    [
        ("09", "train"),
        ("11", "validation"),
        ("09", "test"),
        ("03", "validation"),
        ("09", "development"),
    ],
)
def test_partition_and_sequence_membership_fail_before_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identifier: str,
    partition: str,
) -> None:
    training = [
        _sealable_trace(tmp_path / "training", "02"),
        *[
            _profile_trace(tmp_path / "training", name)
            for name in ("04", "05", "10")
        ],
    ]
    profile = _profile(training)
    trace = _sealable_trace(tmp_path / "candidate", identifier)
    destination = tmp_path / "must-not-exist"
    seal_calls = 0

    def forbidden_seal(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal seal_calls
        seal_calls += 1
        raise AssertionError("partition validation must precede seal_episode")

    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_seal)
    with pytest.raises(ValueError, match="partition|sequence"):
        build_mot17_module._seal_trace(
            destination,
            trace=trace,
            profile=profile,
            partition=partition,
        )

    assert seal_calls == 0
    assert not destination.exists()


@pytest.mark.parametrize("location", ["trace", "manifest"])
def test_non_builtin_sequence_ids_fail_before_membership_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    class EqualitySpoof:
        def __init__(self) -> None:
            self.comparisons = 0

        def __eq__(self, other: object) -> bool:
            del other
            self.comparisons += 1
            return True

        def __str__(self) -> str:
            return "02"

    class SequenceIdSubclass(str):
        pass

    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    spoof = EqualitySpoof()
    if location == "trace":
        object.__setattr__(trace, "sequence_id", spoof)
    else:
        manifest = dict(trace.manifest_fields)
        manifest["sequence_id"] = SequenceIdSubclass("02")
        trace = RawTrace("02", trace.arrays, manifest)
    destination = tmp_path / "published"
    before = {
        path.relative_to(tmp_path): None if path.is_dir() else path.read_bytes()
        for path in tmp_path.rglob("*")
    }
    seal_calls = 0

    def record_seal(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal seal_calls
        seal_calls += 1
        return destination

    monkeypatch.setattr(build_mot17_module, "seal_episode", record_seal)
    with pytest.raises(ValueError, match="sequence_id"):
        build_mot17_module._seal_trace(
            destination,
            trace=trace,
            profile=profile,
            partition="train",
        )

    after = {
        path.relative_to(tmp_path): None if path.is_dir() else path.read_bytes()
        for path in tmp_path.rglob("*")
    }
    assert spoof.comparisons == 0
    assert seal_calls == 0
    assert before == after
    assert not destination.exists()
    assert not tuple(destination.parent.glob(f".{destination.name}.*.incomplete"))


@pytest.mark.parametrize("mutation", ["detector", "hardware"])
def test_identity_mismatch_fails_before_manifest_or_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    hardware = _hardware_identity(device_type="cuda")
    trace = _sealable_trace(tmp_path, "02", hardware=hardware)
    traces = [
        trace,
        *[
            _profile_trace(tmp_path, identifier, hardware=hardware)
            for identifier in ("04", "05", "10")
        ],
    ]
    profile = _profile(traces)
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    if mutation == "detector":
        detector = manifest["detector"]
        assert isinstance(detector, dict)
        detector["precision"] = "float16"
    else:
        candidate_hardware = manifest["hardware"]
        assert isinstance(candidate_hardware, dict)
        platform = candidate_hardware["platform"]
        assert isinstance(platform, dict)
        platform["machine"] = "aarch64"
    candidate = RawTrace("02", trace.arrays, manifest)
    manifest_calls = 0
    seal_calls = 0

    def forbidden_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal manifest_calls
        manifest_calls += 1
        raise AssertionError("identity validation must precede manifest construction")

    def forbidden_seal(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal seal_calls
        seal_calls += 1
        raise AssertionError("identity validation must precede seal_episode")

    monkeypatch.setattr(build_mot17_module, "_episode_manifest", forbidden_manifest)
    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_seal)
    with pytest.raises(ValueError, match=mutation):
        build_mot17_module._seal_trace(
            tmp_path / "must-not-exist",
            trace=candidate,
            profile=profile,
            partition="train",
        )

    assert manifest_calls == seal_calls == 0
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("mutation", ["detection", "latency", "scene"])
def test_coherent_causal_mutation_requires_a_different_training_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    traces = [
        trace,
        *[_profile_trace(tmp_path, identifier) for identifier in ("04", "05", "10")],
    ]
    profile = _profile(traces)
    arrays = {name: np.array(value, copy=True) for name, value in trace.arrays.items()}
    if mutation == "detection":
        arrays["det_scores"][0] = 0.5
    elif mutation == "latency":
        arrays["detector_latency_ms"][0] += 1.0
    else:
        arrays["scene_change"][0, 0, 0] = 0.5
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    manifest["causal_trace_sha256"] = causal_trace_sha256(arrays)
    candidate = RawTrace("02", arrays, manifest)
    assert (
        candidate.manifest_fields["causal_trace_sha256"]
        != trace.manifest_fields["causal_trace_sha256"]
    )
    changed_profile = _profile([candidate, *traces[1:]])
    assert changed_profile.profile_sha256 != profile.profile_sha256
    publication_calls = 0

    def forbidden_publication(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal publication_calls
        publication_calls += 1
        raise AssertionError("training hash validation must precede publication")

    monkeypatch.setattr(build_mot17_module, "_episode_manifest", forbidden_publication)
    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_publication)
    with pytest.raises(ValueError, match="causal|training"):
        build_mot17_module._seal_trace(
            tmp_path / "must-not-exist",
            trace=candidate,
            profile=profile,
            partition="train",
        )

    assert publication_calls == 0
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", _DELETE),
        ("width", True),
        ("width", 0),
        ("height", _DELETE),
        ("height", 1.0),
        ("source_sha256", _DELETE),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("detector", _DELETE),
        ("hardware", _DELETE),
        ("telemetry", _DELETE),
        ("fps", np.float64(25.0)),
    ],
)
def test_missing_or_ill_typed_provenance_fails_before_manifest_or_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    if value is _DELETE:
        manifest.pop(field)
    else:
        manifest[field] = value
    candidate = RawTrace("02", trace.arrays, manifest)
    publication_calls = 0

    def forbidden_publication(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal publication_calls
        publication_calls += 1
        raise AssertionError("provenance validation must precede publication")

    monkeypatch.setattr(build_mot17_module, "_episode_manifest", forbidden_publication)
    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_publication)
    with pytest.raises(ValueError, match=field):
        build_mot17_module._seal_trace(
            tmp_path / "must-not-exist",
            trace=candidate,
            profile=profile,
            partition="train",
        )

    assert publication_calls == 0
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "root-type",
        "root-extra",
        "nvml-extra",
        "available-type",
        "error-code",
        "count-bool",
        "count-negative",
        "count-over-frames",
        "empty-values",
        "nonempty-null",
        "nonempty-integer",
        "gpu-range",
        "vram-negative",
        "mean-over-max",
        "p95-over-max",
        "unavailable-samples",
        "unavailable-error",
        "statistics-extra",
    ],
)
def test_malformed_aggregate_telemetry_fails_before_manifest_or_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    telemetry: object = deepcopy(_telemetry())
    if mutation == "root-type":
        telemetry = []
    else:
        assert isinstance(telemetry, dict)
        if mutation == "root-extra":
            telemetry["extra"] = None
        elif mutation == "nvml-extra":
            nvml = telemetry["nvml"]
            assert isinstance(nvml, dict)
            nvml["extra"] = None
        elif mutation == "available-type":
            telemetry["nvml"] = {"available": 1, "error": None}
        elif mutation == "error-code":
            telemetry["nvml"] = {"available": True, "error": "unknown"}
        elif mutation == "count-bool":
            telemetry["sample_count"] = True
        elif mutation == "count-negative":
            telemetry["sample_count"] = -1
        elif mutation == "count-over-frames":
            telemetry["sample_count"] = 4
        elif mutation == "empty-values":
            telemetry["gpu_utilization_percent"] = {
                "mean": 0.0,
                "p95": None,
                "max": None,
            }
        else:
            telemetry = deepcopy(_telemetry(available=True, sample_count=1))
            assert isinstance(telemetry, dict)
            gpu = telemetry["gpu_utilization_percent"]
            vram = telemetry["used_vram_bytes"]
            assert isinstance(gpu, dict)
            assert isinstance(vram, dict)
            if mutation == "nonempty-null":
                gpu["mean"] = None
            elif mutation == "nonempty-integer":
                gpu["mean"] = 10
            elif mutation == "gpu-range":
                gpu["max"] = 101.0
            elif mutation == "vram-negative":
                vram["mean"] = -1.0
            elif mutation == "mean-over-max":
                gpu["mean"] = 16.0
            elif mutation == "p95-over-max":
                gpu["p95"] = 16.0
            elif mutation == "unavailable-samples":
                telemetry["nvml"] = {"available": False, "error": None}
            elif mutation == "unavailable-error":
                telemetry["sample_count"] = 0
                telemetry["nvml"] = {
                    "available": False,
                    "error": "sample_failed",
                }
                telemetry["gpu_utilization_percent"] = {
                    "mean": None,
                    "p95": None,
                    "max": None,
                }
                telemetry["used_vram_bytes"] = {
                    "mean": None,
                    "p95": None,
                    "max": None,
                }
            else:
                gpu["extra"] = 10.0
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    manifest["telemetry"] = telemetry
    candidate = RawTrace("02", trace.arrays, manifest)
    publication_calls = 0

    def forbidden_publication(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal publication_calls
        publication_calls += 1
        raise AssertionError("telemetry validation must precede publication")

    monkeypatch.setattr(build_mot17_module, "_episode_manifest", forbidden_publication)
    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_publication)
    with pytest.raises(ValueError, match="telemetry|nvml|sample_count|utilization|vram"):
        build_mot17_module._seal_trace(
            tmp_path / "must-not-exist",
            trace=candidate,
            profile=profile,
            partition="train",
        )

    assert publication_calls == 0
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("error_code", ["sample_failed", "invalid_sample"])
def test_telemetry_error_str_subclass_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    class ErrorCodeSubclass(str):
        pass

    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    telemetry = _telemetry()
    telemetry["nvml"] = {
        "available": True,
        "error": ErrorCodeSubclass(error_code),
    }
    manifest["telemetry"] = telemetry
    candidate = RawTrace("02", trace.arrays, manifest)
    destination = tmp_path / "published"
    before = {
        path.relative_to(tmp_path): None if path.is_dir() else path.read_bytes()
        for path in tmp_path.rglob("*")
    }
    seal_calls = 0

    def record_seal(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal seal_calls
        seal_calls += 1
        return destination

    monkeypatch.setattr(build_mot17_module, "seal_episode", record_seal)
    with pytest.raises(ValueError, match=r"nvml\.error"):
        build_mot17_module._seal_trace(
            destination,
            trace=candidate,
            profile=profile,
            partition="train",
        )

    after = {
        path.relative_to(tmp_path): None if path.is_dir() else path.read_bytes()
        for path in tmp_path.rglob("*")
    }
    assert seal_calls == 0
    assert before == after
    assert not destination.exists()
    assert not tuple(destination.parent.glob(f".{destination.name}.*.incomplete"))


def test_seal_trace_calls_compatibility_manifest_then_seal_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    destination = tmp_path / "destination"
    returned = tmp_path / "returned"
    manifest = {"sentinel": "manifest"}
    events: list[str] = []

    def require(
        actual_trace: RawTrace,
        actual_profile: ReferenceProfile,
        *,
        partition: str,
    ) -> None:
        assert (actual_trace, actual_profile, partition) == (trace, profile, "train")
        events.append("require")

    def episode_manifest(
        actual_trace: RawTrace,
        actual_profile: ReferenceProfile,
        *,
        partition: str,
    ) -> dict[str, object]:
        assert (actual_trace, actual_profile, partition) == (trace, profile, "train")
        events.append("manifest")
        return manifest

    def seal(
        actual_destination: str | Path,
        *,
        manifest: Mapping[str, object],
        arrays: Mapping[str, np.ndarray[Any, Any]],
    ) -> Path:
        assert Path(actual_destination) == destination
        assert manifest is globals_manifest
        assert arrays is trace.arrays
        events.append("seal")
        return returned

    globals_manifest = manifest
    monkeypatch.setattr(
        build_mot17_module, "_require_profile_compatible", require, raising=False
    )
    monkeypatch.setattr(
        build_mot17_module, "_episode_manifest", episode_manifest, raising=False
    )
    monkeypatch.setattr(build_mot17_module, "seal_episode", seal, raising=False)

    assert build_mot17_module._seal_trace(
        destination,
        trace=trace,
        profile=profile,
        partition="train",
    ) == returned
    assert events == ["require", "manifest", "seal"]


def test_existing_destination_rejection_is_owned_by_one_real_seal_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    calls: list[Path] = []

    def counted_real_seal(
        actual_destination: str | Path,
        *,
        manifest: Mapping[str, object],
        arrays: Mapping[str, np.ndarray[Any, Any]],
    ) -> Path:
        calls.append(Path(actual_destination))
        return real_seal_episode(
            actual_destination,
            manifest=manifest,
            arrays=arrays,
        )

    monkeypatch.setattr(
        build_mot17_module, "seal_episode", counted_real_seal, raising=False
    )
    with pytest.raises(FileExistsError):
        build_mot17_module._seal_trace(
            destination,
            trace=trace,
            profile=profile,
            partition="train",
        )

    assert calls == [destination]
    assert set(destination.iterdir()) == {sentinel}
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("mutation", ["ground-truth", "source", "telemetry"])
def test_noncausal_mutations_preserve_profile_bytes_but_change_episode_identity(
    tmp_path: Path, mutation: str
) -> None:
    hardware = _hardware_identity(device_type="cuda")
    trace = _sealable_trace(
        tmp_path,
        "02",
        hardware=hardware,
        telemetry=_telemetry(available=True, sample_count=1),
    )
    traces = [
        trace,
        *[
            _profile_trace(tmp_path, identifier, hardware=hardware)
            for identifier in ("04", "05", "10")
        ],
    ]
    profile = _profile(traces)
    arrays = {name: np.array(value, copy=True) for name, value in trace.arrays.items()}
    manifest = _plain_json(trace.manifest_fields)
    assert isinstance(manifest, dict)
    if mutation == "ground-truth":
        arrays["gt_boxes_xyxy"][0, 0] += 0.25
    elif mutation == "source":
        manifest["source_sha256"] = "b" * 64
    else:
        manifest["telemetry"] = {
            "nvml": {"available": True, "error": None},
            "sample_count": 1,
            "gpu_utilization_percent": {"mean": 11.0, "p95": 13.0, "max": 16.0},
            "used_vram_bytes": {"mean": 100.0, "p95": 120.0, "max": 150.0},
        }
    manifest["causal_trace_sha256"] = causal_trace_sha256(arrays)
    changed = RawTrace("02", arrays, manifest)

    changed_profile = _profile([changed, *traces[1:]])
    assert changed_profile.canonical_json.encode() == profile.canonical_json.encode()
    original_episode = Episode.open(
        build_mot17_module._seal_trace(
            tmp_path / f"{mutation}-original",
            trace=trace,
            profile=profile,
            partition="train",
        )
    )
    changed_episode = Episode.open(
        build_mot17_module._seal_trace(
            tmp_path / f"{mutation}-changed",
            trace=changed,
            profile=profile,
            partition="train",
        )
    )
    assert changed_episode.content_sha256 != original_episode.content_sha256


def test_resealing_is_deterministic_and_profile_has_no_content_circularity(
    tmp_path: Path,
) -> None:
    trace = _sealable_trace(tmp_path, "02")
    profile = _profile(
        [trace, *[_profile_trace(tmp_path, name) for name in ("04", "05", "10")]]
    )
    first_path = build_mot17_module._seal_trace(
        tmp_path / "first",
        trace=trace,
        profile=profile,
        partition="train",
    )
    second_path = build_mot17_module._seal_trace(
        tmp_path / "second",
        trace=trace,
        profile=profile,
        partition="train",
    )
    first = Episode.open(first_path)
    second = Episode.open(second_path)

    assert first.content_sha256 == second.content_sha256
    assert (first_path / "manifest.json").read_bytes() == (
        second_path / "manifest.json"
    ).read_bytes()
    assert (first_path / "arrays.npz").read_bytes() == (
        second_path / "arrays.npz"
    ).read_bytes()
    assert "causal_trace_sha256" in profile.canonical_json
    assert all(
        f'"{field}":' not in profile.canonical_json
        for field in ("content_sha256", "source_sha256", "telemetry")
    )
    assert first.content_sha256 not in profile.canonical_json


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


def _partition_profile(
    tmp_path: Path,
    *,
    precision: str = "float32",
    device_type: str = "cpu",
) -> ReferenceProfile:
    detector = _detector_identity()
    detector["precision"] = precision
    hardware = _hardware_identity(device_type=device_type)
    return _profile(
        [
            _profile_trace(
                tmp_path,
                identifier,
                detector=detector,
                hardware=hardware,
                frame_count=2,
            )
            for identifier in ("02", "04", "05", "10")
        ]
    )


def _install_real_partition_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hardware: dict[str, object] | None = None,
    events: list[str] | None = None,
) -> list[Any]:
    identity = _hardware_identity() if hardware is None else hardware
    sessions: list[Any] = []

    def create_session(_cls: type[object], device: str) -> Any:
        if events is not None:
            events.append(f"session:{device}")
        session = build_mot17_module._HardwareSession(identity)
        sessions.append(session)
        return session

    def load_source(root: str | Path, identifier: str) -> Mot17Sequence:
        if events is not None:
            events.append(f"source:{identifier}")
        return _sequence(tmp_path / "sources", frame_count=2, identifier=identifier)

    def load_detector(
        _cls: type[object],
        *,
        device: str,
        precision: str,
    ) -> _ConstantDetector:
        if events is not None:
            events.append(f"detector:{device}:{precision}")
        return _ConstantDetector()

    monkeypatch.setattr(
        build_mot17_module._HardwareSession,
        "create",
        classmethod(create_session),
    )
    monkeypatch.setattr(build_mot17_module, "load_sequence", load_source)
    monkeypatch.setattr(
        build_mot17_module.DFineDetector,
        "load",
        classmethod(load_detector),
    )
    return sessions


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize(
    ("name", "parameters", "defaults"),
    [
        (
            "build_partition",
            [
                "mot17_root",
                "destination",
                "partition",
                "device",
                "precision",
                "profile",
                "warmup_frames",
            ],
            {"device": "cuda", "precision": "float32", "profile": None, "warmup_frames": 10},
        ),
        ("main", ["argv"], {"argv": None}),
    ],
)
def test_partition_builder_exposes_only_the_binding_callable_contract(
    name: str,
    parameters: list[str],
    defaults: dict[str, object],
) -> None:
    function = getattr(build_mot17_module, name)
    signature = inspect.signature(function)
    assert list(signature.parameters) == parameters
    assert {
        parameter: signature.parameters[parameter].default
        for parameter in defaults
    } == defaults
    if name == "build_partition":
        assert all(
            signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters[2:]
        )


def test_train_partition_real_transaction_publishes_exact_replay_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _install_real_partition_boundaries(tmp_path, monkeypatch)
    destination = tmp_path / "published" / "train"

    returned = build_mot17_module.build_partition(
        tmp_path / "MOT17",
        destination,
        partition="train",
        device="cpu",
        warmup_frames=0,
    )

    assert returned == destination.absolute()
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert {path.name for path in destination.iterdir()} == {
        "training-profile.json",
        "MOT17-02",
        "MOT17-04",
        "MOT17-05",
        "MOT17-10",
    }
    profile_path = destination / "training-profile.json"
    profile = ReferenceProfile.load(profile_path)
    assert profile_path.read_bytes() == profile.canonical_json.encode("utf-8")
    for identifier in ("02", "04", "05", "10"):
        episode_path = destination / f"MOT17-{identifier}"
        assert {path.name for path in episode_path.iterdir()} == {
            "manifest.json",
            "arrays.npz",
        }
        episode = Episode.open(episode_path)
        assert episode.manifest["episode"] == {
            "id": f"MOT17-{identifier}-FRCNN"
        }
        assert episode.manifest["source"]["split"] == "train"  # type: ignore[index]
        assert episode.manifest["cost_profile"]["profile_sha256"] == (  # type: ignore[index]
            profile.profile_sha256
        )
        np.testing.assert_array_equal(episode.arrays["timestamps_s"], [0.0, 0.04])
        np.testing.assert_array_equal(
            episode.arrays["detector_latency_ms"], [5.0, 5.0]
        )
        np.testing.assert_array_equal(
            episode.arrays["det_frame_offsets"], [0, 1, 2]
        )
    assert not tuple(destination.parent.glob(f".{destination.name}.*.incomplete"))
    assert not (destination.parent / f".{destination.name}.publish.lock").exists()


def test_partition_order_cardinality_and_shared_objects_are_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    destination = tmp_path / "partition"
    real_atomic_run = AtomicRun

    class RecordingAtomicRun:
        def __init__(self, target: str | Path) -> None:
            self._run = real_atomic_run(target)
            self.destination = self._run.destination

        def __enter__(self) -> Path:
            events.append("atomic-enter")
            return self._run.__enter__()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> bool:
            events.append("atomic-exit")
            return self._run.__exit__(exc_type, exc_value, traceback)  # type: ignore[arg-type]

    hardware = _hardware_identity()

    class Session:
        def __init__(self) -> None:
            self.hardware_identity = hardware
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            events.append("close")

    session = Session()
    detector = object()
    traces = [
        _sealable_trace(tmp_path, identifier)
        for identifier in ("02", "04", "05", "10")
    ]
    selected_profile = _profile(traces)
    source_objects = {identifier: object() for identifier in ("02", "04", "05", "10")}
    detector_identity: object | None = None

    def create_session(_cls: type[object], device: str) -> Session:
        assert device == "cpu"
        events.append("session-create")
        return session

    def load_source(root: str | Path, identifier: str) -> object:
        assert Path(root) == tmp_path / "MOT17"
        events.append(f"load:{identifier}")
        return source_objects[identifier]

    def load_detector(
        _cls: type[object], *, device: str, precision: str
    ) -> object:
        assert (device, precision) == ("cpu", "float32")
        events.append("detector-load")
        return detector

    def build_trace(
        source: object,
        actual_detector: object,
        warmup_frames: int,
        *,
        detector_identity: Mapping[str, object],
        hardware_identity: Mapping[str, object],
        telemetry_session: object,
    ) -> RawTrace:
        nonlocal detector_identity_object
        index = list(source_objects.values()).index(source)
        identifier = ("02", "04", "05", "10")[index]
        assert actual_detector is detector
        assert warmup_frames == 7
        if detector_identity_object is None:
            detector_identity_object = detector_identity
        assert detector_identity is detector_identity_object
        assert hardware_identity is hardware
        assert telemetry_session is session
        events.append(f"build:{identifier}")
        return traces[index]

    def derive(actual_traces: Sequence[RawTrace]) -> ReferenceProfile:
        assert list(actual_traces) == traces
        events.append("profile-derive")
        return selected_profile

    def write_profile(self: ReferenceProfile, path: str | Path) -> None:
        assert self is selected_profile
        assert Path(path).name == "training-profile.json"
        events.append("profile-write")

    def seal(
        path: str | Path,
        *,
        trace: RawTrace,
        profile: ReferenceProfile,
        partition: str,
    ) -> Path:
        assert profile is selected_profile
        assert partition == "train"
        assert Path(path).name == f"MOT17-{trace.sequence_id}"
        events.append(f"seal:{trace.sequence_id}")
        return Path(path)

    def forbidden_direct_seal(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise AssertionError("build_partition must call only _seal_trace")

    detector_identity_object: Mapping[str, object] | None = detector_identity
    monkeypatch.setattr(build_mot17_module, "AtomicRun", RecordingAtomicRun)
    monkeypatch.setattr(
        build_mot17_module._HardwareSession, "create", classmethod(create_session)
    )
    monkeypatch.setattr(build_mot17_module, "load_sequence", load_source)
    monkeypatch.setattr(
        build_mot17_module.DFineDetector, "load", classmethod(load_detector)
    )
    monkeypatch.setattr(build_mot17_module, "build_sequence", build_trace)
    monkeypatch.setattr(build_mot17_module, "profile_training_traces", derive)
    monkeypatch.setattr(ReferenceProfile, "write", write_profile)
    monkeypatch.setattr(build_mot17_module, "_seal_trace", seal)
    monkeypatch.setattr(build_mot17_module, "seal_episode", forbidden_direct_seal)

    assert build_mot17_module.build_partition(
        tmp_path / "MOT17",
        destination,
        partition="train",
        device="cpu",
        warmup_frames=7,
    ) == destination.absolute()
    assert session.close_calls == 1
    assert events == [
        "atomic-enter",
        "session-create",
        "load:02",
        "load:04",
        "load:05",
        "load:10",
        "detector-load",
        "build:02",
        "build:04",
        "build:05",
        "build:10",
        "profile-derive",
        "profile-write",
        "seal:02",
        "seal:04",
        "seal:05",
        "seal:10",
        "close",
        "atomic-exit",
    ]


@pytest.mark.parametrize(
    ("partition", "identifiers"),
    [("validation", ("09",)), ("test", ("11", "13"))],
)
def test_frozen_partitions_use_canonical_order_without_copying_or_changing_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
    identifiers: tuple[str, ...],
) -> None:
    profile = _partition_profile(tmp_path / "profile")
    profile_bytes = profile.canonical_json.encode("utf-8")
    events: list[str] = []
    sessions = _install_real_partition_boundaries(
        tmp_path, monkeypatch, events=events
    )

    def forbidden_profile(*args: object, **kwargs: object) -> ReferenceProfile:
        del args, kwargs
        raise AssertionError("frozen partitions must not derive a profile")

    def forbidden_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("frozen partitions must not write a profile")

    monkeypatch.setattr(
        build_mot17_module, "profile_training_traces", forbidden_profile
    )
    monkeypatch.setattr(ReferenceProfile, "write", forbidden_write)
    destination = tmp_path / partition

    assert build_mot17_module.build_partition(
        tmp_path / "MOT17",
        destination,
        partition=partition,
        device="cpu",
        profile=profile,
        warmup_frames=0,
    ) == destination.absolute()

    assert events == [
        "session:cpu",
        *(f"source:{identifier}" for identifier in identifiers),
        "detector:cpu:float32",
    ]
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert {path.name for path in destination.iterdir()} == {
        f"MOT17-{identifier}" for identifier in identifiers
    }
    assert not (destination / "training-profile.json").exists()
    assert profile.canonical_json.encode("utf-8") == profile_bytes
    for identifier in identifiers:
        episode = Episode.open(destination / f"MOT17-{identifier}")
        assert episode.manifest["source"]["split"] == partition  # type: ignore[index]
        assert episode.manifest["cost_profile"]["profile_sha256"] == (  # type: ignore[index]
            profile.profile_sha256
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"partition": "dev"},
        {"partition": 1},
        {"partition": "train", "device": "CPU"},
        {"partition": "train", "device": "cuda:00"},
        {"partition": "train", "device": "cuda:01"},
        {"partition": "train", "device": "cuda:-1"},
        {"partition": "train", "device": "cuda:"},
        {"partition": "train", "device": 0},
        {"partition": "train", "precision": "bfloat16"},
        {"partition": "train", "precision": 32},
        {"partition": "train", "device": "cpu", "precision": "float16"},
        {"partition": "train", "warmup_frames": True},
        {"partition": "train", "warmup_frames": -1},
        {"partition": "train", "warmup_frames": 1.0},
        {"partition": "train", "profile": object()},
        {"partition": "validation", "profile": None},
        {"partition": "validation", "profile": object()},
    ],
)
def test_invalid_partition_arguments_reject_before_atomic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
) -> None:
    atomic_calls = 0

    def forbidden_atomic_run(destination: str | Path) -> AtomicRun:
        del destination
        nonlocal atomic_calls
        atomic_calls += 1
        raise AssertionError("validation must precede AtomicRun")

    monkeypatch.setattr(build_mot17_module, "AtomicRun", forbidden_atomic_run)
    with pytest.raises(ValueError):
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            tmp_path / "destination",
            **arguments,
        )
    assert atomic_calls == 0


@pytest.mark.parametrize(
    ("device", "precision", "device_type"),
    [
        ("cpu", "float32", "cpu"),
        ("cuda", "float32", "cuda"),
        ("cuda:0", "float16", "cuda"),
        ("cuda:17", "float32", "cuda"),
    ],
)
def test_fake_device_precision_matrix_builds_without_real_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    precision: str,
    device_type: str,
) -> None:
    hardware = _hardware_identity(device_type=device_type)
    sessions = _install_real_partition_boundaries(
        tmp_path, monkeypatch, hardware=hardware
    )
    destination = tmp_path / f"{device.replace(':', '-')}-{precision}"

    assert build_mot17_module.build_partition(
        tmp_path / "MOT17",
        destination,
        partition="train",
        device=device,
        precision=precision,
        warmup_frames=0,
    ) == destination.absolute()
    assert len(sessions) == 1
    assert sessions[0].closed is True


def test_frozen_detector_mismatch_rejects_before_atomic_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _partition_profile(tmp_path / "profile")
    calls: list[str] = []

    def forbidden_atomic(destination: str | Path) -> AtomicRun:
        del destination
        calls.append("atomic")
        raise AssertionError

    def forbidden_session(_cls: type[object], device: str) -> object:
        del device
        calls.append("session")
        raise AssertionError

    monkeypatch.setattr(build_mot17_module, "AtomicRun", forbidden_atomic)
    monkeypatch.setattr(
        build_mot17_module._HardwareSession,
        "create",
        classmethod(forbidden_session),
    )

    with pytest.raises(ValueError, match="detector"):
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            tmp_path / "destination",
            partition="validation",
            device="cuda",
            precision="float16",
            profile=profile,
        )
    assert calls == []


def test_frozen_hardware_mismatch_closes_before_source_model_or_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _partition_profile(tmp_path / "profile")
    events: list[str] = []
    sessions = _install_real_partition_boundaries(
        tmp_path,
        monkeypatch,
        hardware=_hardware_identity(device_type="cuda"),
        events=events,
    )
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="hardware"):
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            destination,
            partition="validation",
            device="cuda",
            profile=profile,
        )

    assert events == ["session:cuda"]
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert not destination.exists()


class _PartitionSessionDouble:
    def __init__(
        self,
        hardware_identity: Mapping[str, object],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.hardware_identity = hardware_identity
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _install_stub_partition_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session: _PartitionSessionDouble,
    *,
    failure_phase: str | None = None,
    failure: BaseException | None = None,
    failure_index: int = 0,
) -> None:
    identifiers = ("02", "04", "05", "10")
    traces = [_sealable_trace(tmp_path, identifier) for identifier in identifiers]
    profile = _profile(traces)
    sources = {identifier: object() for identifier in identifiers}
    real_profile_write = ReferenceProfile.write
    counts = {"source": 0, "trace": 0, "seal": 0}

    def maybe_raise(phase: str, index: int = 0) -> None:
        if failure_phase == phase and index == failure_index:
            assert failure is not None
            raise failure

    def create_session(_cls: type[object], device: str) -> _PartitionSessionDouble:
        assert device == "cpu"
        return session

    def load_source(root: str | Path, identifier: str) -> object:
        del root
        index = counts["source"]
        counts["source"] += 1
        maybe_raise("source", index)
        return sources[identifier]

    def load_detector(
        _cls: type[object], *, device: str, precision: str
    ) -> object:
        assert (device, precision) == ("cpu", "float32")
        maybe_raise("model")
        return object()

    def build_trace(
        source: object,
        detector: object,
        warmup_frames: int,
        **kwargs: object,
    ) -> RawTrace:
        del detector, warmup_frames, kwargs
        index = counts["trace"]
        counts["trace"] += 1
        maybe_raise("trace", index)
        identifier = next(name for name, value in sources.items() if value is source)
        return traces[identifiers.index(identifier)]

    def derive(actual_traces: Sequence[RawTrace]) -> ReferenceProfile:
        assert list(actual_traces) == traces
        maybe_raise("profile")
        return profile

    def write_profile(self: ReferenceProfile, path: str | Path) -> None:
        maybe_raise("write")
        real_profile_write(self, path)

    def seal(
        path: str | Path,
        *,
        trace: RawTrace,
        profile: ReferenceProfile,
        partition: str,
    ) -> Path:
        del trace, profile, partition
        index = counts["seal"]
        counts["seal"] += 1
        maybe_raise("seal", index)
        destination = Path(path)
        destination.mkdir()
        (destination / "sentinel").write_bytes(b"sealed")
        return destination

    monkeypatch.setattr(
        build_mot17_module._HardwareSession, "create", classmethod(create_session)
    )
    monkeypatch.setattr(build_mot17_module, "load_sequence", load_source)
    monkeypatch.setattr(
        build_mot17_module.DFineDetector, "load", classmethod(load_detector)
    )
    monkeypatch.setattr(build_mot17_module, "build_sequence", build_trace)
    monkeypatch.setattr(build_mot17_module, "profile_training_traces", derive)
    monkeypatch.setattr(ReferenceProfile, "write", write_profile)
    monkeypatch.setattr(build_mot17_module, "_seal_trace", seal)


@pytest.mark.parametrize(
    ("phase", "failure_index"),
    [
        ("source", 0),
        ("source", 2),
        ("model", 0),
        ("trace", 0),
        ("trace", 2),
        ("profile", 0),
        ("write", 0),
        ("seal", 0),
        ("seal", 2),
    ],
)
def test_every_post_create_failure_closes_once_and_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    failure_index: int,
) -> None:
    session = _PartitionSessionDouble(_hardware_identity())
    primary = RuntimeError(f"{phase}-primary")
    _install_stub_partition_pipeline(
        tmp_path,
        monkeypatch,
        session,
        failure_phase=phase,
        failure=primary,
        failure_index=failure_index,
    )
    destination = tmp_path / "destination"

    with pytest.raises(RuntimeError) as caught:
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            destination,
            partition="train",
            device="cpu",
        )
    assert caught.value is primary
    assert session.close_calls == 1
    assert not destination.exists()


@pytest.mark.parametrize("primary", [RuntimeError("primary"), KeyboardInterrupt(), SystemExit(9)])
def test_secondary_close_failure_never_masks_primary_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    secondary = RuntimeError("secondary-close")
    session = _PartitionSessionDouble(
        _hardware_identity(), close_error=secondary
    )
    _install_stub_partition_pipeline(
        tmp_path,
        monkeypatch,
        session,
        failure_phase="trace",
        failure=primary,
    )

    with pytest.raises(type(primary)) as caught:
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            tmp_path / "destination",
            partition="train",
            device="cpu",
        )
    assert caught.value is primary
    assert session.close_calls == 1
    assert caught.value is not secondary


def test_lone_close_failure_is_primary_and_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_error = RuntimeError("close-primary")
    session = _PartitionSessionDouble(
        _hardware_identity(), close_error=close_error
    )
    _install_stub_partition_pipeline(tmp_path, monkeypatch, session)
    destination = tmp_path / "destination"

    with pytest.raises(RuntimeError) as caught:
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            destination,
            partition="train",
            device="cpu",
        )
    assert caught.value is close_error
    assert session.close_calls == 1
    assert not destination.exists()


def test_publication_failure_occurs_after_one_close_and_does_not_close_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from squint_rl import artifacts

    session = _PartitionSessionDouble(_hardware_identity())
    _install_stub_partition_pipeline(tmp_path, monkeypatch, session)
    destination = tmp_path / "destination"

    def fail_publish(source: object, target: object) -> None:
        del source, target
        raise OSError("publish-primary")

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish-primary"):
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            destination,
            partition="train",
            device="cpu",
        )
    assert session.close_calls == 1
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["directory", "broken-symlink"])
def test_outer_destination_rejection_preserves_target_without_starting_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    destination = tmp_path / "destination"
    if kind == "directory":
        destination.mkdir()
        sentinel = destination / "sentinel"
        sentinel.write_bytes(b"winner")
    else:
        destination.symlink_to(tmp_path / "missing")
        sentinel = None
    session_calls = 0

    def forbidden_session(_cls: type[object], device: str) -> object:
        del device
        nonlocal session_calls
        session_calls += 1
        raise AssertionError

    monkeypatch.setattr(
        build_mot17_module._HardwareSession,
        "create",
        classmethod(forbidden_session),
    )
    with pytest.raises(FileExistsError):
        build_mot17_module.build_partition(
            tmp_path / "MOT17",
            destination,
            partition="train",
            device="cpu",
        )
    assert session_calls == 0
    assert os.path.lexists(destination)
    if sentinel is not None:
        assert sentinel.read_bytes() == b"winner"


def test_competing_atomic_publisher_is_preserved_without_builder_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    session_calls = 0

    def forbidden_session(_cls: type[object], device: str) -> object:
        del device
        nonlocal session_calls
        session_calls += 1
        raise AssertionError

    monkeypatch.setattr(
        build_mot17_module._HardwareSession,
        "create",
        classmethod(forbidden_session),
    )
    with AtomicRun(destination) as winner_work:
        (winner_work / "sentinel").write_bytes(b"winner")
        with pytest.raises(FileExistsError):
            build_mot17_module.build_partition(
                tmp_path / "MOT17",
                destination,
                partition="train",
                device="cpu",
            )
        assert not destination.exists()

    assert session_calls == 0
    assert (destination / "sentinel").read_bytes() == b"winner"


def test_two_fake_train_builds_are_artifact_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _install_real_partition_boundaries(tmp_path, monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_mot17_module.build_partition(
        tmp_path / "MOT17",
        first,
        partition="train",
        device="cpu",
        warmup_frames=0,
    )
    build_mot17_module.build_partition(
        tmp_path / "MOT17",
        second,
        partition="train",
        device="cpu",
        warmup_frames=0,
    )

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert _tree_bytes(first) == _tree_bytes(second)


def _cli_arguments(
    tmp_path: Path,
    *,
    partition: str = "train",
    profile_manifest: Path | None = None,
) -> list[str]:
    arguments = [
        "--mot17-root",
        str(tmp_path / "MOT17"),
        "--output",
        str(tmp_path / partition),
        "--partition",
        partition,
    ]
    if profile_manifest is not None:
        arguments.extend(["--profile-manifest", str(profile_manifest)])
    return arguments


def test_cli_train_defaults_and_exact_sorted_success_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def build(
        mot17_root: str | Path,
        destination: str | Path,
        *,
        partition: str,
        device: str,
        precision: str,
        profile: ReferenceProfile | None,
        warmup_frames: int,
    ) -> Path:
        calls.append(
            {
                "mot17_root": Path(mot17_root),
                "destination": Path(destination),
                "partition": partition,
                "device": device,
                "precision": precision,
                "profile": profile,
                "warmup_frames": warmup_frames,
            }
        )
        return Path(destination).absolute()

    monkeypatch.setattr(build_mot17_module, "build_partition", build)
    assert build_mot17_module.main(_cli_arguments(tmp_path)) == 0

    captured = capsys.readouterr()
    output = (tmp_path / "train").absolute()
    assert captured.out == json.dumps(
        {
            "output_dir": str(output),
            "partition": "train",
            "profile_manifest": str(output / "training-profile.json"),
            "status": "complete",
        },
        sort_keys=True,
    ) + "\n"
    assert captured.err == ""
    assert calls == [
        {
            "mot17_root": tmp_path / "MOT17",
            "destination": tmp_path / "train",
            "partition": "train",
            "device": "cuda",
            "precision": "float32",
            "profile": None,
            "warmup_frames": 10,
        }
    ]


def test_cli_loads_strict_frozen_profile_and_reports_supplied_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = _partition_profile(tmp_path / "profile-source")
    profile_path = tmp_path / "frozen-profile.json"
    profile.write(profile_path)

    def build(
        mot17_root: str | Path,
        destination: str | Path,
        *,
        partition: str,
        device: str,
        precision: str,
        profile: ReferenceProfile | None,
        warmup_frames: int,
    ) -> Path:
        assert Path(mot17_root) == tmp_path / "MOT17"
        assert partition == "validation"
        assert (device, precision, warmup_frames) == ("cuda:3", "float16", 4)
        assert isinstance(profile, ReferenceProfile)
        assert profile.canonical_json == globals_profile.canonical_json
        return Path(destination).absolute()

    globals_profile = profile
    monkeypatch.setattr(build_mot17_module, "build_partition", build)
    arguments = _cli_arguments(
        tmp_path, partition="validation", profile_manifest=profile_path
    )
    arguments.extend(
        ["--device", "cuda:3", "--precision", "float16", "--warmup-frames", "4"]
    )
    assert build_mot17_module.main(arguments) == 0

    captured = capsys.readouterr()
    assert captured.out == json.dumps(
        {
            "output_dir": str((tmp_path / "validation").absolute()),
            "partition": "validation",
            "profile_manifest": str(profile_path.absolute()),
            "status": "complete",
        },
        sort_keys=True,
    ) + "\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--mot17-root", "root"],
        [
            "--mot17-root",
            "root",
            "--output",
            "out",
            "--partition",
            "unknown",
        ],
        [
            "--mot17-root",
            "root",
            "--output",
            "out",
            "--partition",
            "train",
            "--precision",
            "bfloat16",
        ],
        [
            "--mot17-root",
            "root",
            "--output",
            "out",
            "--partition",
            "train",
            "--warmup-frames",
            "not-an-int",
        ],
        ["--unknown-option"],
    ],
)
def test_cli_argparse_errors_propagate_standard_system_exit_two(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        build_mot17_module.main(arguments)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: ")


@pytest.mark.parametrize(
    ("partition", "with_profile"),
    [("train", True), ("validation", False), ("test", False)],
)
def test_cli_conditional_profile_errors_are_one_stderr_line_and_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    partition: str,
    with_profile: bool,
) -> None:
    calls = 0

    def forbidden_build(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError

    monkeypatch.setattr(build_mot17_module, "build_partition", forbidden_build)
    profile_path = tmp_path / "profile.json" if with_profile else None
    arguments = _cli_arguments(
        tmp_path, partition=partition, profile_manifest=profile_path
    )

    assert build_mot17_module.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
    assert calls == 0


def test_cli_noncanonical_profile_is_input_error_without_builder_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}\n", encoding="utf-8")
    calls = 0

    def forbidden_build(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise AssertionError

    monkeypatch.setattr(build_mot17_module, "build_partition", forbidden_build)
    assert build_mot17_module.main(
        _cli_arguments(
            tmp_path, partition="validation", profile_manifest=profile_path
        )
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
    assert calls == 0


def test_cli_strict_mot17_error_is_input_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = Mot17FormatError(
        "bad source inventory",
        sequence="02",
        path=tmp_path / "MOT17",
    )

    def fail_build(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise error

    monkeypatch.setattr(build_mot17_module, "build_partition", fail_build)
    assert build_mot17_module.main(_cli_arguments(tmp_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"


@pytest.mark.parametrize(
    "error",
    [ValueError("runtime-value"), FileExistsError("inner-seal"), RuntimeError("two\nlines")],
)
def test_cli_does_not_broadly_misclassify_runtime_value_or_inner_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail_build(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise error

    monkeypatch.setattr(build_mot17_module, "build_partition", fail_build)
    assert build_mot17_module.main(_cli_arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    expected = str(error).replace("\r", r"\r").replace("\n", r"\n")
    assert captured.err == f"error: {expected}\n"


def test_cli_outer_existing_destination_is_input_exit_two_without_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "train"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"winner")
    session_calls = 0

    def forbidden_session(_cls: type[object], device: str) -> object:
        del device
        nonlocal session_calls
        session_calls += 1
        raise AssertionError

    monkeypatch.setattr(
        build_mot17_module._HardwareSession,
        "create",
        classmethod(forbidden_session),
    )
    assert build_mot17_module.main(
        [*_cli_arguments(tmp_path), "--device", "cpu"]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: completed run already exists:")
    assert session_calls == 0
    assert (destination / "sentinel").read_bytes() == b"winner"


def test_cli_inner_seal_exists_and_publication_failure_are_runtime_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from squint_rl import artifacts

    session = _PartitionSessionDouble(_hardware_identity())
    inner_error = FileExistsError("inner-seal")
    _install_stub_partition_pipeline(
        tmp_path,
        monkeypatch,
        session,
        failure_phase="seal",
        failure=inner_error,
    )
    assert build_mot17_module.main(
        [*_cli_arguments(tmp_path), "--device", "cpu"]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: inner-seal\n"
    assert session.close_calls == 1

    second_root = tmp_path / "publication"
    second_session = _PartitionSessionDouble(_hardware_identity())
    _install_stub_partition_pipeline(second_root, monkeypatch, second_session)

    def fail_publish(source: object, target: object) -> None:
        del source, target
        raise OSError("publish-failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    assert build_mot17_module.main(
        [*_cli_arguments(second_root), "--device", "cpu"]
    ) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: publish-failed\n"
    assert second_session.close_calls == 1


@pytest.mark.parametrize("primary", [KeyboardInterrupt(), SystemExit(7)])
def test_cli_process_control_exceptions_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    def fail_build(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise primary

    monkeypatch.setattr(build_mot17_module, "build_partition", fail_build)
    with pytest.raises(type(primary)) as caught:
        build_mot17_module.main(_cli_arguments(tmp_path))
    assert caught.value is primary


def test_python_m_module_guard_exposes_only_the_binding_parser() -> None:
    root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "squint_rl.reference.build_mot17", "--help"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    for option in (
        "--mot17-root",
        "--output",
        "--partition",
        "--device",
        "--precision",
        "--warmup-frames",
        "--profile-manifest",
    ):
        assert option in completed.stdout
    assert "episode" not in completed.stdout
    assert "benchmark" not in completed.stdout
