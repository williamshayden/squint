from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from edge_perception.contracts import BatchPrediction, Box, Detection
from edge_perception.detector import Detector
from edge_perception.detectors import dfine
from edge_perception.detectors.dfine import DfineDetector


class FakeInferenceMode(AbstractContextManager[None]):
    def __init__(self, torch: FakeTorch) -> None:
        self._torch = torch

    def __enter__(self) -> None:
        self._torch.inference_mode_entries += 1

    def __exit__(self, *args: object) -> None:
        return None


class FakeCuda:
    def __init__(self, *, available: bool = False) -> None:
        self.available = available
        self.synchronize_calls = 0
        self.peak_bytes = 123_456

    def is_available(self) -> bool:
        return self.available

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak_bytes


class FakeTorch:
    __version__ = "2.11.0-fake"
    float32 = "float32"

    def __init__(self, *, cuda_available: bool = False) -> None:
        self.cuda = FakeCuda(available=cuda_available)
        self.inference_mode_entries = 0

    @property
    def cuda_synchronize_calls(self) -> int:
        return self.cuda.synchronize_calls

    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode(self)


class FakeInputs(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(pixel_values="pixels")
        self.moved_to: str | None = None

    def to(self, device: str) -> FakeInputs:
        self.moved_to = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.image_modes: list[str] = []
        self.target_sizes: list[tuple[int, int]] = []
        self.threshold: float | None = None

    def __call__(self, *, images: list[Any], return_tensors: str) -> FakeInputs:
        assert return_tensors == "pt"
        self.image_modes = [image.mode for image in images]
        return self.inputs

    def post_process_object_detection(
        self,
        outputs: object,
        *,
        target_sizes: list[tuple[int, int]],
        threshold: float,
    ) -> list[dict[str, np.ndarray[Any, Any]]]:
        assert outputs == {"logits": "outputs"}
        self.target_sizes = target_sizes
        self.threshold = threshold
        return [
            {
                "boxes": np.array([[1, 2, 11, 22]], dtype=np.int64),
                "scores": np.array([np.float32(0.875)]),
                "labels": np.array([1], dtype=np.int64),
            }
            for _ in target_sizes
        ]


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={1: "person"})
        self.eval_calls = 0
        self.moved_to: list[tuple[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def to(self, *, device: str, dtype: object) -> FakeModel:
        self.moved_to.append((device, dtype))
        return self

    def __call__(self, **inputs: object) -> dict[str, str]:
        self.calls.append(inputs)
        return {"logits": "outputs"}


class FakeClock:
    def __init__(self) -> None:
        self._now = 0

    def __call__(self) -> int:
        self._now += 1_000_000
        return self._now


@dataclass
class FakeRuntime:
    processor: FakeProcessor = field(default_factory=FakeProcessor)
    model: FakeModel = field(default_factory=FakeModel)
    torch: FakeTorch = field(default_factory=FakeTorch)
    clock: FakeClock = field(default_factory=FakeClock)
    device: str = "cpu"
    threshold: float = 0.3
    model_id: str = "fake/dfine"
    revision: str = "revision-123"
    weights_sha256: str = "a" * 64

    @property
    def backend_version(self) -> str:
        return self.torch.__version__


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


def test_dfine_requests_output_boxes_in_each_input_image_size(fake_runtime: FakeRuntime) -> None:
    detector = DfineDetector.from_components(fake_runtime)
    images = (
        np.zeros((1080, 1280, 3), dtype=np.uint8),
        np.zeros((2160, 3840, 3), dtype=np.uint8),
    )

    result = detector.predict(images)

    assert fake_runtime.processor.target_sizes == [(1080, 1280), (2160, 3840)]
    assert len(result.detections) == 2


def test_dfine_uses_inference_mode_and_synchronizes_cuda(fake_runtime: FakeRuntime) -> None:
    fake_runtime.device = "cuda"
    DfineDetector.from_components(fake_runtime).predict(
        (np.zeros((32, 32, 3), dtype=np.uint8),)
    )

    assert fake_runtime.torch.inference_mode_entries == 1
    assert fake_runtime.torch.cuda_synchronize_calls == 2


def test_dfine_returns_backend_neutral_float_detections_and_stage_timing(
    fake_runtime: FakeRuntime,
) -> None:
    result = DfineDetector.from_components(fake_runtime).predict(
        (np.zeros((32, 48, 3), dtype=np.uint8),)
    )

    assert result == BatchPrediction(
        detections=((Detection(Box(1.0, 2.0, 11.0, 22.0), 1, 0.875, "person"),),),
        timing=result.timing,
    )
    assert result.timing.to_dict() == {
        "preprocess_ms": 1.0,
        "inference_ms": 1.0,
        "postprocess_ms": 1.0,
        "total_ms": 4.0,
    }
    assert fake_runtime.processor.image_modes == ["RGB"]
    assert fake_runtime.processor.inputs.moved_to == "cpu"
    assert fake_runtime.model.calls == [{"pixel_values": "pixels"}]
    assert isinstance(result.detections[0][0].score, float)
    assert all(isinstance(value, float) for value in result.detections[0][0].box.to_list())


def test_dfine_warmup_runs_predictions_without_returning_measurements(
    fake_runtime: FakeRuntime,
) -> None:
    detector = DfineDetector.from_components(fake_runtime)

    result = detector.warmup(np.zeros((16, 16, 3), dtype=np.uint8), runs=2)

    assert result is None
    assert len(fake_runtime.model.calls) == 2


def test_dfine_reports_peak_cuda_memory_only_for_cuda(fake_runtime: FakeRuntime) -> None:
    cpu_detector = DfineDetector.from_components(fake_runtime)
    fake_runtime.device = "cuda"
    cuda_detector = DfineDetector.from_components(fake_runtime)

    assert cpu_detector.peak_device_memory_bytes() is None
    assert cuda_detector.peak_device_memory_bytes() == 123_456


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def from_pretrained(self, model_id: str, *, revision: str) -> object:
        self.calls.append((model_id, revision))
        return self.value


@dataclass
class FakeDependencies:
    torch: FakeTorch
    processor_factory: FakeFactory
    model_factory: FakeFactory
    artifact_path: Path
    resolve_calls: list[tuple[str, str, str]] = field(default_factory=list)

    def resolve_artifact(self, *, repo_id: str, filename: str, revision: str) -> str:
        self.resolve_calls.append((repo_id, filename, revision))
        return str(self.artifact_path)


def fake_dependencies(tmp_path: Path, *, cuda_available: bool = False) -> FakeDependencies:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pinned fake weights")
    return FakeDependencies(
        torch=FakeTorch(cuda_available=cuda_available),
        processor_factory=FakeFactory(FakeProcessor()),
        model_factory=FakeFactory(FakeModel()),
        artifact_path=weights,
    )


def test_load_preserves_revision_hashes_weights_and_places_fp32_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = fake_dependencies(tmp_path)
    monkeypatch.setattr(dfine, "_load_dependencies", lambda: dependencies)

    detector = DfineDetector.load(device="cpu")

    expected_sha256 = hashlib.sha256(b"pinned fake weights").hexdigest()
    assert dependencies.resolve_calls == [
        (dfine.DEFAULT_MODEL_ID, dfine.WEIGHTS_FILENAME, dfine.DEFAULT_REVISION)
    ]
    assert dependencies.processor_factory.calls == [
        (dfine.DEFAULT_MODEL_ID, dfine.DEFAULT_REVISION)
    ]
    assert dependencies.model_factory.calls == [(dfine.DEFAULT_MODEL_ID, dfine.DEFAULT_REVISION)]
    model = dependencies.model_factory.value
    assert isinstance(model, FakeModel)
    assert model.eval_calls == 1
    assert model.moved_to == [("cpu", dependencies.torch.float32)]
    assert detector.identity.to_dict() == {
        "adapter": "transformers-dfine",
        "model_id": dfine.DEFAULT_MODEL_ID,
        "revision": dfine.DEFAULT_REVISION,
        "weights_sha256": expected_sha256,
        "backend": "torch",
        "backend_version": "2.11.0-fake",
        "device": "cpu",
        "dtype": "float32",
    }
    assert len(detector.identity.weights_sha256) == 64


def test_load_auto_falls_back_explicitly_to_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = fake_dependencies(tmp_path, cuda_available=False)
    monkeypatch.setattr(dfine, "_load_dependencies", lambda: dependencies)

    detector = DfineDetector.load(device="auto")

    assert detector.identity.device == "cpu"
    model = dependencies.model_factory.value
    assert isinstance(model, FakeModel)
    assert model.moved_to == [("cpu", dependencies.torch.float32)]


def test_load_rejects_requested_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = fake_dependencies(tmp_path, cuda_available=False)
    monkeypatch.setattr(dfine, "_load_dependencies", lambda: dependencies)

    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        DfineDetector.load(device="cuda")


def test_load_rejects_unknown_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dependencies = fake_dependencies(tmp_path)
    monkeypatch.setattr(dfine, "_load_dependencies", lambda: dependencies)

    with pytest.raises(ValueError, match="device must be one of: auto, cpu, cuda"):
        DfineDetector.load(device="tpu")


def test_load_fails_clearly_when_resolved_weights_are_not_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = fake_dependencies(tmp_path)
    dependencies.artifact_path = tmp_path / "missing.safetensors"
    monkeypatch.setattr(dfine, "_load_dependencies", lambda: dependencies)

    with pytest.raises(FileNotFoundError, match="resolved model weights do not exist"):
        DfineDetector.load(device="cpu")


def test_dfine_satisfies_dependency_neutral_detector_protocol(fake_runtime: FakeRuntime) -> None:
    detector: Detector = DfineDetector.from_components(fake_runtime)

    assert detector.identity.model_id == "fake/dfine"


def test_dfine_rejects_malformed_weight_sha256(fake_runtime: FakeRuntime) -> None:
    fake_runtime.weights_sha256 = "not-a-sha256"

    with pytest.raises(ValueError, match="weights_sha256 must be 64 hexadecimal characters"):
        DfineDetector.from_components(fake_runtime)
