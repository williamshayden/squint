from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from squint_rl.reference.dfine import (
    MODEL_ID,
    MODEL_REVISION,
    THRESHOLD,
    WEIGHTS_SHA256,
    DFineDetector,
    DFineRuntimeError,
    resolve_verified_snapshot,
    scene_change_grid,
)


class _FakeTensor:
    def __init__(
        self,
        name: str,
        value: object,
        events: list[tuple[object, ...]],
        *,
        floating: bool,
    ) -> None:
        self.name = name
        self.value = np.asarray(value)
        self.events = events
        self.floating = floating

    def to(self, **kwargs: object) -> _FakeTensor:
        self.events.append(("transfer", self.name, kwargs))
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> NDArray[Any]:
        return self.value.copy()


class _FakeInferenceMode:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append(("inference_enter",))

    def __exit__(self, *_: object) -> None:
        self.events.append(("inference_exit",))


def _prediction_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str = "cuda",
    precision: str = "float16",
    results: list[dict[str, _FakeTensor]] | None = None,
    ticks: tuple[float, float] = (10.0, 10.025),
) -> tuple[DFineDetector, list[tuple[object, ...]], dict[str, object]]:
    events: list[tuple[object, ...]] = []
    state: dict[str, object] = {}
    device = SimpleNamespace(type=device_type, spec=f"{device_type}:0")
    torch = ModuleType("torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.is_floating_point = lambda tensor: tensor.floating  # type: ignore[attr-defined]

    def synchronize(selected: object) -> None:
        events.append(("synchronize", selected))

    torch.cuda = SimpleNamespace(synchronize=synchronize)  # type: ignore[attr-defined]
    torch.inference_mode = lambda: _FakeInferenceMode(events)  # type: ignore[attr-defined]

    def tensor(value: object, **kwargs: object) -> _FakeTensor:
        events.append(("target_tensor", value, kwargs))
        return _FakeTensor("target_sizes", value, events, floating=False)

    torch.tensor = tensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Processor:
        def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, _FakeTensor]:
            events.append(("preprocess", images.size, return_tensors))
            return {
                "pixel_values": _FakeTensor(
                    "pixel_values", np.zeros((1, 3, 640, 640)), events, floating=True
                ),
                "pixel_mask": _FakeTensor(
                    "pixel_mask", np.ones((1, 640, 640), dtype=np.bool_), events, floating=False
                ),
            }

        def post_process_object_detection(self, outputs: object, **kwargs: object) -> object:
            events.append(("postprocess", outputs, kwargs))
            state["postprocess_kwargs"] = kwargs
            if results is not None:
                return results
            return [
                {
                    "boxes": _FakeTensor(
                        "boxes",
                        [[1.0, 2.0, 20.0, 30.0], [2.0, 3.0, 10.0, 12.0]],
                        events,
                        floating=True,
                    ),
                    "scores": _FakeTensor("scores", [0.9, 0.8], events, floating=True),
                    "labels": _FakeTensor("labels", [0, 1], events, floating=False),
                }
            ]

    class Model:
        def __call__(self, **inputs: _FakeTensor) -> str:
            events.append(("model", inputs))
            return "outputs"

    tick_values = iter(ticks)

    def clock() -> float:
        events.append(("clock",))
        return next(tick_values)

    detector = DFineDetector(
        Processor(),
        Model(),
        device,
        person_label_id=0,
        precision=precision,  # type: ignore[arg-type]
        clock=clock,
    )
    return detector, events, state


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = b"synthetic safetensors"
    (snapshot / "model.safetensors").write_bytes(weights)
    monkeypatch.setattr(
        "squint_rl.reference.dfine.WEIGHTS_SHA256",
        hashlib.sha256(weights).hexdigest(),
    )
    return snapshot


def _install_loader_fakes(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[object, ...]],
    *,
    id2label: dict[int, str] | None = None,
    processor_size: dict[str, int] | None = None,
) -> None:
    torch = ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]

    def device(value: str) -> SimpleNamespace:
        result = SimpleNamespace(type=value.split(":", maxsplit=1)[0], spec=value)
        events.append(("device", value, result))
        return result

    torch.device = device  # type: ignore[attr-defined]

    class FakeProcessor:
        def __init__(self) -> None:
            self.size = processor_size or {"height": 640, "width": 640}
            self.do_pad = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeProcessor:
            events.append(("processor", path, kwargs))
            return cls()

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(id2label=id2label or {0: "person", 1: "car"})

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeModel:
            events.append(("model", path, kwargs))
            return cls()

        def to(self, **kwargs: object) -> FakeModel:
            events.append(("to", kwargs))
            return self

        def eval(self) -> FakeModel:
            events.append(("eval",))
            return self

    transformers = ModuleType("transformers")
    transformers.RTDetrImageProcessor = FakeProcessor  # type: ignore[attr-defined]
    transformers.DFineForObjectDetection = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_reference_identity_is_pinned() -> None:
    assert MODEL_ID == "ustc-community/dfine-nano-coco"
    assert MODEL_REVISION == "066438d3d8f0da137a37b38fdf3368fd4afceced"
    assert WEIGHTS_SHA256 == (
        "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
    )
    assert THRESHOLD == 0.10


def test_import_does_not_load_accelerator_or_model_runtimes() -> None:
    source_root = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(source_root), environment.get("PYTHONPATH", ""))
        if part
    )
    command = (
        "import json, sys; "
        "import squint_rl.reference.dfine; "
        "print(json.dumps(sorted(name for name in "
        "('torch', 'transformers', 'huggingface_hub', 'pynvml') "
        "if name in sys.modules)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == []


def test_scene_change_is_a_normalized_three_by_three_grid() -> None:
    previous = Image.fromarray(np.zeros((6, 6), np.uint8))
    current_array = np.zeros((6, 6), np.uint8)
    current_array[:2, :2] = 255
    current = Image.fromarray(current_array)

    grid = scene_change_grid(previous, current)

    assert grid.shape == (3, 3)
    assert grid.dtype == np.float32
    assert grid[0, 0] == 1.0
    assert np.count_nonzero(grid) == 1


def test_first_frame_is_zero_without_mutating_the_image() -> None:
    image = Image.fromarray(np.arange(27, dtype=np.uint8).reshape(3, 3, 3), mode="RGB")
    before = np.asarray(image).copy()

    grid = scene_change_grid(None, image)

    np.testing.assert_array_equal(grid, np.zeros((3, 3), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(image), before)


@pytest.mark.parametrize("size", [(2, 3), (3, 2)])
def test_scene_change_rejects_images_smaller_than_grid(size: tuple[int, int]) -> None:
    image = Image.new("L", size)

    with pytest.raises(ValueError, match="at least 3x3"):
        scene_change_grid(None, image)


def test_scene_change_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimensions"):
        scene_change_grid(Image.new("L", (3, 3)), Image.new("RGB", (4, 3)))


def test_snapshot_resolution_is_local_pinned_and_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = b"synthetic safetensors"
    (snapshot / "model.safetensors").write_bytes(weights)
    expected_digest = hashlib.sha256(weights).hexdigest()
    monkeypatch.setattr("squint_rl.reference.dfine.WEIGHTS_SHA256", expected_digest)
    calls: list[dict[str, object]] = []

    def resolver(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    resolved = resolve_verified_snapshot(resolver=resolver)

    assert resolved == snapshot.resolve()
    assert calls == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_files_only": True,
        }
    ]


def test_snapshot_rejects_wrong_or_extra_weight_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"wrong")

    with pytest.raises(DFineRuntimeError, match="SHA-256"):
        resolve_verified_snapshot(resolver=lambda **_: snapshot)

    digest = hashlib.sha256(b"wrong").hexdigest()
    monkeypatch.setattr("squint_rl.reference.dfine.WEIGHTS_SHA256", digest)
    (snapshot / "unexpected.safetensors").write_bytes(b"extra")

    with pytest.raises(DFineRuntimeError, match="exactly model.safetensors"):
        resolve_verified_snapshot(resolver=lambda **_: snapshot)


def test_cpu_float16_is_rejected_before_snapshot_resolution() -> None:
    def unexpected_resolver(**_: object) -> Path:
        raise AssertionError("snapshot resolution must not run")

    with pytest.raises(ValueError, match="float16.*CPU"):
        DFineDetector.load(
            device="cpu",
            precision="float16",
            snapshot_resolver=unexpected_resolver,
        )


def test_load_uses_verified_local_direct_transformers_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path, monkeypatch)
    events: list[tuple[object, ...]] = []
    _install_loader_fakes(monkeypatch, events)

    detector = DFineDetector.load(
        device="cpu",
        precision="float32",
        snapshot_resolver=lambda **_: snapshot,
    )

    processor_event = next(event for event in events if event[0] == "processor")
    model_event = next(event for event in events if event[0] == "model")
    assert processor_event == (
        "processor",
        str(snapshot.resolve()),
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": False,
        },
    )
    assert model_event == (
        "model",
        str(snapshot.resolve()),
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "float32",
        },
    )
    assert [event[0] for event in events].count("to") == 1
    assert [event[0] for event in events].count("eval") == 1
    assert detector.person_label_id == 0
    assert detector.device.type == "cpu"
    assert detector.precision == "float32"


@pytest.mark.parametrize(
    "id2label",
    [
        {1: "car"},
        {0: "person", 1: "PERSON"},
    ],
)
def test_load_requires_exactly_one_person_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    id2label: dict[int, str],
) -> None:
    snapshot = _snapshot(tmp_path, monkeypatch)
    events: list[tuple[object, ...]] = []
    _install_loader_fakes(monkeypatch, events, id2label=id2label)

    with pytest.raises(DFineRuntimeError, match="exactly one person"):
        DFineDetector.load(
            device="cpu",
            precision="float32",
            snapshot_resolver=lambda **_: snapshot,
        )


def test_load_rejects_non_pinned_processor_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path, monkeypatch)
    events: list[tuple[object, ...]] = []
    _install_loader_fakes(
        monkeypatch,
        events,
        processor_size={"height": 800, "width": 800},
    )

    with pytest.raises(DFineRuntimeError, match="640x640"):
        DFineDetector.load(
            device="cpu",
            precision="float32",
            snapshot_resolver=lambda **_: snapshot,
        )


def test_predict_times_only_forward_and_returns_person_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, events, state = _prediction_fixture(monkeypatch)
    image = Image.new("RGB", (40, 32))

    detections, latency_ms = detector.predict(image)

    assert [event[0] for event in events] == [
        "preprocess",
        "transfer",
        "transfer",
        "synchronize",
        "clock",
        "inference_enter",
        "model",
        "inference_exit",
        "synchronize",
        "clock",
        "target_tensor",
        "postprocess",
    ]
    transfer_events = [event for event in events if event[0] == "transfer"]
    assert transfer_events[0][2] == {"device": detector.device, "dtype": "float16"}
    assert transfer_events[1][2] == {"device": detector.device}
    postprocess_kwargs = state["postprocess_kwargs"]
    assert isinstance(postprocess_kwargs, dict)
    assert postprocess_kwargs["threshold"] == 0.10
    target_sizes = postprocess_kwargs["target_sizes"]
    assert isinstance(target_sizes, _FakeTensor)
    np.testing.assert_array_equal(target_sizes.value, [[32, 40]])
    assert latency_ms == pytest.approx(25.0)
    np.testing.assert_array_equal(detections.boxes_xyxy, [[1.0, 2.0, 20.0, 30.0]])
    np.testing.assert_array_equal(detections.scores, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(detections.class_ids, [1])
    assert detections.boxes_xyxy.dtype == np.float32
    assert detections.scores.dtype == np.float32
    assert detections.class_ids.dtype == np.int64
    assert not detections.boxes_xyxy.flags.writeable


def test_cpu_predict_does_not_synchronize_or_cast_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, events, _ = _prediction_fixture(
        monkeypatch,
        device_type="cpu",
        precision="float32",
    )

    detector.predict(Image.new("RGB", (40, 32)))

    assert not any(event[0] == "synchronize" for event in events)
    assert [event[2] for event in events if event[0] == "transfer"] == [
        {"device": detector.device},
        {"device": detector.device},
    ]


def test_predict_rejects_nonfinite_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    detector, _, _ = _prediction_fixture(monkeypatch, ticks=(0.0, float("nan")))

    with pytest.raises(DFineRuntimeError, match="finite latency"):
        detector.predict(Image.new("RGB", (40, 32)))


@pytest.mark.parametrize(
    ("boxes", "scores", "labels", "message"),
    [
        ([[1.0, 2.0, 41.0, 30.0]], [0.9], [0], "source dimensions"),
        ([[1.0, 2.0, 20.0, 30.0]], [float("nan")], [0], "scores"),
        ([[1.0, 2.0, 1.0, 30.0]], [0.9], [0], "positive area"),
        ([[1.0, 2.0, 20.0]], [0.9], [0], "shape"),
    ],
)
def test_predict_rejects_malformed_postprocessor_output(
    monkeypatch: pytest.MonkeyPatch,
    boxes: object,
    scores: object,
    labels: object,
    message: str,
) -> None:
    events: list[tuple[object, ...]] = []
    results = [
        {
            "boxes": _FakeTensor("boxes", boxes, events, floating=True),
            "scores": _FakeTensor("scores", scores, events, floating=True),
            "labels": _FakeTensor("labels", labels, events, floating=False),
        }
    ]
    detector, _, _ = _prediction_fixture(monkeypatch, results=results)

    with pytest.raises(DFineRuntimeError, match=message):
        detector.predict(Image.new("RGB", (40, 32)))


def test_predict_rejects_unexpected_batch_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, _, _ = _prediction_fixture(monkeypatch, results=[])

    with pytest.raises(DFineRuntimeError, match="one result"):
        detector.predict(Image.new("RGB", (40, 32)))


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("SQUINT_RUN_MODEL_TESTS") != "1",
    reason="set SQUINT_RUN_MODEL_TESTS=1 during a coordinated model-test window",
)
def test_pinned_model_smoke_is_opt_in() -> None:
    snapshot = os.environ.get("SQUINT_DFINE_SNAPSHOT")
    if snapshot is None:
        pytest.fail("SQUINT_DFINE_SNAPSHOT must name the verified local snapshot")
    detector = DFineDetector.load(
        device=os.environ.get("SQUINT_DFINE_DEVICE", "cuda"),
        precision="float32",
        snapshot_resolver=lambda **_: snapshot,
    )
    image = Image.fromarray(
        np.arange(640 * 640 * 3, dtype=np.uint32).reshape(640, 640, 3).astype(np.uint8),
        mode="RGB",
    )

    detections, latency_ms = detector.predict(image)

    assert np.isfinite(latency_ms) and latency_ms >= 0
    assert detections.boxes_xyxy.shape[1:] == (4,)
