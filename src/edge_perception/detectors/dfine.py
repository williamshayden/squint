"""Pinned Hugging Face Transformers adapter for D-FINE-N."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import monotonic_ns
from typing import Any, Final, Protocol

import numpy as np
from PIL import Image

from edge_perception.contracts import (
    BatchPrediction,
    Box,
    Detection,
    DetectorIdentity,
    StageTiming,
)

DEFAULT_MODEL_ID: Final = "ustc-community/dfine-nano-coco"
DEFAULT_REVISION: Final = "066438d3d8f0da137a37b38fdf3368fd4afceced"
DEFAULT_THRESHOLD: Final = 0.3
WEIGHTS_FILENAME: Final = "model.safetensors"


class _RuntimeComponents(Protocol):
    @property
    def processor(self) -> Any: ...

    @property
    def model(self) -> Any: ...

    @property
    def torch(self) -> Any: ...

    @property
    def clock(self) -> Callable[[], int]: ...

    @property
    def device(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def weights_sha256(self) -> str: ...

    @property
    def backend_version(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _Dependencies:
    torch: Any
    processor_factory: Any
    model_factory: Any
    resolve_artifact: Callable[..., str]


@dataclass(slots=True)
class _Components:
    processor: Any
    model: Any
    torch: Any
    clock: Callable[[], int]
    device: str
    threshold: float
    model_id: str
    revision: str
    weights_sha256: str
    backend_version: str


def _load_dependencies() -> _Dependencies:
    """Import the optional ML runtime only when model loading is requested."""
    torch = import_module("torch")
    transformers = import_module("transformers")
    huggingface_hub = import_module("huggingface_hub")
    return _Dependencies(
        torch=torch,
        processor_factory=transformers.AutoImageProcessor,
        model_factory=transformers.DFineForObjectDetection,
        resolve_artifact=huggingface_hub.hf_hub_download,
    )


def _select_device(torch: Any, requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"resolved model weights do not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("weights_sha256 must be 64 hexadecimal characters")
    return value.lower()


def _milliseconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def _to_list(value: Any) -> list[Any]:
    converted = value.tolist() if hasattr(value, "tolist") else list(value)
    return list(converted)


class DfineDetector:
    """D-FINE object detector backed by pinned Transformers and PyTorch artifacts."""

    def __init__(self, runtime: _RuntimeComponents) -> None:
        self._processor = runtime.processor
        self._model = runtime.model
        self._torch = runtime.torch
        self._clock = runtime.clock
        self._device = runtime.device
        self._threshold = float(runtime.threshold)
        self._identity = DetectorIdentity(
            adapter="transformers-dfine",
            model_id=runtime.model_id,
            revision=runtime.revision,
            weights_sha256=_validate_sha256(runtime.weights_sha256),
            backend="torch",
            backend_version=runtime.backend_version,
            device=runtime.device,
            dtype="float32",
        )

    @classmethod
    def from_components(cls, runtime: _RuntimeComponents) -> DfineDetector:
        """Construct from an already loaded runtime boundary."""
        return cls(runtime)

    @classmethod
    def load(
        cls,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "auto",
    ) -> DfineDetector:
        dependencies = _load_dependencies()
        selected_device = _select_device(dependencies.torch, device)
        artifact_path = Path(
            dependencies.resolve_artifact(
                repo_id=DEFAULT_MODEL_ID,
                filename=WEIGHTS_FILENAME,
                revision=DEFAULT_REVISION,
            )
        )
        weights_sha256 = _sha256_file(artifact_path)
        processor = dependencies.processor_factory.from_pretrained(
            DEFAULT_MODEL_ID,
            revision=DEFAULT_REVISION,
        )
        model = dependencies.model_factory.from_pretrained(
            DEFAULT_MODEL_ID,
            revision=DEFAULT_REVISION,
        )
        model.to(device=selected_device, dtype=dependencies.torch.float32)
        model.eval()
        return cls(
            _Components(
                processor=processor,
                model=model,
                torch=dependencies.torch,
                clock=monotonic_ns,
                device=selected_device,
                threshold=threshold,
                model_id=DEFAULT_MODEL_ID,
                revision=DEFAULT_REVISION,
                weights_sha256=weights_sha256,
                backend_version=str(dependencies.torch.__version__),
            )
        )

    @property
    def identity(self) -> DetectorIdentity:
        return self._identity

    def warmup(self, image: np.ndarray, runs: int) -> None:
        for _ in range(runs):
            self.predict((image,))

    def predict(self, images: Sequence[np.ndarray]) -> BatchPrediction:
        total_start_ns = self._clock()
        pil_images = [Image.fromarray(image) for image in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = inputs.to(self._device)
        preprocess_end_ns = self._clock()

        if self._device == "cuda":
            self._torch.cuda.synchronize()
        inference_start_ns = self._clock()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        if self._device == "cuda":
            self._torch.cuda.synchronize()
        inference_end_ns = self._clock()

        target_sizes = [(int(image.shape[0]), int(image.shape[1])) for image in images]
        processed = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._threshold,
        )
        if len(processed) != len(images):
            raise RuntimeError(
                "D-FINE postprocessing cardinality mismatch: "
                f"expected {len(images)}, got {len(processed)}"
            )
        detections = tuple(self._convert_image_detections(result) for result in processed)
        postprocess_end_ns = self._clock()
        return BatchPrediction(
            detections=detections,
            timing=StageTiming(
                preprocess_ms=_milliseconds(total_start_ns, preprocess_end_ns),
                inference_ms=_milliseconds(inference_start_ns, inference_end_ns),
                postprocess_ms=_milliseconds(inference_end_ns, postprocess_end_ns),
                total_ms=_milliseconds(total_start_ns, postprocess_end_ns),
            ),
        )

    def peak_device_memory_bytes(self) -> int | None:
        if self._device != "cuda":
            return None
        return int(self._torch.cuda.max_memory_allocated())

    def _convert_image_detections(self, result: dict[str, Any]) -> tuple[Detection, ...]:
        boxes = _to_list(result["boxes"])
        scores = _to_list(result["scores"])
        labels = _to_list(result["labels"])
        id2label = self._model.config.id2label
        converted = []
        for box, score, class_id in zip(boxes, scores, labels, strict=True):
            box_values = _to_list(box)
            numeric_class_id = int(class_id)
            label = id2label.get(numeric_class_id, id2label.get(str(numeric_class_id)))
            converted.append(
                Detection(
                    Box(*(float(value) for value in box_values)),
                    numeric_class_id,
                    float(score),
                    None if label is None else str(label),
                )
            )
        return tuple(converted)
