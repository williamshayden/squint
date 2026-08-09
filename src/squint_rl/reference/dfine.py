from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from squint_rl.tracker import DetectionBatch

MODEL_ID = "ustc-community/dfine-nano-coco"
MODEL_REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"
WEIGHTS_SHA256 = "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
THRESHOLD = 0.10


class DFineRuntimeError(RuntimeError):
    """The pinned D-FINE reference runtime contract was not satisfied."""


class SnapshotResolver(Protocol):
    def __call__(
        self, *, repo_id: str, revision: str, local_files_only: bool
    ) -> str | Path: ...


def resolve_verified_snapshot(*, resolver: SnapshotResolver | None = None) -> Path:
    if resolver is None:
        from huggingface_hub import snapshot_download

        resolver = snapshot_download

    try:
        snapshot = Path(
            resolver(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
            )
        ).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DFineRuntimeError("pinned D-FINE snapshot is unavailable locally") from error
    if not snapshot.is_dir():
        raise DFineRuntimeError("pinned D-FINE snapshot must be a directory")

    weight_names = tuple(
        sorted(path.name for path in snapshot.glob("*.safetensors") if path.is_file())
    )
    if weight_names != ("model.safetensors",):
        raise DFineRuntimeError("snapshot must contain exactly model.safetensors")

    digest = hashlib.sha256()
    try:
        with (snapshot / "model.safetensors").open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DFineRuntimeError("could not read pinned D-FINE weights") from error
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise DFineRuntimeError("model.safetensors SHA-256 does not match the pin")
    return snapshot


Precision = Literal["float32", "float16"]


@dataclass(slots=True)
class DFineDetector:
    processor: Any
    model: Any
    device: Any
    person_label_id: int
    precision: Precision
    clock: Callable[[], float] = time.perf_counter

    @classmethod
    def load(
        cls,
        *,
        device: str = "cuda",
        precision: Precision = "float32",
        snapshot_resolver: SnapshotResolver | None = None,
    ) -> DFineDetector:
        if precision not in ("float32", "float16"):
            raise ValueError("precision must be float32 or float16")
        if device.partition(":")[0].lower() == "cpu" and precision == "float16":
            raise ValueError("float16 inference is not supported on CPU")

        snapshot = resolve_verified_snapshot(resolver=snapshot_resolver)

        import torch
        from transformers import DFineForObjectDetection, RTDetrImageProcessor

        torch_device = torch.device(device)
        dtype = torch.float32 if precision == "float32" else torch.float16
        processor = RTDetrImageProcessor.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=False,
        )
        if not isinstance(processor, RTDetrImageProcessor):
            raise DFineRuntimeError("snapshot must load RTDetrImageProcessor")
        if processor.size != {"height": 640, "width": 640} or processor.do_pad is not False:
            raise DFineRuntimeError("D-FINE processor must use 640x640 without padding")

        model = DFineForObjectDetection.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        labels = {
            int(label_id): str(label).strip().casefold()
            for label_id, label in model.config.id2label.items()
        }
        person_ids = [label_id for label_id, label in labels.items() if label == "person"]
        if len(person_ids) != 1:
            raise DFineRuntimeError("D-FINE label map must contain exactly one person class")

        model = model.to(device=torch_device, dtype=dtype)
        model = model.eval()
        return cls(processor, model, torch_device, person_ids[0], precision)

    def predict(self, image: Image.Image) -> tuple[DetectionBatch, float]:
        import torch

        raw_inputs = self.processor(images=image, return_tensors="pt")
        inputs: dict[str, Any] = {}
        for name, tensor in raw_inputs.items():
            if self.precision == "float16" and torch.is_floating_point(tensor):
                inputs[name] = tensor.to(device=self.device, dtype=torch.float16)
            else:
                inputs[name] = tensor.to(device=self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = self.clock()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        stopped = self.clock()
        latency_ms = (stopped - started) * 1000.0
        if not np.isfinite(latency_ms) or latency_ms < 0:
            raise DFineRuntimeError("D-FINE forward must produce a finite latency")

        target_sizes = torch.tensor(
            [[image.height, image.width]],
            device=self.device,
        )
        processed = self.processor.post_process_object_detection(
            outputs,
            threshold=THRESHOLD,
            target_sizes=target_sizes,
        )
        if not isinstance(processed, (list, tuple)) or len(processed) != 1:
            raise DFineRuntimeError("D-FINE postprocessor must return exactly one result")
        result = processed[0]
        if not isinstance(result, Mapping) or not {"boxes", "scores", "labels"} <= result.keys():
            raise DFineRuntimeError("D-FINE postprocessor result is malformed")

        boxes = _tensor_array(result["boxes"], "boxes")
        scores = _tensor_array(result["scores"], "scores")
        labels = _tensor_array(result["labels"], "labels")
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise DFineRuntimeError("D-FINE boxes must have shape (N, 4)")
        if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
            raise DFineRuntimeError("D-FINE scores and labels must have shape (N,)")
        if labels.dtype.kind not in "iu":
            raise DFineRuntimeError("D-FINE labels must be integers")
        if not np.all(np.isfinite(boxes)):
            raise DFineRuntimeError("D-FINE boxes must be finite")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise DFineRuntimeError("D-FINE scores must be finite values in [0, 1]")
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise DFineRuntimeError("D-FINE boxes must have positive area")
        if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2] > image.width) or np.any(
            boxes[:, 3] > image.height
        ):
            raise DFineRuntimeError("D-FINE boxes must stay within source dimensions")

        keep = labels == self.person_label_id
        person_boxes = np.asarray(boxes[keep], dtype=np.float32)
        person_scores = np.asarray(scores[keep], dtype=np.float32)
        person_classes = np.ones(len(person_boxes), dtype=np.int64)
        try:
            detections = DetectionBatch(person_boxes, person_scores, person_classes)
        except ValueError as error:
            raise DFineRuntimeError("D-FINE produced an invalid detection batch") from error
        return detections, float(latency_ms)


def _tensor_array(value: Any, name: str) -> NDArray[Any]:
    try:
        return np.asarray(value.detach().cpu().numpy())
    except (AttributeError, TypeError, ValueError) as error:
        raise DFineRuntimeError(f"D-FINE {name} output is not a tensor") from error


def scene_change_grid(previous: Image.Image | None, current: Image.Image) -> NDArray[np.float32]:
    width, height = current.size
    if width < 3 or height < 3:
        raise ValueError("scene images must be at least 3x3 pixels")
    if previous is None:
        return np.zeros((3, 3), dtype=np.float32)
    if previous.size != current.size:
        raise ValueError("scene images must have the same dimensions")

    left = np.asarray(previous.convert("L"), dtype=np.float32)
    right = np.asarray(current.convert("L"), dtype=np.float32)
    difference = np.abs(right - left) / np.float32(255.0)
    rows = np.array_split(difference, 3, axis=0)
    return np.array(
        [[cell.mean() for cell in np.array_split(row, 3, axis=1)] for row in rows],
        dtype=np.float32,
    )
