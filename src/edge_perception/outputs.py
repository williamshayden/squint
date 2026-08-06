"""Deterministic, JSON-native artifacts for one checkpoint run."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from math import isfinite
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image, ImageDraw

from edge_perception.contracts import Detection, Region

SCHEMA_VERSION = "0.1.0"


def _json_dump(record: Mapping[str, Any]) -> str:
    """Serialize one record, rejecting non-finite or non-JSON-native values."""
    return json.dumps(record, allow_nan=False, sort_keys=True)


def _atomic_json_write(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically replace a JSON document after it has serialized successfully."""
    encoded = _json_dump(record)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _record(run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "run_id": run_id, "schema_version": SCHEMA_VERSION}


def _region_color(region_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(region_id.encode("utf-8")).digest()
    return (64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160)


def summarize_latencies(records: Iterable[float | int]) -> dict[str, int | float | None]:
    """Return finite linear-percentile latency statistics in milliseconds."""
    values = [float(value) for value in records]
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    if not all(isfinite(value) for value in values):
        raise ValueError("latencies must be finite")
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "p99_ms": float(np.percentile(values, 99, method="linear")),
    }


class RunOutputs(AbstractContextManager["RunOutputs"]):
    """Own the structured files and diagnostic frames for a single run directory."""

    def __init__(self, run_dir: Path | str, *, run_id: str, manifest: Mapping[str, Any]) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._annotated_dir = self.run_dir / "annotated"
        self._annotated_dir.mkdir(exist_ok=True)
        _atomic_json_write(self.run_dir / "manifest.json", _record(run_id, manifest))
        self._streams = {
            "inferences": (self.run_dir / "inferences.jsonl").open("w", encoding="utf-8", newline="\n"),
            "detections": (self.run_dir / "detections.jsonl").open("w", encoding="utf-8", newline="\n"),
            "hardware": (self.run_dir / "hardware.jsonl").open("w", encoding="utf-8", newline="\n"),
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

    def close(self) -> None:
        """Close append-only streams; safe to call more than once."""
        for stream in self._streams.values():
            if not stream.closed:
                stream.close()

    def _append(self, stream_name: str, payload: Mapping[str, Any]) -> None:
        stream = self._streams[stream_name]
        if stream.closed:
            raise RuntimeError("run output streams are closed")
        stream.write(_json_dump(_record(self.run_id, payload)) + "\n")

    def write_inference(self, record: Mapping[str, Any]) -> None:
        """Append one inference record, retaining caller-provided join identifiers."""
        self._append("inferences", record)

    def write_detections(
        self,
        *,
        frame_id: str,
        inference_id: str,
        region_id: str,
        detections: Iterable[Detection],
    ) -> None:
        """Append flattened, source-space detection records for one inference."""
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                raise TypeError("detections must contain Detection values")
            self._append(
                "detections",
                {
                    "frame_id": frame_id,
                    "inference_id": inference_id,
                    "region_id": region_id,
                    "detection_index": detection_index,
                    **detection.to_dict(),
                },
            )

    def write_hardware(self, record: Mapping[str, Any]) -> None:
        """Append one hardware telemetry record."""
        self._append("hardware", record)

    def flush_frame(self) -> None:
        """Make all completed-frame rows durable enough for interruption recovery."""
        for stream in self._streams.values():
            stream.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Atomically publish the current run summary."""
        _atomic_json_write(self.run_dir / "summary.json", _record(self.run_id, summary))

    def annotate(
        self,
        frame_index: int,
        frame: np.ndarray,
        *,
        regions: Iterable[Region],
        detections: Iterable[Detection],
    ) -> Path:
        """Render regions and mapped detections to a lossless diagnostic PNG."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be an RGB uint8 array")
        image = Image.fromarray(frame, mode="RGB").copy()
        drawing = ImageDraw.Draw(image)
        for region in regions:
            if not isinstance(region, Region):
                raise TypeError("regions must contain Region values")
            x2 = region.x + region.width - 1
            y2 = region.y + region.height - 1
            drawing.rectangle((region.x, region.y, x2, y2), outline=_region_color(region.region_id), width=2)
        for detection in detections:
            if not isinstance(detection, Detection):
                raise TypeError("detections must contain Detection values")
            box = detection.box
            drawing.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(255, 255, 255), width=2)
            label = detection.label if detection.label is not None else str(detection.class_id)
            drawing.text((box.x1, max(0.0, box.y1 - 12.0)), f"{label} {detection.score:.3f}", fill=(255, 255, 255))
        path = self._annotated_dir / f"{frame_index:06d}.png"
        image.save(path, format="PNG")
        return path
