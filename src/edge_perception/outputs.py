"""Deterministic, JSON-native artifacts for one checkpoint run."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Self, TextIO, cast
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from edge_perception.contracts import Detection, Region

SCHEMA_VERSION = "0.1.0"
_OWNER_FILENAME = ".run-outputs-owner"
_MOVEFILE_REPLACE_EXISTING = 0x1
_MOVEFILE_WRITE_THROUGH = 0x8


def _json_dump(record: Mapping[str, Any]) -> str:
    """Serialize one record, rejecting non-finite or non-JSON-native values."""
    return json.dumps(record, allow_nan=False, sort_keys=True)


def _load_windows_move_file_ex() -> Callable[[str, str, int], int]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file_ex.restype = wintypes.BOOL
    return cast(Callable[[str, str, int], int], move_file_ex)


def _windows_error() -> OSError:
    import ctypes

    error_code = int(ctypes.get_last_error())
    return ctypes.WinError(error_code)


def _replace_file_windows(source: Path, destination: Path) -> None:
    move_file_ex = _load_windows_move_file_ex()
    flags = _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH
    if not move_file_ex(str(source), str(destination), flags):
        raise _windows_error()


def _sync_directory(directory: Path, *, platform: str | None = None) -> None:
    """Flush directory metadata on platforms that expose directory fsync."""
    platform_name = os.name if platform is None else platform
    if platform_name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(
    source: Path,
    destination: Path,
    *,
    platform: str | None = None,
) -> None:
    """Atomically replace a file and durably commit its directory entry."""
    platform_name = os.name if platform is None else platform
    if platform_name == "nt":
        _replace_file_windows(source, destination)
        return
    os.replace(source, destination)
    _sync_directory(destination.parent, platform=platform_name)


def _durable_unlink(path: Path, *, platform: str | None = None) -> None:
    """Remove a directory entry and flush its parent where supported."""
    platform_name = os.name if platform is None else platform
    path.unlink()
    _sync_directory(path.parent, platform=platform_name)


def _atomic_json_write(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically replace a JSON document after it has serialized successfully."""
    encoded = _json_dump(record)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "run_id": run_id, "schema_version": SCHEMA_VERSION}


def _region_color(region_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(region_id.encode("utf-8")).digest()
    return (64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160)


def _milliseconds(elapsed_ns: int) -> float:
    return elapsed_ns / 1_000_000.0


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
        self._streams: dict[str, TextIO] = {}
        self._owner_path = self.run_dir / _OWNER_FILENAME
        run_dir_existed = self.run_dir.exists()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self._owner_path.open("x", encoding="utf-8", newline="\n") as owner:
                owner.write(f"{run_id}\n")
        except FileExistsError as error:
            raise ValueError(f"output directory is already owned: {self.run_dir}") from error

        created_paths = [self._owner_path]
        try:
            if set(self.run_dir.iterdir()) != {self._owner_path}:
                raise ValueError(f"output directory must be empty: {self.run_dir}")

            self._annotated_dir = self.run_dir / "annotated"
            self._annotated_dir.mkdir()
            created_paths.append(self._annotated_dir)

            manifest_path = self.run_dir / "manifest.json"
            _atomic_json_write(manifest_path, _record(run_id, manifest))
            created_paths.append(manifest_path)

            for stream_name in ("inferences", "detections", "hardware"):
                stream_path = self.run_dir / f"{stream_name}.jsonl"
                self._streams[stream_name] = stream_path.open(
                    "x", encoding="utf-8", newline="\n"
                )
                created_paths.append(stream_path)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve setup error
                error.add_note(f"failed to close a created JSONL stream: {cleanup_error}")
            for path in reversed(created_paths):
                try:
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(f"failed to remove created path {path}: {cleanup_error}")
            if not run_dir_existed:
                try:
                    self.run_dir.rmdir()
                except OSError:
                    pass
            raise

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

    def _serialized_record(self, payload: Mapping[str, Any]) -> str:
        return _json_dump(_record(self.run_id, payload)) + "\n"

    @staticmethod
    def _detection_records(
        *,
        frame_id: str,
        inference_id: str,
        region_id: str,
        detections: Iterable[Detection],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                raise TypeError("detections must contain Detection values")
            records.append(
                {
                    "frame_id": frame_id,
                    "inference_id": inference_id,
                    "region_id": region_id,
                    "detection_index": detection_index,
                    **detection.to_dict(),
                }
            )
        return records

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
        records = self._detection_records(
            frame_id=frame_id,
            inference_id=inference_id,
            region_id=region_id,
            detections=detections,
        )
        for record in records:
            self._append("detections", record)

    def write_hardware(self, record: Mapping[str, Any]) -> None:
        """Append one hardware telemetry record."""
        self._append("hardware", record)

    def flush_frame(self) -> None:
        """Make all completed-frame rows durable enough for interruption recovery."""
        for stream in self._streams.values():
            stream.flush()

    def commit_frame(
        self,
        *,
        inferences: Iterable[Mapping[str, Any]],
        detection_batches: Iterable[tuple[str, str, str, Iterable[Detection]]],
        annotation: tuple[int, np.ndarray, Iterable[Region], Iterable[Detection]] | None,
    ) -> tuple[float | None, float]:
        """Publish one complete frame, rolling back every stream on failure."""
        serialization_started_ns = perf_counter_ns()
        serialized_inferences = [self._serialized_record(record) for record in inferences]
        serialized_detections = [
            self._serialized_record(record)
            for frame_id, inference_id, region_id, detections in detection_batches
            for record in self._detection_records(
                frame_id=frame_id,
                inference_id=inference_id,
                region_id=region_id,
                detections=detections,
            )
        ]
        serialization_ns = perf_counter_ns() - serialization_started_ns

        prepared_annotation: tuple[Path, Path] | None = None
        annotation_ns: int | None = None
        if annotation is not None:
            annotation_started_ns = perf_counter_ns()
            prepared_annotation = self._prepare_annotation(*annotation)
            annotation_ns = perf_counter_ns() - annotation_started_ns

        positions: dict[str, int] = {}
        annotation_published = False
        try:
            for stream_name, stream in self._streams.items():
                if stream.closed:
                    raise RuntimeError("run output streams are closed")
                positions[stream_name] = stream.tell()

            publication_started_ns = perf_counter_ns()
            self._streams["inferences"].writelines(serialized_inferences)
            self._streams["detections"].writelines(serialized_detections)
            self.flush_frame()
            serialization_ns += perf_counter_ns() - publication_started_ns

            if prepared_annotation is not None:
                temporary, canonical = prepared_annotation
                annotation_publish_started_ns = perf_counter_ns()
                os.replace(temporary, canonical)
                annotation_ns = (annotation_ns or 0) + (
                    perf_counter_ns() - annotation_publish_started_ns
                )
                annotation_published = True
        except BaseException as error:
            for stream_name, position in positions.items():
                stream = self._streams[stream_name]
                try:
                    stream.seek(position)
                    stream.truncate()
                    stream.flush()
                except Exception as rollback_error:  # noqa: BLE001 - preserve primary error
                    error.add_note(
                        f"failed to restore {stream_name}.jsonl: {rollback_error}"
                    )
            if annotation_published and prepared_annotation is not None:
                prepared_annotation[1].unlink(missing_ok=True)
            raise
        finally:
            if prepared_annotation is not None:
                prepared_annotation[0].unlink(missing_ok=True)

        return (
            None if annotation_ns is None else _milliseconds(annotation_ns),
            _milliseconds(serialization_ns),
        )

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Close streams, publish the terminal summary, then release ownership."""
        self._close_durable_streams()
        _atomic_json_write(self.run_dir / "summary.json", _record(self.run_id, summary))
        _durable_unlink(self._owner_path)

    def _close_durable_streams(self) -> None:
        for stream in self._streams.values():
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
        self.close()

    def annotate(
        self,
        frame_index: int,
        frame: np.ndarray,
        *,
        regions: Iterable[Region],
        detections: Iterable[Detection],
    ) -> Path:
        """Render regions and mapped detections to a lossless diagnostic PNG."""
        temporary, path = self._prepare_annotation(
            frame_index,
            frame,
            regions,
            detections,
        )
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _prepare_annotation(
        self,
        frame_index: int,
        frame: np.ndarray,
        regions: Iterable[Region],
        detections: Iterable[Detection],
    ) -> tuple[Path, Path]:
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
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            image.save(temporary, format="PNG")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return temporary, path
