"""Detector-free validation for configured checkpoint runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from edge_perception.config import RunConfig
from edge_perception.geometry import validate_region
from edge_perception.video import iter_video


@dataclass(frozen=True, slots=True)
class RunPreflight:
    frame_width: int
    frame_height: int
    source_sha256: str


def validate_output_directory(output_dir: Path) -> None:
    """Allow absent or empty run directories while preserving existing contents."""

    resolved = Path(output_dir).resolve()
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise ValueError(f"output is not a directory: {resolved}")
    if any(resolved.iterdir()):
        raise ValueError(f"output directory must be empty: {resolved}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_run(config: RunConfig) -> RunPreflight:
    """Validate source, output, regions, and capture provenance without a detector."""

    source_path = config.input_path
    if not source_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {source_path}")
    validate_output_directory(config.output_dir)

    frames = iter_video(source_path)
    try:
        try:
            first_frame = next(frames)
        except StopIteration as error:
            raise ValueError(f"video contains no decoded frames: {source_path}") from error
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()

    frame_height, frame_width, _channels = first_frame.image.shape
    for region in config.regions:
        validate_region(region, frame_width, frame_height)

    source_sha256 = _sha256_file(source_path)
    capture = config.capture
    if capture is not None:
        if capture.path.resolve() != source_path.resolve():
            raise ValueError("capture path must equal input path")
        if capture.sha256 != source_sha256:
            raise ValueError("capture SHA-256 must match source SHA-256")

    return RunPreflight(
        frame_width=frame_width,
        frame_height=frame_height,
        source_sha256=source_sha256,
    )
