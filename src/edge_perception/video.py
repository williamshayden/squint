"""Chronological decoding of local video files."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

import av
import numpy as np


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One decoded RGB frame and its source-container timestamp."""

    frame_index: int
    source_time_ms: float | None
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Ground-truth properties probed from one finalized video file."""

    width: int
    height: int
    average_fps: float
    container: str
    codec: str
    duration_seconds: float
    has_audio: bool
    file_size_bytes: int


def probe_video(path: Path) -> VideoMetadata:
    """Probe one local video's primary stream and close the container on every path."""

    source_path = Path(path)
    container = av.open(str(source_path))
    try:
        video_stream = next(
            (candidate for candidate in container.streams if candidate.type == "video"),
            None,
        )
        if video_stream is None:
            raise ValueError(f"container has no video stream: {source_path}")

        stream_details = cast(Any, video_stream)
        codec_context = stream_details.codec_context
        width = int(stream_details.width or codec_context.width)
        height = int(stream_details.height or codec_context.height)
        average_rate = video_stream.average_rate
        if average_rate is None:
            raise ValueError("video stream has no average FPS")
        average_fps = float(average_rate)
        if not isfinite(average_fps) or average_fps <= 0.0:
            raise ValueError("video stream has invalid average FPS")

        duration_seconds = _video_duration_seconds(container, video_stream)
        container_name = str(container.format.name)
        codec_name = str(codec_context.name)
        has_audio = any(stream.type == "audio" for stream in container.streams)
        return VideoMetadata(
            width=width,
            height=height,
            average_fps=average_fps,
            container=container_name,
            codec=codec_name,
            duration_seconds=duration_seconds,
            has_audio=has_audio,
            file_size_bytes=source_path.stat().st_size,
        )
    finally:
        container.close()


def _video_duration_seconds(container: Any, video_stream: Any) -> float:
    if video_stream.duration is not None and video_stream.time_base is not None:
        duration_seconds = float(video_stream.duration * video_stream.time_base)
    elif container.duration is not None:
        duration_seconds = float(container.duration / av.time_base)
    else:
        duration_seconds = 0.0
    if not isfinite(duration_seconds) or duration_seconds < 0.0:
        raise ValueError("video stream has invalid duration")
    return duration_seconds


def iter_video(path: Path) -> Iterator[DecodedFrame]:
    """Decode the first video stream sequentially in container order."""

    with av.open(str(path)) as container:
        stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
        if stream is None:
            raise ValueError(f"container has no video stream: {path}")

        for frame_index, decoded in enumerate(container.decode(stream)):
            frame = cast(av.VideoFrame, decoded)
            source_time_ms = None
            if frame.pts is not None and frame.time_base is not None:
                source_time_ms = float(frame.pts * frame.time_base * 1_000)
            yield DecodedFrame(
                frame_index=frame_index,
                source_time_ms=source_time_ms,
                image=frame.to_ndarray(format="rgb24"),
            )


def first_video_frame(path: Path) -> DecodedFrame:
    """Decode one frame and promptly close the underlying iterator."""

    frames = iter_video(path)
    try:
        try:
            return next(frames)
        except StopIteration as error:
            raise ValueError(f"video contains no decoded frames: {path}") from error
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
