"""Chronological decoding of local video files."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import av
import numpy as np


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One decoded RGB frame and its source-container timestamp."""

    frame_index: int
    source_time_ms: float | None
    image: np.ndarray


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
