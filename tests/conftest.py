from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from edge_perception.contracts import (
    BatchPrediction,
    Box,
    Detection,
    DetectorIdentity,
    StageTiming,
)


class FakeDetector:
    def __init__(self) -> None:
        self._identity = DetectorIdentity(
            adapter="fake-detector",
            model_id="tests/fake-detector",
            revision="fake-revision-1",
            weights_sha256="a" * 64,
            backend="numpy",
            backend_version=np.__version__,
            device="cpu",
            dtype="uint8",
        )
        self.warmup_calls: list[tuple[tuple[int, ...], int]] = []
        self.predict_batch_sizes: list[int] = []
        self.predict_shapes: list[tuple[int, ...]] = []

    @property
    def identity(self) -> DetectorIdentity:
        return self._identity

    def warmup(self, image: np.ndarray, runs: int) -> None:
        self.warmup_calls.append((image.shape, runs))

    def predict(self, images: tuple[np.ndarray, ...]) -> BatchPrediction:
        self.predict_batch_sizes.append(len(images))
        self.predict_shapes.extend(image.shape for image in images)
        detections = tuple(
            (Detection(Box(10.0, 5.0, 20.0, 15.0), 1, 0.75, "object"),)
            for _image in images
        )
        return BatchPrediction(
            detections=detections,
            timing=StageTiming(0.1, 0.2, 0.3, 0.6),
        )

    def peak_device_memory_bytes(self) -> int | None:
        return None


def write_test_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 200
        stream.height = 100
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)

        for frame_index, red_value in enumerate((20, 120, 220)):
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[..., 0] = red_value
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "three-frames.mp4"
    write_test_video(path)
    return path


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
