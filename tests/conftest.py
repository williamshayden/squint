from __future__ import annotations

import json
from collections.abc import Callable
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
from squint_rl.episode import Episode, seal_episode
from squint_rl.synthetic import make_synthetic_episode
from squint_rl.tracker import DetectionBatch, TrackBatch, TrackerSummary


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


class RecordingTracker:
    """A real tracker-shaped test double that preserves measurement identity."""

    def __init__(self) -> None:
        self.last = TrackBatch.empty()
        self.measurements: list[DetectionBatch | None] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.last = TrackBatch.empty()
        self.measurements.clear()
        self.reset_calls += 1

    def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
        del timestamp_s
        self.measurements.append(detections)
        if detections is not None:
            self.last = TrackBatch(
                detections.boxes_xyxy,
                np.arange(1, len(detections) + 1, dtype=np.int64),
                detections.class_ids,
                detections.scores,
            )
        return self.last

    def summary(self) -> TrackerSummary:
        count = len(self.last)
        return TrackerSummary(
            active_tracks=count,
            confirmed_tracks=count,
            stale_tracks=0,
            mean_age_s=0.0,
            mean_motion_px_s=0.0,
            mean_confidence=float(np.mean(self.last.scores)) if count else 0.0,
        )


@pytest.fixture
def sealed_episode(tmp_path: Path) -> Episode:
    return Episode.open(
        make_synthetic_episode(
            tmp_path / "episode",
            frame_count=4,
            fps=2.0,
            change_frames=(0,),
            latency_ms=10.0,
        )
    )


def reseal_variant(
    base: Episode,
    destination: Path,
    mutate: Callable[[dict[str, np.ndarray]], None],
) -> Episode:
    """Copy a sealed replay, mutate its arrays, and seal a distinct valid replay."""
    arrays = {name: np.array(value, copy=True) for name, value in base.arrays.items()}
    mutate(arrays)
    manifest = json.loads((base.path / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = {}
    return Episode.open(seal_episode(destination, manifest=manifest, arrays=arrays))
