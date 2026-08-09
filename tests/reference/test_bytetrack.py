from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv

from squint_rl.reference.bytetrack import ByteTrackAdapter, tracker_factory
from squint_rl.tracker import DetectionBatch


def _detections(
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> DetectionBatch:
    return DetectionBatch(
        np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        np.asarray(scores, dtype=np.float32),
        np.asarray(class_ids, dtype=np.int64),
    )


def _tracked(
    boxes: list[list[float]],
    tracker_ids: list[int],
    *,
    scores: list[float] | None = None,
    class_ids: list[int] | None = None,
) -> sv.Detections:
    result = sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        confidence=None if scores is None else np.asarray(scores, dtype=np.float32),
        class_id=None if class_ids is None else np.asarray(class_ids, dtype=np.int64),
    )
    result.tracker_id = np.asarray(tracker_ids, dtype=np.int64)
    return result


class FakeBackend:
    """Boundary double with only the three public adapter dependencies."""

    def __init__(self, responses: list[tuple[sv.Detections, sv.Detections]]) -> None:
        self._responses = iter(responses)
        self.tracked_objects = sv.Detections.empty()
        self.calls: list[tuple[sv.Detections, float | None]] = []
        self.reset_calls = 0

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        del frame
        self.calls.append((detections, timestamp))
        returned, tracked = next(self._responses)
        self.tracked_objects = tracked
        return returned

    def reset(self) -> None:
        self.reset_calls += 1
        self.tracked_objects = sv.Detections.empty()


def test_skip_emits_public_prediction_not_empty_measurement_result() -> None:
    """Would fail if step converted the empty update return instead of tracked_objects."""
    backend = FakeBackend(
        [(sv.Detections.empty(), _tracked([[2, 0, 12, 10]], [4]))]
    )
    adapter = ByteTrackAdapter(frame_rate=30.0, backend=backend)

    predicted = adapter.step(None, timestamp_s=1.0 / 30.0)

    assert predicted.track_ids.tolist() == [4]
    assert predicted.boxes_xyxy.tolist() == [[2.0, 0.0, 12.0, 10.0]]
    assert len(backend.calls[0][0]) == 0
    assert backend.calls[0][1] == pytest.approx(1.0 / 30.0)


def test_measurement_conversion_and_metadata_follow_confirmed_ids() -> None:
    """Would fail if arrays are not converted or class/confidence are not carried forward."""
    measured = _tracked([[0, 0, 10, 10]], [5], scores=[0.75], class_ids=[3])
    backend = FakeBackend(
        [
            (measured, _tracked([[0, 0, 10, 10]], [5])),
            (sv.Detections.empty(), _tracked([[1, 0, 11, 10]], [5])),
        ]
    )
    adapter = ByteTrackAdapter(frame_rate=10.0, backend=backend)

    first = adapter.step(_detections([[0, 0, 10, 10]], [0.75], [3]), timestamp_s=0.0)
    predicted = adapter.step(None, timestamp_s=0.5)

    measurement, timestamp = backend.calls[0]
    assert measurement.xyxy.tolist() == [[0.0, 0.0, 10.0, 10.0]]
    assert measurement.confidence.tolist() == pytest.approx([0.75])
    assert measurement.class_id.tolist() == [3]
    assert timestamp == 0.0
    assert first.class_ids.tolist() == [3]
    assert first.scores.tolist() == pytest.approx([0.75])
    assert predicted.track_ids.tolist() == [5]
    assert predicted.class_ids.tolist() == [3]
    assert predicted.scores.tolist() == pytest.approx([0.75])


def test_output_is_sorted_filters_unconfirmed_and_reports_adapter_owned_summary() -> None:
    """Would fail if output uses backend order/private lifecycle fields rather than adapter state."""
    backend = FakeBackend(
        [
            (
                _tracked(
                    [[20, 0, 30, 10], [0, 0, 10, 10]],
                    [9, 2],
                    scores=[0.6, 0.8],
                    class_ids=[4, 8],
                ),
                _tracked(
                    [[20, 0, 30, 10], [0, 0, 10, 10], [40, 0, 50, 10]],
                    [9, 2, -1],
                ),
            ),
            (
                _tracked([[2, 0, 12, 10]], [2], scores=[0.9], class_ids=[8]),
                _tracked([[23, 0, 33, 10], [2, 0, 12, 10], [40, 0, 50, 10]], [9, 2, -1]),
            ),
        ]
    )
    adapter = ByteTrackAdapter(frame_rate=10.0, backend=backend)

    adapter.step(
        _detections([[20, 0, 30, 10], [0, 0, 10, 10]], [0.6, 0.8], [4, 8]),
        timestamp_s=0.0,
    )
    tracks = adapter.step(_detections([[2, 0, 12, 10]], [0.9], [8]), timestamp_s=0.5)
    summary = adapter.summary()

    assert tracks.track_ids.tolist() == [2, 9]
    assert tracks.class_ids.tolist() == [8, 4]
    assert tracks.scores.tolist() == pytest.approx([0.9, 0.6])
    assert summary.active_tracks == 2
    assert summary.confirmed_tracks == 2
    assert summary.stale_tracks == 1
    assert summary.mean_age_s == pytest.approx(0.5)
    assert summary.mean_motion_px_s == pytest.approx(5.0)
    assert summary.mean_confidence == pytest.approx(0.75)


def test_none_and_empty_measurements_share_empty_backend_boundary() -> None:
    """Would fail if None leaks a distinct backend representation."""
    backend = FakeBackend(
        [
            (sv.Detections.empty(), sv.Detections.empty()),
            (sv.Detections.empty(), sv.Detections.empty()),
        ]
    )
    adapter = ByteTrackAdapter(frame_rate=30.0, backend=backend)

    adapter.step(None, timestamp_s=0.0)
    adapter.step(DetectionBatch.empty(), timestamp_s=1.0 / 30.0)

    assert [len(measurement) for measurement, _ in backend.calls] == [0, 0]
    assert [measurement.tracker_id for measurement, _ in backend.calls] == [None, None]


def test_reset_clears_adapter_history_and_resets_backend() -> None:
    """Would fail if a new episode inherits prior class/confidence or lifecycle state."""
    backend = FakeBackend(
        [
            (
                _tracked([[0, 0, 10, 10]], [1], scores=[0.9], class_ids=[7]),
                _tracked([[0, 0, 10, 10]], [1]),
            ),
            (sv.Detections.empty(), _tracked([[2, 0, 12, 10]], [1])),
        ]
    )
    adapter = ByteTrackAdapter(frame_rate=10.0, backend=backend)

    adapter.step(_detections([[0, 0, 10, 10]], [0.9], [7]), timestamp_s=0.0)
    adapter.reset()
    tracks = adapter.step(None, timestamp_s=1.0)

    assert backend.reset_calls == 1
    assert tracks.class_ids.tolist() == [1]
    assert tracks.scores.tolist() == [0.0]
    assert adapter.summary().mean_age_s == 0.0
    assert adapter.summary().stale_tracks == 0


def test_factory_uses_episode_fps_and_forwards_parameters() -> None:
    """Would fail if factory ignores replay fps or custom tracker parameters."""
    episode = SimpleNamespace(fps=17.5)
    tracker = tracker_factory(
        episode=episode,
        minimum_consecutive_frames=1,
        lost_track_buffer=4,
    )

    assert isinstance(tracker, ByteTrackAdapter)
    assert tracker._frame_rate == 17.5
    assert tracker._backend.minimum_consecutive_frames == 1
    assert tracker._backend.maximum_time_without_update == pytest.approx(4.0 / 30.0)


def test_pinned_reference_stack_predicts_after_empty_measurement() -> None:
    """Would fail if the installed 2.6.0/0.26.1 boundary stops retaining confirmed predictions."""
    assert version("trackers") == "2.6.0"
    assert version("supervision") == "0.26.1"
    adapter = ByteTrackAdapter(
        frame_rate=30.0,
        minimum_consecutive_frames=2,
        track_activation_threshold=0.25,
    )
    measurement = _detections([[10, 10, 30, 40]], [0.9], [1])

    adapter.step(measurement, timestamp_s=0.0)
    adapter.step(measurement, timestamp_s=1.0 / 30.0)
    predicted = adapter.step(None, timestamp_s=2.0 / 30.0)

    assert predicted.track_ids.tolist() == [0]
    assert np.all(np.isfinite(predicted.boxes_xyxy))
    assert predicted.class_ids.tolist() == [1]
    assert predicted.scores.tolist() == pytest.approx([0.9])
