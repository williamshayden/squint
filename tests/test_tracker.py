import importlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from squint_rl.tracker import (
    DetectionBatch,
    GroundTruthBatch,
    ObservationScales,
    PolicyContext,
    TrackBatch,
    Tracker,
    TrackerSummary,
)


def _assert_irreversibly_readonly(array: np.ndarray) -> None:
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)


def test_tracker_module_is_available() -> None:
    try:
        module = importlib.import_module("squint_rl.tracker")
    except ModuleNotFoundError:
        module = None

    assert module is not None


def test_detection_batch_validates_shapes_and_freezes_arrays() -> None:
    boxes = np.array([[1, 2, 5, 8]], dtype=np.float32)
    scores = np.array([0.7], dtype=np.float32)
    class_ids = np.array([1], dtype=np.int64)
    batch = DetectionBatch(boxes, scores, class_ids)

    assert len(batch) == 1
    assert not batch.boxes_xyxy.flags.writeable
    assert not batch.scores.flags.writeable
    assert not batch.class_ids.flags.writeable
    boxes[0, 0] = 99
    scores[0] = 0.2
    class_ids[0] = 9
    assert batch.boxes_xyxy[0, 0] == 1
    assert batch.scores[0] == 0.7
    assert batch.class_ids[0] == 1
    for value in (batch.boxes_xyxy, batch.scores, batch.class_ids):
        _assert_irreversibly_readonly(value)
    with pytest.raises(ValueError, match="boxes_xyxy"):
        DetectionBatch(np.zeros((1, 5), np.float32), np.ones(1), np.ones(1, np.int64))
    with pytest.raises(FrozenInstanceError):
        batch.scores = scores


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        (np.array([[1, 2, 1, 8]], np.float32), "boxes_xyxy"),
        (np.array([[1, 2, 5, 2]], np.float32), "boxes_xyxy"),
        (np.array([[np.nan, 2, 5, 8]], np.float32), "finite"),
    ],
)
def test_detection_batch_rejects_invalid_boxes(boxes: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionBatch(boxes, np.ones(1), np.ones(1, np.int64))


@pytest.mark.parametrize(
    "scores",
    [np.array([np.nan]), np.array([-0.1]), np.array([1.1])],
)
def test_detection_batch_rejects_invalid_scores(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scores"):
        DetectionBatch(np.array([[1, 2, 5, 8]], np.float32), scores, np.ones(1, np.int64))


def test_ground_truth_batch_validates_and_freezes_all_vectors() -> None:
    boxes = np.array([[1, 2, 5, 8]], np.float32)
    track_ids = np.array([4], np.int64)
    class_ids = np.array([1], np.int64)
    visibility = np.array([0.5], np.float32)
    valid = np.array([True])
    ignore = np.array([False])
    batch = GroundTruthBatch(boxes, track_ids, class_ids, visibility, valid, ignore)

    assert len(batch) == 1
    boxes[0, 0] = 99
    track_ids[0] = 9
    class_ids[0] = 9
    visibility[0] = 0.2
    valid[0] = False
    ignore[0] = True
    np.testing.assert_array_equal(batch.boxes_xyxy, [[1, 2, 5, 8]])
    np.testing.assert_array_equal(batch.track_ids, [4])
    np.testing.assert_array_equal(batch.class_ids, [1])
    np.testing.assert_array_equal(batch.visibility, [0.5])
    np.testing.assert_array_equal(batch.valid, [True])
    np.testing.assert_array_equal(batch.ignore, [False])
    for value in (
        batch.boxes_xyxy,
        batch.track_ids,
        batch.class_ids,
        batch.visibility,
        batch.valid,
        batch.ignore,
    ):
        _assert_irreversibly_readonly(value)

    with pytest.raises(ValueError, match="both valid and ignored"):
        GroundTruthBatch(
            np.array([[1, 2, 5, 8]], np.float32),
            np.array([4]),
            np.array([1]),
            np.array([0.5]),
            np.array([True]),
            np.array([True]),
        )
    with pytest.raises(ValueError, match="unique"):
        GroundTruthBatch(
            np.array([[1, 2, 5, 8], [2, 3, 6, 9]], np.float32),
            np.array([4, 4]),
            np.array([1, 1]),
            np.array([0.5, 0.5]),
            np.array([True, True]),
            np.array([False, False]),
        )
    with pytest.raises(ValueError, match="visibility"):
        GroundTruthBatch(
            np.array([[1, 2, 5, 8]], np.float32),
            np.array([4]),
            np.array([1]),
            np.array([1.1]),
            np.array([True]),
            np.array([False]),
        )


def test_track_batch_rejects_invalid_ids_scores_and_freezes_arrays() -> None:
    boxes = np.array([[1, 2, 5, 8]], np.float32)
    track_ids = np.array([4], np.int64)
    class_ids = np.array([1], np.int64)
    scores = np.array([0.8], np.float32)
    batch = TrackBatch(boxes, track_ids, class_ids, scores)
    assert len(batch) == 1
    boxes[0, 0] = 99
    track_ids[0] = 9
    class_ids[0] = 9
    scores[0] = 0.2
    np.testing.assert_array_equal(batch.boxes_xyxy, [[1, 2, 5, 8]])
    np.testing.assert_array_equal(batch.track_ids, [4])
    np.testing.assert_array_equal(batch.class_ids, [1])
    np.testing.assert_array_equal(batch.scores, np.array([0.8], np.float32))
    for value in (batch.boxes_xyxy, batch.track_ids, batch.class_ids, batch.scores):
        _assert_irreversibly_readonly(value)

    with pytest.raises(ValueError, match="unique and nonnegative"):
        TrackBatch(
            np.array([[1, 2, 5, 8], [2, 3, 6, 9]], np.float32),
            np.array([4, 4]),
            np.array([1, 1]),
            np.array([0.8, 0.8]),
        )
    with pytest.raises(ValueError, match="unique and nonnegative"):
        TrackBatch(
            np.array([[1, 2, 5, 8]], np.float32),
            np.array([-1]),
            np.array([1]),
            np.array([0.8]),
        )
    with pytest.raises(ValueError, match="track scores"):
        TrackBatch(
            np.array([[1, 2, 5, 8]], np.float32),
            np.array([4]),
            np.array([1]),
            np.array([1.1]),
        )


def test_empty_batches_have_zero_length_and_frozen_arrays() -> None:
    for batch in (DetectionBatch.empty(), GroundTruthBatch.empty(), TrackBatch.empty()):
        assert len(batch) == 0
        for field in batch.__dataclass_fields__:
            _assert_irreversibly_readonly(getattr(batch, field))


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (TrackerSummary, (-1, 0, 0, 0.0, 0.0, 0.0)),
        (TrackerSummary, (1, 2, 0, 0.0, 0.0, 0.0)),
        (TrackerSummary, (1, 0, 2, 0.0, 0.0, 0.0)),
        (TrackerSummary, (0, 0, 0, -1.0, 0.0, 0.0)),
        (TrackerSummary, (0, 0, 0, 0.0, np.nan, 0.0)),
        (TrackerSummary, (0, 0, 0, 0.0, 0.0, 1.1)),
        (ObservationScales, (1.0, 0.0, 1.0, 1.0)),
        (ObservationScales, (1.0, np.inf, 1.0, 1.0)),
    ],
)
def test_summary_and_scales_reject_invalid_values(factory: type[object], args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        factory(*args)  # type: ignore[call-arg]


def test_summary_scales_and_policy_context_are_frozen_value_objects() -> None:
    summary = TrackerSummary.empty()
    scales = ObservationScales(1.0, 2.0, 3.0, 4.0)
    context = PolicyContext(30.0, 29.97, 5.0, 7)

    with pytest.raises(FrozenInstanceError):
        summary.active_tracks = 1
    with pytest.raises(FrozenInstanceError):
        scales.age_s = 1.0
    with pytest.raises(FrozenInstanceError):
        context.seed = 8


def test_tracker_protocol_preserves_skip_vs_empty_measurement() -> None:
    class RecordingTracker:
        def __init__(self) -> None:
            self.seen: list[DetectionBatch | None] = []

        def reset(self) -> None:
            self.seen.clear()

        def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
            del timestamp_s
            self.seen.append(detections)
            return TrackBatch.empty()

        def summary(self) -> TrackerSummary:
            return TrackerSummary.empty()

    tracker = RecordingTracker()
    assert isinstance(tracker, Tracker)
    tracker.step(None, 0.0)
    tracker.step(DetectionBatch.empty(), 1.0)
    assert tracker.seen[0] is None
    assert tracker.seen[1] is not None and len(tracker.seen[1]) == 0
