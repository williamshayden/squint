from __future__ import annotations

import numpy as np
import pytest

from squint_rl.reward import RewardState, box_iou
from squint_rl.tracker import GroundTruthBatch, TrackBatch


def ground_truth(
    boxes: list[list[float]],
    *,
    track_ids: list[int] | None = None,
    class_ids: list[int] | None = None,
    valid: list[bool] | None = None,
    ignore: list[bool] | None = None,
) -> GroundTruthBatch:
    count = len(boxes)
    ignored = np.array(ignore if ignore is not None else [False] * count, dtype=np.bool_)
    return GroundTruthBatch(
        boxes_xyxy=np.array(boxes, dtype=np.float32).reshape((-1, 4)),
        track_ids=np.array(track_ids if track_ids is not None else range(7, 7 + count)),
        class_ids=np.array(class_ids if class_ids is not None else [1] * count),
        visibility=np.ones(count, dtype=np.float32),
        valid=np.array(valid if valid is not None else ~ignored, dtype=np.bool_),
        ignore=ignored,
    )


def tracks(
    boxes: list[list[float]],
    *,
    track_ids: list[int] | None = None,
    class_ids: list[int] | None = None,
) -> TrackBatch:
    count = len(boxes)
    return TrackBatch(
        boxes_xyxy=np.array(boxes, dtype=np.float32).reshape((-1, 4)),
        track_ids=np.array(track_ids if track_ids is not None else range(3, 3 + count)),
        class_ids=np.array(class_ids if class_ids is not None else [1] * count),
        scores=np.full(count, 0.9, dtype=np.float32),
    )


def test_box_iou_handles_empty_inputs_and_pairwise_matrix() -> None:
    empty = np.empty((0, 4), dtype=np.float32)
    boxes = np.array([[0, 0, 2, 2], [0, 0, 1, 1]], dtype=np.float32)
    assert box_iou(empty, boxes).shape == (0, 2)
    assert box_iou(boxes, empty).shape == (2, 0)
    np.testing.assert_allclose(
        box_iou(boxes, np.array([[1, 1, 3, 3], [0, 0, 1, 1]], dtype=np.float32)),
        np.array([[1 / 7, 1 / 4], [0, 1]], dtype=np.float32),
    )


def test_perfect_match_has_reward_one() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]])
    )
    assert reward == 1.0
    assert counts.matches == 1
    assert counts.false_positives == 0
    assert counts.false_negatives == 0
    assert counts.identity_switches == 0
    assert counts.localization_error == 0.0


def test_identity_switch_is_charged_on_second_match() -> None:
    state = RewardState()
    state.score(ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]], track_ids=[3]))
    reward, counts = state.score(
        ground_truth([[1, 0, 11, 10]]), tracks([[1, 0, 11, 10]], track_ids=[4])
    )
    assert counts.identity_switches == 1
    assert reward == 0.0


def test_ignored_overlap_suppresses_an_otherwise_false_positive() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]], ignore=[True]), tracks([[0, 0, 10, 10]])
    )
    assert reward == 1.0
    assert counts.false_positives == 0
    assert counts.false_negatives == 0


def test_unmatched_valid_ground_truth_is_a_false_negative() -> None:
    reward, counts = RewardState().score(ground_truth([[0, 0, 10, 10]]), tracks([]))
    assert reward == 0.0
    assert counts.false_negatives == 1
    assert counts.false_positives == 0


def test_unmatched_track_is_a_false_positive_when_ground_truth_is_empty() -> None:
    reward, counts = RewardState().score(ground_truth([]), tracks([[0, 0, 10, 10]]))
    assert reward == 0.0
    assert counts.false_negatives == 0
    assert counts.false_positives == 1


def test_empty_ground_truth_and_tracks_have_no_error() -> None:
    reward, counts = RewardState().score(ground_truth([]), tracks([]))
    assert reward == 1.0
    assert counts.matches == 0
    assert counts.false_positives == 0
    assert counts.false_negatives == 0


def test_iou_below_the_inclusive_gate_creates_a_false_negative_and_positive() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]]), tracks([[5, 0, 15, 10]])
    )
    assert reward == -1.0
    assert counts.matches == 0
    assert counts.false_negatives == 1
    assert counts.false_positives == 1


def test_iou_exactly_at_the_inclusive_gate_is_a_match() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 6, 6]]), tracks([[2, 0, 8, 6]])
    )
    assert counts.matches == 1
    assert counts.false_negatives == 0
    assert counts.false_positives == 0
    assert counts.localization_error == pytest.approx(0.5)
    assert reward == pytest.approx(0.5)


def test_class_mismatch_cannot_match_even_at_iou_one() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]], class_ids=[1]),
        tracks([[0, 0, 10, 10]], class_ids=[2]),
    )
    assert reward == -1.0
    assert counts.matches == 0
    assert counts.false_negatives == 1
    assert counts.false_positives == 1


def test_localization_error_is_one_minus_iou() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]]), tracks([[2.5, 0, 12.5, 10]])
    )
    assert counts.matches == 1
    assert counts.localization_error == pytest.approx(0.4)
    assert reward == pytest.approx(0.6)


def test_reward_is_clipped_at_negative_one() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]]),
        tracks([[20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]]),
    )
    assert reward == -1.0
    assert counts.false_negatives == 1
    assert counts.false_positives == 3


def test_reset_clears_identity_history() -> None:
    state = RewardState()
    state.score(ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]], track_ids=[3]))
    state.reset()
    reward, counts = state.score(
        ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]], track_ids=[4])
    )
    assert counts.identity_switches == 0
    assert reward == 1.0


def test_unmatched_ground_truth_preserves_identity_history() -> None:
    state = RewardState()
    state.score(ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]], track_ids=[3]))
    state.score(ground_truth([[0, 0, 10, 10]]), tracks([]))
    reward, counts = state.score(
        ground_truth([[0, 0, 10, 10]]), tracks([[0, 0, 10, 10]], track_ids=[4])
    )
    assert counts.identity_switches == 1
    assert reward == 0.0


def test_gated_hungarian_prefers_one_valid_edge_over_two_invalid_edges() -> None:
    reward, counts = RewardState().score(
        ground_truth(
            [[7.5, 0, 19.2, 10], [3.3, 0, 9.5, 10]], track_ids=[7, 8]
        ),
        tracks([[3.1, 0, 22.8, 10], [10.3, 0, 27.9, 10]], track_ids=[3, 4]),
    )
    assert counts.matches == 1
    assert counts.false_negatives == 1
    assert counts.false_positives == 1
    assert counts.localization_error == pytest.approx(1 - 0.5939086)
    assert reward == pytest.approx(-0.2030457)


def test_invalid_non_ignored_ground_truth_does_not_match_or_count_as_false_negative() -> None:
    reward, counts = RewardState().score(
        ground_truth([[0, 0, 10, 10]], valid=[False], ignore=[False]),
        tracks([[0, 0, 10, 10]]),
    )
    assert counts.matches == 0
    assert counts.false_negatives == 0
    assert counts.false_positives == 1
    assert reward == 0.0


def test_identity_history_is_keyed_by_ground_truth_id_across_absence() -> None:
    state = RewardState()
    state.score(
        ground_truth(
            [[0, 0, 10, 10], [20, 0, 30, 10]], track_ids=[7, 8]
        ),
        tracks([[0, 0, 10, 10], [20, 0, 30, 10]], track_ids=[3, 4]),
    )
    state.score(
        ground_truth([[20, 0, 30, 10]], track_ids=[8]),
        tracks([[20, 0, 30, 10]], track_ids=[4]),
    )
    reward, counts = state.score(
        ground_truth([[0, 0, 10, 10]], track_ids=[7]),
        tracks([[0, 0, 10, 10]], track_ids=[3]),
    )
    assert counts.identity_switches == 0
    assert reward == 1.0
