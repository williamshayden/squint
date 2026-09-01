from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from squint_rl.tracker import FloatArray, GroundTruthBatch, TrackBatch

IOU_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class MatchCounts:
    matches: int
    false_positives: int
    false_negatives: int
    identity_switches: int
    localization_error: float


def box_iou(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return pairwise IoU for boxes in ``[x1, y1, x2, y2]`` format."""
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)

    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0.0, None), axis=2)
    left_area = np.prod(left[:, 2:] - left[:, :2], axis=1)
    right_area = np.prod(right[:, 2:] - right[:, :2], axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return cast(
        FloatArray,
        np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        ),
    )


def _gated_assignment(
    similarities: FloatArray, eligible: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if similarities.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    max_matches = min(similarities.shape)
    costs = np.zeros(similarities.shape, dtype=np.float64)
    costs[eligible] = -(max_matches + 1 + similarities[eligible])
    rows, cols = linear_sum_assignment(costs)
    accepted = eligible[rows, cols]
    return rows[accepted], cols[accepted]


@dataclass(slots=True)
class RewardState:
    _previous_assignment: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._previous_assignment.clear()

    def score(
        self, ground_truth: GroundTruthBatch, tracks: TrackBatch
    ) -> tuple[float, MatchCounts]:
        valid_mask = ground_truth.valid & ~ground_truth.ignore
        valid_boxes = ground_truth.boxes_xyxy[valid_mask]
        valid_ids = ground_truth.track_ids[valid_mask]
        valid_classes = ground_truth.class_ids[valid_mask]

        similarities = box_iou(valid_boxes, tracks.boxes_xyxy)
        eligible = (similarities >= IOU_THRESHOLD) & (
            valid_classes[:, None] == tracks.class_ids[None, :]
        )
        rows, cols = _gated_assignment(similarities, eligible)

        matched_tracks = np.zeros(len(tracks), dtype=np.bool_)
        matched_tracks[cols] = True
        unmatched_tracks = np.flatnonzero(~matched_tracks)
        ignored_boxes = ground_truth.boxes_xyxy[ground_truth.ignore]
        ignored_overlap = box_iou(ignored_boxes, tracks.boxes_xyxy[unmatched_tracks])
        suppressed = np.any(ignored_overlap >= IOU_THRESHOLD, axis=0)

        identity_switches = 0
        for row, col in zip(rows, cols, strict=True):
            ground_truth_id = int(valid_ids[row])
            track_id = int(tracks.track_ids[col])
            previous_track_id = self._previous_assignment.get(ground_truth_id)
            if previous_track_id is not None and previous_track_id != track_id:
                identity_switches += 1
            self._previous_assignment[ground_truth_id] = track_id

        counts = MatchCounts(
            matches=len(rows),
            false_positives=len(unmatched_tracks) - int(np.count_nonzero(suppressed)),
            false_negatives=len(valid_boxes) - len(rows),
            identity_switches=identity_switches,
            localization_error=float(np.sum(1.0 - similarities[rows, cols])),
        )
        total_error = (
            counts.false_negatives
            + counts.false_positives
            + counts.identity_switches
            + counts.localization_error
        )
        reward = float(
            np.clip(1.0 - total_error / max(1, len(valid_boxes)), -1.0, 1.0)
        )
        return reward, counts
