from __future__ import annotations

import numpy as np

from squint_rl.env import RUN_DETECTOR, SKIP
from squint_rl.tracker import DetectionBatch, TrackBatch, TrackerSummary

stateful_tracker_instances: list[object] = []
stateful_policy_instances: list[object] = []
stateful_policy_seeds: list[int] = []


def reset_stateful_records() -> None:
    stateful_tracker_instances.clear()
    stateful_policy_instances.clear()
    stateful_policy_seeds.clear()


class EchoTracker:
    """Small tracker-shaped double for replay benchmark contracts."""

    def __init__(self) -> None:
        self._tracks = TrackBatch.empty()

    def reset(self) -> None:
        self._tracks = TrackBatch.empty()

    def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
        del timestamp_s
        if detections is not None:
            self._tracks = TrackBatch(
                detections.boxes_xyxy,
                np.arange(len(detections), dtype=np.int64),
                detections.class_ids,
                detections.scores,
            )
        return self._tracks

    def summary(self) -> TrackerSummary:
        count = len(self._tracks)
        return TrackerSummary(
            active_tracks=count,
            confirmed_tracks=count,
            stale_tracks=0,
            mean_age_s=0.0,
            mean_motion_px_s=0.0,
            mean_confidence=float(np.mean(self._tracks.scores)) if count else 0.0,
        )


def tracker_factory(*, episode: object) -> EchoTracker:
    del episode
    return EchoTracker()


def raising_tracker_factory(*, episode: object) -> EchoTracker:
    del episode
    raise RuntimeError("tracker construction failed")


class StatefulTracker(EchoTracker):
    def __init__(self) -> None:
        super().__init__()
        stateful_tracker_instances.append(self)


def stateful_tracker_factory(*, episode: object) -> StatefulTracker:
    del episode
    return StatefulTracker()


def greedy_factory(*, context: object) -> object:
    del context

    def choose(_observation: object) -> int:
        return RUN_DETECTOR

    return choose


def never_detect_factory(*, context: object) -> object:
    del context

    def choose(_observation: object) -> int:
        return SKIP

    return choose


def raising_policy_factory(*, context: object) -> object:
    del context

    def choose(_observation: object) -> int:
        raise RuntimeError("policy execution failed")

    return choose


class StatefulPolicy:
    def __init__(self) -> None:
        stateful_policy_instances.append(self)

    def reset(self, *, seed: int) -> None:
        stateful_policy_seeds.append(seed)

    def __call__(self, _observation: object) -> int:
        return SKIP


def stateful_policy_factory(*, context: object) -> StatefulPolicy:
    del context
    return StatefulPolicy()
