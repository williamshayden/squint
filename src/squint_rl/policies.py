from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from squint_rl.env import RUN_DETECTOR, SKIP
from squint_rl.tracker import Observation, PolicyContext


class Policy(Protocol):
    def __call__(self, observation: Observation) -> int:
        pass


def reset_policy(policy: Policy, *, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset(seed=seed)


def _affordable(observation: Observation) -> bool:
    return bool(observation["compute_budget"][2] >= 0.5)


@dataclass(slots=True)
class GreedyAffordable:
    def __call__(self, observation: Observation) -> int:
        return RUN_DETECTOR if _affordable(observation) else SKIP


@dataclass(slots=True)
class Periodic:
    interval_s: float
    elapsed_scale_s: float

    def __call__(self, observation: Observation) -> int:
        elapsed_s = float(observation["compute_budget"][3]) * self.elapsed_scale_s
        return RUN_DETECTOR if _affordable(observation) and elapsed_s >= self.interval_s else SKIP


@dataclass(slots=True)
class SceneChange:
    threshold: float

    def __call__(self, observation: Observation) -> int:
        changed = float(np.mean(observation["scene_change"])) >= self.threshold
        return RUN_DETECTOR if _affordable(observation) and changed else SKIP


@dataclass(slots=True)
class TrackStaleness:
    threshold: float

    def __call__(self, observation: Observation) -> int:
        stale = float(observation["tracker_state"][2]) >= self.threshold
        return RUN_DETECTOR if _affordable(observation) and stale else SKIP


def greedy_factory(*, context: PolicyContext) -> Policy:
    del context
    return GreedyAffordable()


def periodic_factory(*, context: PolicyContext) -> Policy:
    return Periodic(
        interval_s=1.0 / (context.nominal_rate * context.source_fps),
        elapsed_scale_s=context.time_since_detector_scale_s,
    )


def scene_change_factory(*, context: PolicyContext, threshold: float) -> Policy:
    del context
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("scene-change threshold must be in [0, 1]")
    return SceneChange(threshold)


def staleness_factory(*, context: PolicyContext, threshold: float) -> Policy:
    del context
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("staleness threshold must be in [0, 1]")
    return TrackStaleness(threshold)
