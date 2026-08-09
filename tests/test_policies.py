from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from squint_rl.env import RUN_DETECTOR, SKIP
from squint_rl.policies import (
    greedy_factory,
    periodic_factory,
    reset_policy,
    scene_change_factory,
    staleness_factory,
)
from squint_rl.tracker import Observation, PolicyContext

CONTEXT = PolicyContext(
    nominal_rate=0.25,
    source_fps=10.0,
    reserve_ms=10.0,
    seed=3,
    time_since_detector_scale_s=5.0,
)


def observation(
    *,
    affordable: float = 1.0,
    elapsed: float = 0.0,
    scene: float | np.ndarray = 0.0,
    stale: float = 0.0,
) -> Observation:
    thumbnail = (
        np.full((3, 3), scene, np.float32)
        if np.isscalar(scene)
        else np.asarray(scene, dtype=np.float32)
    )
    return {
        "scene_change": thumbnail,
        "tracker_state": np.array([0, 0, stale, 0, 0, 0], np.float32),
        "compute_budget": np.array([0.5, 0.25, affordable, elapsed, 0], np.float32),
    }


def baseline_factories() -> list[Callable[[Observation], int]]:
    return [
        greedy_factory(context=CONTEXT),
        periodic_factory(context=CONTEXT),
        scene_change_factory(context=CONTEXT, threshold=0.5),
        staleness_factory(context=CONTEXT, threshold=0.5),
    ]


def test_each_baseline_returns_only_binary_actions() -> None:
    actions = [policy(observation(scene=1, stale=1, elapsed=1)) for policy in baseline_factories()]
    assert actions == [RUN_DETECTOR, RUN_DETECTOR, RUN_DETECTOR, RUN_DETECTOR]
    assert all(action in (SKIP, RUN_DETECTOR) for action in actions)


def test_every_baseline_skips_when_run_is_unaffordable() -> None:
    assert [policy(observation(affordable=0.0, scene=1, stale=1, elapsed=1)) for policy in baseline_factories()] == [
        SKIP,
        SKIP,
        SKIP,
        SKIP,
    ]


def test_periodic_uses_elapsed_source_time_not_an_observation_index() -> None:
    policy = periodic_factory(context=CONTEXT)

    assert policy(observation(elapsed=0.05)) == SKIP
    assert policy(observation(elapsed=0.081)) == RUN_DETECTOR


def test_scene_change_uses_the_causal_three_by_three_mean() -> None:
    policy = scene_change_factory(context=CONTEXT, threshold=0.5)
    mostly_quiet = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], np.float32)
    mostly_changed = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]], np.float32)

    assert policy(observation(scene=mostly_quiet)) == SKIP
    assert policy(observation(scene=mostly_changed)) == RUN_DETECTOR


def test_staleness_uses_tracker_state_observation_index_two() -> None:
    policy = staleness_factory(context=CONTEXT, threshold=0.5)
    fresh = observation(stale=0.4)
    fresh["tracker_state"][[0, 1, 3, 4, 5]] = 1.0
    stale = observation(stale=0.6)
    stale["tracker_state"][[0, 1, 3, 4, 5]] = 0.0

    assert policy(fresh) == SKIP
    assert policy(stale) == RUN_DETECTOR


def test_reset_policy_calls_optional_seeded_reset_and_ignores_plain_callables() -> None:
    class Resettable:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def reset(self, *, seed: int) -> None:
            self.seeds.append(seed)

        def __call__(self, _observation: Observation) -> int:
            return SKIP

    resettable = Resettable()
    reset_policy(resettable, seed=9)
    assert resettable.seeds == [9]

    called = False

    def ordinary(_observation: Observation) -> int:
        nonlocal called
        called = True
        return SKIP

    reset_policy(ordinary, seed=9)
    assert ordinary(observation()) == SKIP
    assert called


@pytest.mark.parametrize("factory", [scene_change_factory, staleness_factory])
def test_threshold_factories_require_unit_interval(factory: Callable[..., object]) -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        factory(context=CONTEXT, threshold=1.1)
