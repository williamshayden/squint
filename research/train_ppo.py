from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from squint_rl.budget import BudgetConfig
from squint_rl.env import Info, SquintEnv
from squint_rl.episode import Episode
from squint_rl.tracker import (
    Observation,
    ObservationScales,
    PolicyContext,
    Tracker,
)

_TRAIN_EPISODE_IDS = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN")
_TRAIN_SOURCE_IDS = ("02", "04", "05", "10")
_NORMALIZATION_FIELDS = ("active_tracks", "age_s", "motion_px_s", "time_since_detector_s")


def _actual_interactions(requested: int, rollout_steps: int) -> int:
    if type(requested) is not int or requested <= 0:
        raise ValueError("requested interactions must be a positive built-in integer")
    if type(rollout_steps) is not int or rollout_steps <= 0:
        raise ValueError("rollout_steps must be a positive built-in integer")
    return ((requested + rollout_steps - 1) // rollout_steps) * rollout_steps


@dataclass(frozen=True, slots=True)
class PPORecipe:
    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    training_steps: int = 500_000
    validation_interval: int = 50_000
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    net_arch: tuple[int, int] = (64, 64)

    def to_record(self) -> dict[str, object]:
        validation_actual = _actual_interactions(self.validation_interval, self.rollout_steps)
        training_actual = _actual_interactions(self.training_steps, self.rollout_steps)
        recipe = asdict(self)
        recipe["seeds"] = list(self.seeds)
        recipe["net_arch"] = list(self.net_arch)
        return {
            "algorithm": "PPO",
            "library": {"name": "stable-baselines3", "version": "2.8.0"},
            "policy": "MultiInputPolicy",
            "environment_count": 1,
            "device": "cpu",
            "verbose": 1,
            "recipe": recipe,
            "policy_kwargs": {
                "activation_fn": "torch.nn.Tanh",
                "net_arch": {
                    "pi": list(self.net_arch),
                    "vf": list(self.net_arch),
                },
            },
            "requested_actual_interactions": [
                {"requested": self.validation_interval, "actual": validation_actual},
                {"requested": self.training_steps, "actual": training_actual},
            ],
            "omitted_defaults": {
                "n_epochs": 10,
                "clip_range_vf": None,
                "normalize_advantage": True,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": -1,
                "rollout_buffer_class": None,
                "rollout_buffer_kwargs": None,
                "target_kl": None,
                "stats_window_size": 100,
                "tensorboard_log": None,
                "_init_setup_model": True,
            },
        }


def deterministic_windows(*, frame_count: int, fps: float) -> tuple[tuple[int, int], ...]:
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive built-in integer")
    if type(fps) not in (int, float):
        raise ValueError("fps must be a positive finite built-in number")
    try:
        numeric_fps = float(fps)
        width_value = 10.0 * numeric_fps
        stride_value = 5.0 * numeric_fps
    except OverflowError as error:
        raise ValueError("fps-derived window durations must be finite") from error
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (numeric_fps, width_value, stride_value)
    ):
        raise ValueError("fps-derived window durations must be finite and positive")
    width = round(width_value)
    stride = round(stride_value)
    if width <= 0 or stride <= 0:
        raise ValueError("fps must produce positive window width and stride")
    if frame_count < width:
        raise ValueError("episode must contain at least one complete window")
    return tuple((start, start + width) for start in range(0, frame_count - width + 1, stride))


def _manifest_object(episode: Episode, name: str) -> Mapping[str, object]:
    value = episode.manifest.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"manifest {name} must be an object")
    return cast(Mapping[str, object], value)


def _positive_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validate_training_episodes(
    episodes: tuple[Episode, ...], scales: ObservationScales
) -> float:
    if len(episodes) != len(_TRAIN_EPISODE_IDS):
        raise ValueError("training episodes must contain exactly 02, 04, 05, and 10")
    profiles: list[tuple[Mapping[str, object], ...]] = []
    reserve_ms: float | None = None
    expected_normalization = {name: getattr(scales, name) for name in _NORMALIZATION_FIELDS}
    for index, episode in enumerate(episodes):
        if type(episode) is not Episode:
            raise TypeError("training episodes must be full Episode objects")
        episode_record = _manifest_object(episode, "episode")
        source = _manifest_object(episode, "source")
        if episode_record.get("id") != _TRAIN_EPISODE_IDS[index]:
            raise ValueError("training episode IDs or order are not canonical")
        if source.get("id") != _TRAIN_SOURCE_IDS[index]:
            raise ValueError("training source IDs or order are not canonical")
        if source.get("dataset") != "MOT17" or source.get("split") != "train":
            raise ValueError("training episodes must be MOT17 train episodes")
        artifacts = _manifest_object(episode, "artifacts")
        if artifacts.get("content_sha256") != episode.content_sha256:
            raise ValueError("episode content identity does not match its manifest")
        detector = _manifest_object(episode, "detector")
        hardware = _manifest_object(episode, "hardware")
        cost_profile = _manifest_object(episode, "cost_profile")
        normalization = _manifest_object(episode, "normalization")
        profile_sha256 = cost_profile.get("profile_sha256")
        if (
            not isinstance(profile_sha256, str)
            or len(profile_sha256) != 64
            or any(character not in "0123456789abcdef" for character in profile_sha256)
        ):
            raise ValueError("cost_profile.profile_sha256 must be lowercase 64-hex")
        current_reserve = _positive_number(
            cost_profile.get("reserve_ms"), "cost_profile.reserve_ms"
        )
        if normalization != expected_normalization:
            raise ValueError("manifest normalization must match observation scales")
        if reserve_ms is None:
            reserve_ms = current_reserve
        elif current_reserve != reserve_ms:
            raise ValueError("training episodes must share reserve_ms")
        profiles.append((detector, hardware, cost_profile, normalization))
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError("training episodes must share one frozen profile")
    assert reserve_ms is not None
    return reserve_ms


class WindowSamplerEnv(gym.Env[Observation, int]):
    def __init__(
        self,
        *,
        episodes: Sequence[Episode],
        tracker_factory: Callable[..., Tracker],
        tracker_parameters: Mapping[str, object],
        observation_scales: ObservationScales,
    ) -> None:
        parents = tuple(episodes)
        reserve_ms = _validate_training_episodes(parents, observation_scales)
        self._windows = tuple(
            (episode, start, stop)
            for episode in parents
            for start, stop in deterministic_windows(
                frame_count=episode.frame_count, fps=episode.fps
            )
        )
        self._tracker_factory = tracker_factory
        self._tracker_parameters = dict(tracker_parameters)
        self._observation_scales = observation_scales
        self._reserve_ms = reserve_ms
        parent, start, stop = self._windows[0]
        sample = self._make_env(parent, start, stop, rho=0.10)
        self.action_space = sample.action_space
        self.observation_space = sample.observation_space
        self.metadata = sample.metadata
        self._inner: SquintEnv | None = None

    def _make_env(self, parent: Episode, start: int, stop: int, *, rho: float) -> SquintEnv:
        tracker = self._tracker_factory(episode=parent, **self._tracker_parameters)
        view = parent.slice(start, stop)
        budget = BudgetConfig.for_rate(
            reserve_ms=self._reserve_ms, source_fps=view.fps, nominal_rate=rho
        )
        return SquintEnv(
            episode=view,
            tracker=tracker,
            budget=budget,
            observation_scales=self._observation_scales,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self._windows)))
        rho = float(self.np_random.uniform(0.10, 1.00))
        parent, start, stop = self._windows[index]
        inner = self._make_env(parent, start, stop, rho=rho)
        if (
            inner.action_space != self.action_space
            or inner.observation_space != self.observation_space
        ):
            raise ValueError("sampled environment spaces do not match stable spaces")
        observation, inner_info = inner.reset(seed=seed, options=options)
        self._inner = inner
        info = dict(inner_info)
        info.update({"episode_id": _manifest_object(parent, "episode")["id"],
                     "window_start": start, "window_stop": stop, "rho": rho})
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, Info]:
        if self._inner is None:
            raise RuntimeError("reset must be called before step")
        return self._inner.step(action)


class SceneChangeMask(gym.ObservationWrapper[Observation, int, Observation]):
    def observation(self, observation: Observation) -> Observation:
        masked = dict(observation)
        masked["scene_change"] = np.zeros_like(observation["scene_change"])
        return masked


def build_model(env: gym.Env[Observation, int], *, seed: int, recipe: PPORecipe) -> PPO:
    if type(seed) is not int:
        raise ValueError("seed must be a built-in integer")
    if type(recipe) is not PPORecipe:
        raise TypeError("recipe must be exactly PPORecipe")
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=recipe.learning_rate,
        n_steps=recipe.rollout_steps,
        batch_size=recipe.batch_size,
        gamma=recipe.gamma,
        gae_lambda=recipe.gae_lambda,
        clip_range=recipe.clip_range,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": {"pi": list(recipe.net_arch), "vf": list(recipe.net_arch)},
        },
        device="cpu",
        seed=seed,
        verbose=1,
    )


class FrozenPPOPolicy:
    def __init__(self, checkpoint: str | Path, *, mask_scene_change: bool = False) -> None:
        if type(mask_scene_change) is not bool:
            raise ValueError("mask_scene_change must be a bool")
        self.model = PPO.load(checkpoint, device="cpu")
        self.mask_scene_change = mask_scene_change

    def __call__(self, observation: Observation) -> int:
        model_observation = dict(observation)
        if self.mask_scene_change:
            model_observation["scene_change"] = np.zeros_like(observation["scene_change"])
        action, _ = self.model.predict(model_observation, deterministic=True)
        return int(action)


def frozen_policy_factory(
    *,
    context: PolicyContext,
    checkpoint: str,
    observation_ablation: str | None = None,
) -> FrozenPPOPolicy:
    del context
    if observation_ablation not in (None, "no-scene-change"):
        raise ValueError("observation_ablation must be None or no-scene-change")
    return FrozenPPOPolicy(checkpoint, mask_scene_change=observation_ablation == "no-scene-change")
