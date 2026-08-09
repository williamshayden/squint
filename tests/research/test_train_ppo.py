from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
import pytest
import torch
from research.train_ppo import (
    FrozenPPOPolicy,
    PPORecipe,
    SceneChangeMask,
    WindowSamplerEnv,
    build_model,
    deterministic_windows,
    frozen_policy_factory,
)

from research import train_ppo
from squint_rl.episode import Episode, EpisodeView, seal_episode
from squint_rl.synthetic import make_synthetic_episode
from squint_rl.tracker import (
    DetectionBatch,
    Observation,
    ObservationScales,
    PolicyContext,
    TrackBatch,
    TrackerSummary,
)

SCALES = ObservationScales(
    active_tracks=8.0,
    age_s=5.0,
    motion_px_s=20.0,
    time_since_detector_s=5.0,
)
TRACKER_PARAMETERS = {"track_activation_threshold": 0.25}
TRAIN_IDS = ("02", "04", "05", "10")


class RecordingTracker:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.steps: list[tuple[DetectionBatch | None, float]] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.steps.clear()

    def step(
        self, detections: DetectionBatch | None, timestamp_s: float
    ) -> TrackBatch:
        self.steps.append((detections, timestamp_s))
        return TrackBatch.empty()

    def summary(self) -> TrackerSummary:
        return TrackerSummary.empty()


class RecordingTrackerFactory:
    def __init__(self) -> None:
        self.parents: list[Episode] = []
        self.parameters: list[dict[str, object]] = []
        self.trackers: list[RecordingTracker] = []

    def __call__(self, *, episode: Episode, **parameters: object) -> RecordingTracker:
        tracker = RecordingTracker()
        self.parents.append(episode)
        self.parameters.append(dict(parameters))
        self.trackers.append(tracker)
        return tracker


def _plain_manifest(episode: Episode) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((episode.path / "manifest.json").read_text(encoding="utf-8")),
    )


def _training_episode(path: Path, source_id: str) -> Episode:
    source = make_synthetic_episode(
        path.with_name(f"{path.name}-source"),
        frame_count=31,
        fps=2.0,
        change_frames=(0, 10, 20, 30),
        latency_ms=10.0,
    )
    base = Episode.open(source)
    manifest = _plain_manifest(base)
    manifest["episode"]["id"] = f"MOT17-{source_id}-FRCNN"
    manifest["source"].update(
        {"id": source_id, "dataset": "MOT17", "split": "train"}
    )
    manifest["artifacts"] = {}
    arrays = {name: np.array(value, copy=True) for name, value in base.arrays.items()}
    return Episode.open(seal_episode(path, manifest=manifest, arrays=arrays))


@pytest.fixture
def training_episodes(tmp_path: Path) -> tuple[Episode, ...]:
    return tuple(
        _training_episode(tmp_path / f"train-{source_id}", source_id)
        for source_id in TRAIN_IDS
    )


def _sampler(
    episodes: tuple[Episode, ...], factory: RecordingTrackerFactory
) -> WindowSamplerEnv:
    return WindowSamplerEnv(
        episodes=episodes,
        tracker_factory=factory,
        tracker_parameters=TRACKER_PARAMETERS,
        observation_scales=SCALES,
    )


def _mutated_episode(
    episode: Episode,
    mutate: Any,
    *,
    content_sha256: str | None = None,
) -> Episode:
    manifest = _plain_manifest(episode)
    mutate(manifest)
    return replace(
        episode,
        manifest=manifest,
        content_sha256=(
            episode.content_sha256 if content_sha256 is None else content_sha256
        ),
    )


def _assert_sampler_rejects_before_tracker(
    episodes: tuple[Episode | EpisodeView, ...],
    *,
    scales: ObservationScales = SCALES,
) -> None:
    factory = RecordingTrackerFactory()
    with pytest.raises((TypeError, ValueError)):
        WindowSamplerEnv(
            episodes=cast(Any, episodes),
            tracker_factory=factory,
            tracker_parameters=TRACKER_PARAMETERS,
            observation_scales=scales,
        )
    assert factory.parents == []


def test_research_package_is_cold_on_import() -> None:
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
import research
for name in (
    'research.train_ppo', 'stable_baselines3', 'torch', 'gymnasium',
    'numpy', 'squint_rl',
):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_default_recipe_has_exact_json_safe_record() -> None:
    recipe = PPORecipe()
    assert {field.name: getattr(recipe, field.name) for field in fields(recipe)} == {
        "learning_rate": 3e-4,
        "rollout_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "training_steps": 500_000,
        "validation_interval": 50_000,
        "seeds": (0, 1, 2, 3, 4),
        "net_arch": (64, 64),
    }
    expected = {
        "algorithm": "PPO",
        "library": {"name": "stable-baselines3", "version": "2.8.0"},
        "policy": "MultiInputPolicy",
        "environment_count": 1,
        "device": "cpu",
        "verbose": 1,
        "recipe": {
            "learning_rate": 3e-4,
            "rollout_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "training_steps": 500_000,
            "validation_interval": 50_000,
            "seeds": [0, 1, 2, 3, 4],
            "net_arch": [64, 64],
        },
        "policy_kwargs": {
            "activation_fn": "torch.nn.Tanh",
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        "requested_actual_interactions": [
            {"requested": 50_000, "actual": 51_200},
            {"requested": 500_000, "actual": 501_760},
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
    record = recipe.to_record()
    assert record == expected
    assert len(cast(dict[str, object], record["omitted_defaults"])) == 14
    assert json.loads(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    ) == expected


def test_recipe_record_derives_overrides_and_returns_fresh_lists() -> None:
    recipe = PPORecipe(
        learning_rate=1e-4,
        rollout_steps=7,
        batch_size=8,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.1,
        training_steps=20,
        validation_interval=10,
        seeds=(7, 8),
        net_arch=(32, 16),
    )
    first = recipe.to_record()
    second = recipe.to_record()
    recipe_record = cast(dict[str, object], first["recipe"])
    policy_arch = cast(
        dict[str, list[int]],
        cast(dict[str, object], first["policy_kwargs"])["net_arch"],
    )
    assert recipe_record == {
        "learning_rate": 1e-4,
        "rollout_steps": 7,
        "batch_size": 8,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "clip_range": 0.1,
        "training_steps": 20,
        "validation_interval": 10,
        "seeds": [7, 8],
        "net_arch": [32, 16],
    }
    assert policy_arch == {"pi": [32, 16], "vf": [32, 16]}
    assert policy_arch["pi"] is not policy_arch["vf"]
    assert first["requested_actual_interactions"] == [
        {"requested": 10, "actual": 14},
        {"requested": 20, "actual": 21},
    ]
    cast(list[int], recipe_record["seeds"]).append(99)
    policy_arch["pi"].append(99)
    assert cast(dict[str, object], second["recipe"])["seeds"] == [7, 8]
    second_policy_arch = cast(
        dict[str, list[int]],
        cast(dict[str, object], second["policy_kwargs"])["net_arch"],
    )
    assert second_policy_arch == {"pi": [32, 16], "vf": [32, 16]}


@pytest.mark.parametrize("value", [True, np.int64(7), 0, -1])
@pytest.mark.parametrize("field", ["rollout_steps", "validation_interval", "training_steps"])
def test_recipe_rejects_invalid_interaction_arithmetic(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        PPORecipe(**{field: value}).to_record()  # type: ignore[arg-type]


def test_deterministic_windows_uses_complete_half_open_windows() -> None:
    assert deterministic_windows(frame_count=301, fps=10.0) == (
        (0, 100),
        (50, 150),
        (100, 200),
        (150, 250),
        (200, 300),
    )
    assert deterministic_windows(frame_count=100, fps=10) == ((0, 100),)


@pytest.mark.parametrize("frame_count", [True, np.int64(301), 0, -1, "301"])
def test_deterministic_windows_rejects_invalid_frame_count(frame_count: object) -> None:
    with pytest.raises(ValueError):
        deterministic_windows(frame_count=cast(Any, frame_count), fps=10.0)


@pytest.mark.parametrize(
    "fps",
    [True, np.float64(10.0), "10", 0, -1, float("nan"), float("inf"), 0.01, 10**400],
)
def test_deterministic_windows_rejects_invalid_fps(fps: object) -> None:
    with pytest.raises(ValueError):
        deterministic_windows(frame_count=301, fps=cast(Any, fps))


def test_deterministic_windows_rejects_short_episode() -> None:
    with pytest.raises(ValueError, match="complete"):
        deterministic_windows(frame_count=99, fps=10.0)


def test_sampler_flattens_canonical_windows_and_exposes_stable_spaces(
    training_episodes: tuple[Episode, ...],
) -> None:
    factory = RecordingTrackerFactory()
    sampler = _sampler(training_episodes, factory)
    assert [
        (
            cast(str, episode.manifest["source"]["id"]),  # type: ignore[index]
            start,
            stop,
        )
        for episode, start, stop in sampler._windows
    ] == [
        (source_id, start, stop)
        for source_id in TRAIN_IDS
        for start, stop in ((0, 20), (10, 30))
    ]
    assert factory.parents == [training_episodes[0]]
    assert factory.parameters == [TRACKER_PARAMETERS]
    action_space = sampler.action_space
    observation_space = sampler.observation_space
    metadata = sampler.metadata

    _observation, info = sampler.reset(seed=19, options={"caller": "retained"})
    assert set(info) == {"episode_id", "window_start", "window_stop", "rho"}
    assert sampler.action_space is action_space
    assert sampler.observation_space is observation_space
    assert sampler.metadata is metadata
    assert isinstance(sampler._inner, train_ppo.SquintEnv)
    assert isinstance(sampler._inner.episode, EpisodeView)
    assert factory.parents[-1] is sampler._inner.episode.parent
    assert factory.parameters[-1] == TRACKER_PARAMETERS


def test_sampler_seed_stream_is_deterministic_continuous_and_reseedable(
    training_episodes: tuple[Episode, ...],
) -> None:
    left = _sampler(training_episodes, RecordingTrackerFactory())
    right = _sampler(training_episodes, RecordingTrackerFactory())

    left_records = [left.reset(seed=123)[1]]
    right_records = [right.reset(seed=123)[1]]
    for _ in range(24):
        left_records.append(left.reset()[1])
        right_records.append(right.reset()[1])
    assert left_records == right_records
    rhos = [cast(float, item["rho"]) for item in left_records]
    assert all(0.10 <= rho < 1.00 for rho in rhos)
    assert any(rho not in {0.10, 0.25, 0.50, 0.75, 1.00} for rho in rhos)

    replay = [left.reset(seed=123)[1], left.reset()[1]]
    assert replay == left_records[:2]


def test_sampler_creates_fresh_tracker_view_budget_and_inner_per_reset(
    training_episodes: tuple[Episode, ...],
) -> None:
    factory = RecordingTrackerFactory()
    sampler = _sampler(training_episodes, factory)
    _first_observation, first_info = sampler.reset(seed=4)
    first_inner = sampler._inner
    first_tracker = factory.trackers[-1]
    assert first_inner is not None
    assert first_inner.bucket.config.reserve_ms == 10.0
    assert first_inner.bucket.config.refill_ms_per_s == pytest.approx(
        cast(float, first_info["rho"]) * first_inner.episode.fps * 10.0
    )

    _second_observation, _second_info = sampler.reset()
    second_inner = sampler._inner
    second_tracker = factory.trackers[-1]
    assert second_inner is not None
    assert second_inner is not first_inner
    assert second_tracker is not first_tracker
    assert second_inner.episode is not first_inner.episode
    assert len(factory.trackers) == 3  # throwaway sample plus two resets

    _observation, _reward, _terminated, _truncated, info = sampler.step(0)
    assert set(info) == {
        "requested_action",
        "applied_action",
        "denied",
        "balance_ms",
        "charged_ms",
        "detector_calls",
        "matches",
        "false_positives",
        "false_negatives",
        "identity_switches",
        "localization_error",
    }
    assert len(second_tracker.steps) == 1
    assert first_tracker.steps == []


def test_sampler_step_requires_reset(
    training_episodes: tuple[Episode, ...],
) -> None:
    sampler = _sampler(training_episodes, RecordingTrackerFactory())
    with pytest.raises(RuntimeError, match="reset"):
        sampler.step(0)


def test_sampler_rejects_inner_space_drift(
    training_episodes: tuple[Episode, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_env = train_ppo.SquintEnv
    constructed = 0

    def drifting_env(**kwargs: Any) -> train_ppo.SquintEnv:
        nonlocal constructed
        env = real_env(**kwargs)
        constructed += 1
        if constructed > 1:
            env.action_space = gym.spaces.Discrete(3)
        return env

    monkeypatch.setattr(train_ppo, "SquintEnv", drifting_env)
    sampler = _sampler(training_episodes, RecordingTrackerFactory())
    with pytest.raises(ValueError, match="spaces"):
        sampler.reset(seed=1)


def test_sampler_rejects_noncanonical_members_before_tracker(
    training_episodes: tuple[Episode, ...],
) -> None:
    _assert_sampler_rejects_before_tracker(training_episodes[:3])
    _assert_sampler_rejects_before_tracker(
        (*training_episodes[:3], training_episodes[2])
    )
    _assert_sampler_rejects_before_tracker(
        (training_episodes[1], training_episodes[0], *training_episodes[2:])
    )
    _assert_sampler_rejects_before_tracker(
        cast(tuple[EpisodeView, ...], (training_episodes[0].slice(0, 20),))
        + cast(tuple[EpisodeView, ...], training_episodes[1:])
    )


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("split", lambda manifest: manifest["source"].__setitem__("split", "validation")),
        ("dataset", lambda manifest: manifest["source"].__setitem__("dataset", "other")),
        ("episode", lambda manifest: manifest["episode"].__setitem__("id", "wrong")),
        ("source", lambda manifest: manifest["source"].__setitem__("id", "99")),
        ("detector", lambda manifest: manifest["detector"].__setitem__("revision", "other")),
        ("hardware", lambda manifest: manifest["hardware"].__setitem__("runtime_version", "other")),
        ("cost", lambda manifest: manifest["cost_profile"].__setitem__("unit", "frames")),
        ("normalization", lambda manifest: manifest["normalization"].__setitem__("age_s", 6)),
        ("profile", lambda manifest: manifest["cost_profile"].__setitem__("profile_sha256", "b" * 64)),
        ("profile-format", lambda manifest: manifest["cost_profile"].__setitem__("profile_sha256", "A" * 64)),
        ("content", lambda manifest: manifest["artifacts"].__setitem__("content_sha256", "b" * 64)),
    ],
)
def test_sampler_rejects_manifest_spoofs_before_tracker(
    training_episodes: tuple[Episode, ...], label: str, mutate: Any
) -> None:
    del label
    changed = _mutated_episode(training_episodes[0], mutate)
    _assert_sampler_rejects_before_tracker((changed, *training_episodes[1:]))


@pytest.mark.parametrize("reserve", [True, "10", 0, -1, float("nan"), float("inf")])
def test_sampler_rejects_invalid_reserve_before_tracker(
    training_episodes: tuple[Episode, ...], reserve: object
) -> None:
    changed = _mutated_episode(
        training_episodes[0],
        lambda manifest: manifest["cost_profile"].__setitem__(
            "reserve_ms", reserve
        ),
    )
    _assert_sampler_rejects_before_tracker((changed, *training_episodes[1:]))


@pytest.mark.parametrize("source_id", ["09", "11", "13"])
def test_sampler_rejects_heldout_labels_before_tracker(
    training_episodes: tuple[Episode, ...], source_id: str
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["episode"]["id"] = f"MOT17-{source_id}-FRCNN"
        manifest["source"].update(
            {
                "id": source_id,
                "split": "validation" if source_id == "09" else "test",
            }
        )

    changed = _mutated_episode(training_episodes[0], mutate)
    _assert_sampler_rejects_before_tracker((changed, *training_episodes[1:]))


def test_sampler_rejects_content_identity_and_scale_spoofs_before_tracker(
    training_episodes: tuple[Episode, ...],
) -> None:
    changed = replace(training_episodes[0], content_sha256="c" * 64)
    _assert_sampler_rejects_before_tracker((changed, *training_episodes[1:]))
    _assert_sampler_rejects_before_tracker(
        training_episodes,
        scales=replace(SCALES, active_tracks=9.0),
    )


class ObservationEnv(gym.Env[Observation, int]):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "scene_change": gym.spaces.Box(0.0, 1.0, (3, 3), np.float32),
                "tracker_state": gym.spaces.Box(0.0, 1.0, (6,), np.float32),
                "compute_budget": gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
            }
        )
        self.last_observation: Observation | None = None

    def _observation(self) -> Observation:
        observation = {
            "scene_change": np.ones((3, 3), np.float32),
            "tracker_state": np.arange(6, dtype=np.float32),
            "compute_budget": np.arange(5, dtype=np.float32),
            "future_key": np.array([7.0], np.float32),
        }
        self.last_observation = observation
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return self._observation(), {"origin": 1}

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        del action
        return self._observation(), 2.0, True, False, {"origin": 2}


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_scene_change_mask_only_replaces_scene_array(operation: str) -> None:
    base = ObservationEnv()
    wrapper = SceneChangeMask(base)
    action_space = base.action_space
    observation_space = base.observation_space
    if operation == "reset":
        masked, info = wrapper.reset(seed=3)
        assert info == {"origin": 1}
    else:
        masked, reward, terminated, truncated, info = wrapper.step(0)
        assert (reward, terminated, truncated, info) == (2.0, True, False, {"origin": 2})
    original = cast(Observation, base.last_observation)
    assert masked is not original
    assert masked["scene_change"] is not original["scene_change"]
    np.testing.assert_array_equal(masked["scene_change"], np.zeros((3, 3), np.float32))
    np.testing.assert_array_equal(original["scene_change"], np.ones((3, 3), np.float32))
    for key in ("tracker_state", "compute_budget", "future_key"):
        assert masked[key] is original[key]
    assert wrapper.action_space is action_space
    assert wrapper.observation_space is observation_space


def test_build_model_calls_ppo_once_with_exact_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def constructor(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(train_ppo, "PPO", constructor)
    env = cast(Any, object())
    recipe = PPORecipe(net_arch=(32, 16))
    assert build_model(env, seed=7, recipe=recipe) is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("MultiInputPolicy", env)
    assert set(kwargs) == {
        "learning_rate",
        "n_steps",
        "batch_size",
        "gamma",
        "gae_lambda",
        "clip_range",
        "policy_kwargs",
        "device",
        "seed",
        "verbose",
    }
    assert {key: kwargs[key] for key in kwargs if key != "policy_kwargs"} == {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "device": "cpu",
        "seed": 7,
        "verbose": 1,
    }
    policy_kwargs = cast(dict[str, object], kwargs["policy_kwargs"])
    assert policy_kwargs["activation_fn"] is torch.nn.Tanh
    architecture = cast(dict[str, list[int]], policy_kwargs["net_arch"])
    assert architecture == {"pi": [32, 16], "vf": [32, 16]}
    assert architecture["pi"] is not architecture["vf"]


@pytest.mark.parametrize("seed", [True, np.int64(1), 1.0, "1"])
def test_build_model_rejects_non_builtin_integer_seed(
    monkeypatch: pytest.MonkeyPatch, seed: object
) -> None:
    monkeypatch.setattr(
        train_ppo,
        "PPO",
        lambda *args, **kwargs: pytest.fail("PPO must not be constructed"),
    )
    with pytest.raises(ValueError):
        build_model(cast(Any, object()), seed=cast(Any, seed), recipe=PPORecipe())


def test_build_model_rejects_recipe_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    class DerivedRecipe(PPORecipe):
        pass

    monkeypatch.setattr(
        train_ppo,
        "PPO",
        lambda *args, **kwargs: pytest.fail("PPO must not be constructed"),
    )
    with pytest.raises(TypeError):
        build_model(cast(Any, object()), seed=1, recipe=DerivedRecipe())


class PredictModel:
    def __init__(self, action: object = np.int64(1)) -> None:
        self.action = action
        self.calls: list[tuple[Mapping[str, np.ndarray[Any, Any]], dict[str, object]]] = []

    def predict(
        self, observation: Mapping[str, np.ndarray[Any, Any]], **kwargs: object
    ) -> tuple[object, None]:
        self.calls.append((observation, kwargs))
        return self.action, None


class Loader:
    loads: ClassVar[list[tuple[object, str]]] = []
    models: ClassVar[list[PredictModel]] = []

    @classmethod
    def load(cls, checkpoint: object, *, device: str) -> PredictModel:
        model = PredictModel()
        cls.loads.append((checkpoint, device))
        cls.models.append(model)
        return model


@pytest.fixture(autouse=True)
def clear_loader() -> None:
    Loader.loads.clear()
    Loader.models.clear()


def _policy_observation() -> Observation:
    return {
        "scene_change": np.ones((3, 3), np.float32),
        "tracker_state": np.arange(6, dtype=np.float32),
        "compute_budget": np.arange(5, dtype=np.float32),
    }


@pytest.mark.parametrize("masked", [False, True])
def test_frozen_policy_loads_cpu_and_predicts_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, masked: bool
) -> None:
    monkeypatch.setattr(train_ppo, "PPO", Loader)
    checkpoint = tmp_path / "best.zip"
    policy = FrozenPPOPolicy(checkpoint, mask_scene_change=masked)
    observation = _policy_observation()
    original_scene = observation["scene_change"].copy()
    assert policy(observation) == 1
    assert Loader.loads == [(checkpoint, "cpu")]
    model_observation, kwargs = Loader.models[0].calls[0]
    assert model_observation is not observation
    assert kwargs == {"deterministic": True}
    assert model_observation["tracker_state"] is observation["tracker_state"]
    assert model_observation["compute_budget"] is observation["compute_budget"]
    if masked:
        assert model_observation["scene_change"] is not observation["scene_change"]
        np.testing.assert_array_equal(
            model_observation["scene_change"], np.zeros((3, 3), np.float32)
        )
    else:
        assert model_observation["scene_change"] is observation["scene_change"]
    np.testing.assert_array_equal(observation["scene_change"], original_scene)


@pytest.mark.parametrize("mask", [0, 1, None, "yes", np.bool_(True)])
def test_frozen_policy_rejects_non_bool_mask(
    monkeypatch: pytest.MonkeyPatch, mask: object
) -> None:
    monkeypatch.setattr(train_ppo, "PPO", Loader)
    with pytest.raises(ValueError):
        FrozenPPOPolicy("checkpoint.zip", mask_scene_change=cast(Any, mask))
    assert Loader.loads == []


def test_frozen_factory_uses_exact_ablation_and_loads_fresh_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_ppo, "PPO", Loader)
    context = PolicyContext(
        nominal_rate=0.25,
        source_fps=30.0,
        reserve_ms=10.0,
        seed=4,
        time_since_detector_scale_s=5.0,
    )
    first = frozen_policy_factory(context=context, checkpoint="best.zip")
    second = frozen_policy_factory(
        context=context,
        checkpoint="best.zip",
        observation_ablation="no-scene-change",
    )
    assert first is not second
    assert not first.mask_scene_change
    assert second.mask_scene_change
    assert Loader.loads == [("best.zip", "cpu"), ("best.zip", "cpu")]


@pytest.mark.parametrize("ablation", ["", "scene-change", "NO-SCENE-CHANGE", 1])
def test_frozen_factory_rejects_other_ablations(
    monkeypatch: pytest.MonkeyPatch, ablation: object
) -> None:
    monkeypatch.setattr(train_ppo, "PPO", Loader)
    context = PolicyContext(0.25, 30.0, 10.0, 4, 5.0)
    with pytest.raises(ValueError):
        frozen_policy_factory(
            context=context,
            checkpoint="best.zip",
            observation_ablation=cast(Any, ablation),
        )
    assert Loader.loads == []
