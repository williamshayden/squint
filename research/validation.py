from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from research.train_ppo import PPORecipe
from squint_rl.policies import Policy
from squint_rl.tracker import Observation

M0_VALIDATION_RATES = (0.10, 0.25, 0.50, 0.75, 1.00)
ObservationCondition = Literal["full", "no-scene-change"]
_M0_MILESTONES = (
    (50_000, 51_200), (100_000, 100_352), (150_000, 151_552), (200_000, 200_704), (250_000, 251_904), (300_000, 301_056), (350_000, 350_208), (400_000, 401_408), (450_000, 450_560), (500_000, 501_760),
)


@dataclass(frozen=True, slots=True)
class ValidationPoint:
    nominal_rate: float
    realized_compute: float
    hota: float


@dataclass(frozen=True, slots=True)
class ValidationCurve:
    episode_id: str
    source_id: str
    dataset: str
    split: str
    episode_content_sha256: str
    condition: ObservationCondition
    points: tuple[ValidationPoint, ...]
    action_sha256: str
    metric_input_sha256: str


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    requested_interactions: int
    trained_through_interactions: int
    auc: float
    selected: bool
    curve: ValidationCurve


class ValidationRunner(Protocol):
    def __call__(
        self, *, policy: Policy,
        requested_interactions: int,
        trained_through_interactions: int,
    ) -> ValidationCurve: ...


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite built-in number")
    try:
        number = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_support(support: object) -> tuple[float, float]:
    if type(support) is not tuple or len(support) != 2:
        raise TypeError("support must be an exact two-item tuple")
    lower = _finite_number(support[0], "support lower endpoint")
    upper = _finite_number(support[1], "support upper endpoint")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("support must satisfy 0 <= lower < upper <= 1")
    return lower, upper


def fixed_support_auc(
    points: Sequence[ValidationPoint], *, support: tuple[float, float]
) -> float:
    lower, upper = _validate_support(support)
    snapshot = tuple(points)
    if len(snapshot) < 2:
        raise ValueError("at least two validation points are required")
    copied: list[tuple[float, float]] = []
    for point in snapshot:
        if type(point) is not ValidationPoint:
            raise TypeError("points must contain exact ValidationPoint objects")
        nominal_rate = _finite_number(point.nominal_rate, "nominal_rate")
        realized_compute = _finite_number(point.realized_compute, "realized_compute")
        hota = _finite_number(point.hota, "hota")
        if not 0.0 < nominal_rate <= 1.0:
            raise ValueError("nominal_rate must be in (0, 1]")
        if not 0.0 <= realized_compute <= 1.0:
            raise ValueError("realized_compute must be in [0, 1]")
        if not 0.0 <= hota <= 1.0:
            raise ValueError("hota must be in [0, 1]")
        copied.append((realized_compute, hota))
    copied.sort()
    if any(left[0] == right[0] for left, right in pairwise(copied)):
        raise ValueError("duplicate realized-compute coordinates are forbidden")
    if copied[0][0] > lower or copied[-1][0] < upper:
        raise ValueError("observed points must cover both support endpoints")
    grid = np.linspace(lower, upper, 101)
    values = np.interp(grid, [item[0] for item in copied], [item[1] for item in copied])
    result = float(np.trapezoid(values, grid) / (upper - lower))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("normalized AUC must be finite and in [0, 1]")
    return result


def _validate_condition(condition: object) -> ObservationCondition:
    if type(condition) is not str or condition not in ("full", "no-scene-change"):
        raise ValueError("condition must be exactly full or no-scene-change")
    return cast(ObservationCondition, condition)


def _validate_digest(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a built-in lowercase 64-hex string")


def _validate_m0_curve(
    curve: object, *, condition: ObservationCondition, support: tuple[float, float]
) -> float:
    if type(curve) is not ValidationCurve:
        raise TypeError("runner must return exactly ValidationCurve")
    expected_condition = _validate_condition(condition)
    provenance = (
        (curve.episode_id, "MOT17-09-FRCNN", "episode_id"), (curve.source_id, "09", "source_id"),
        (curve.dataset, "MOT17", "dataset"), (curve.split, "validation", "split"),
    )
    for value, expected, name in provenance:
        if type(value) is not str or value != expected:
            raise ValueError(f"curve {name} is not canonical")
    actual_condition = _validate_condition(curve.condition)
    if actual_condition != expected_condition:
        raise ValueError("curve condition does not match requested condition")
    _validate_digest(curve.episode_content_sha256, "episode_content_sha256")
    _validate_digest(curve.action_sha256, "action_sha256")
    _validate_digest(curve.metric_input_sha256, "metric_input_sha256")
    if type(curve.points) is not tuple:
        raise TypeError("curve points must be an exact tuple")
    auc = fixed_support_auc(curve.points, support=support)
    if tuple(point.nominal_rate for point in curve.points) != M0_VALIDATION_RATES:
        raise ValueError("curve nominal rates are not canonical")
    return auc


class _VecEnvLike(Protocol):
    num_envs: int


class _ModelLike(Protocol):
    n_steps: int
    num_timesteps: int
    device: torch.device

    def predict(self, observation: Observation, *, deterministic: bool) -> tuple[object, object]: ...

    def save(self, path: str | Path) -> None: ...

    def get_env(self) -> _VecEnvLike | None: ...


class _CurrentModelPolicy:
    def __init__(self, model: _ModelLike, condition: ObservationCondition) -> None:
        self._model = model
        self._condition = _validate_condition(condition)

    def __call__(self, observation: Observation) -> int:
        model_observation = dict(observation)
        if self._condition == "no-scene-change":
            model_observation["scene_change"] = np.zeros_like(observation["scene_change"])
        action, _ = self._model.predict(model_observation, deterministic=True)
        if type(action) is not np.ndarray or action.shape != () or not np.issubdtype(
            action.dtype, np.integer
        ):
            raise TypeError("model action must be a scalar NumPy integer array")
        value = int(action.item())
        if value not in (0, 1):
            raise ValueError("model action must be 0 or 1")
        return value


class ValidationAucCallback(BaseCallback):
    def __init__(
        self,
        *,
        runner: ValidationRunner,
        support: tuple[float, float],
        recipe: PPORecipe,
        condition: ObservationCondition,
        checkpoint_dir: str | Path,
    ) -> None:
        if type(recipe) is not PPORecipe:
            raise TypeError("recipe must be exactly PPORecipe")
        if recipe != PPORecipe():
            raise ValueError("recipe must equal the complete M0 PPORecipe")
        if not callable(runner):
            raise TypeError("runner must be callable")
        validated_support = _validate_support(support)
        validated_condition = _validate_condition(condition)
        path = Path(checkpoint_dir)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("checkpoint_dir must be an existing nonsymlink directory")
        if any(path.iterdir()):
            raise ValueError("checkpoint_dir must be empty")
        super().__init__(verbose=0)
        self._runner = runner
        self._support = validated_support
        self._condition = validated_condition
        self._checkpoint_dir = path
        self._history: list[ValidationRecord] = []
        requests = range(recipe.validation_interval, recipe.training_steps + 1, recipe.validation_interval)
        milestones = tuple(
            (requested, ((requested + recipe.rollout_steps - 1) // recipe.rollout_steps) * recipe.rollout_steps)
            for requested in requests
        )
        if milestones != _M0_MILESTONES:
            raise RuntimeError("derived validation milestones do not match M0")
        self._milestones = milestones
        self._started = self._closed = False
        self._next_rollout_start = self._next_candidate = 0
        self._best_auc: float | None = None

    @property
    def history(self) -> tuple[ValidationRecord, ...]:
        return tuple(self._history)

    def _model(self) -> _ModelLike:
        raw_model = cast(object, self.model)
        methods = ("predict", "save", "get_env")
        attributes = ("n_steps", "num_timesteps", "device")
        if any(not callable(getattr(raw_model, name, None)) for name in methods) or any(
            not hasattr(raw_model, name) for name in attributes
        ):
            raise TypeError("model does not provide the required callback surface")
        return cast(_ModelLike, raw_model)

    def _require_open(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("validation callback lifecycle is not open")

    def _model_timestep(self) -> int:
        timestep = self._model().num_timesteps
        if type(timestep) is not int:
            raise TypeError("model num_timesteps must be a built-in integer timestep")
        return timestep

    def _on_training_start(self) -> None:
        if self._started or self._closed or self._history:
            raise RuntimeError("validation callback lifecycle cannot be reused")
        total_timesteps: object = self.locals.get("total_timesteps")
        reset_num_timesteps: object = self.locals.get("reset_num_timesteps")
        if type(total_timesteps) is not int or total_timesteps != 500_000:
            raise ValueError("total_timesteps must be built-in integer 500000")
        if reset_num_timesteps is not True:
            raise ValueError("reset_num_timesteps must be exactly True")
        model = self._model()
        if type(model.n_steps) is not int or model.n_steps != 2_048:
            raise ValueError("model n_steps must be built-in integer 2048")
        if type(model.device) is not torch.device or model.device != torch.device("cpu"):
            raise ValueError("model device must be exactly torch.device('cpu')")
        environment = model.get_env()
        num_envs: object = getattr(environment, "num_envs", None)
        if type(num_envs) is not int or num_envs != 1:
            raise ValueError("model environment count must be built-in integer 1")
        if self._model_timestep() != 0:
            raise ValueError("model num_timesteps must be built-in integer 0")
        self._started = True

    def _on_step(self) -> bool:
        self._require_open()
        return True

    def _on_rollout_end(self) -> None:
        self._require_open()

    def _save_selected(self, model: _ModelLike) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".best-", suffix=".zip", dir=self._checkpoint_dir)
        temporary = Path(temporary_name)
        try:
            os.close(descriptor)
            model.save(temporary)
            if temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("model save did not produce a nonempty regular file")
            os.replace(temporary, self._checkpoint_dir / "best.zip")
        finally:
            if temporary.is_symlink() or temporary.is_file():
                temporary.unlink()
            elif temporary.is_dir():
                shutil.rmtree(temporary)
            elif os.path.lexists(temporary):
                temporary.unlink()

    def _run_candidate(self) -> None:
        requested, trained_through = self._milestones[self._next_candidate]
        model = self._model()
        curve = self._runner(
            policy=_CurrentModelPolicy(model, self._condition),
            requested_interactions=requested,
            trained_through_interactions=trained_through,
        )
        auc = _validate_m0_curve(curve, condition=self._condition, support=self._support)
        selected = self._best_auc is None or auc > self._best_auc
        if selected:
            self._save_selected(model)
            self._best_auc = auc
        self._history.append(ValidationRecord(requested, trained_through, auc, selected, curve))

    def _on_rollout_start(self) -> None:
        self._require_open()
        timestep = self._model_timestep()
        if self._next_rollout_start >= self._milestones[-1][1] or (
            timestep != self._next_rollout_start
        ):
            raise ValueError("model timestep is not the next exact rollout start")
        if self._next_candidate < len(self._milestones) - 1 and (
            timestep == self._milestones[self._next_candidate][1]
        ):
            self._run_candidate()
            self._next_candidate += 1
        self._next_rollout_start += 2_048

    def _on_training_end(self) -> None:
        self._require_open()
        timestep = self._model_timestep()
        final_timestep = self._milestones[-1][1]
        if timestep != final_timestep or self._next_rollout_start != final_timestep or (
            self._next_candidate != len(self._milestones) - 1
        ):
            raise ValueError("model timestep does not match the exact final boundary")
        self._run_candidate()
        self._next_candidate += 1
        self._closed = True
