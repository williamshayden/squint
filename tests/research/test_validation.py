from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from research.train_ppo import PPORecipe
from research.validation import (
    M0_VALIDATION_RATES,
    ObservationCondition,
    ValidationAucCallback,
    ValidationCurve,
    ValidationPoint,
    ValidationRecord,
    ValidationRunner,
    fixed_support_auc,
)

from research import validation as validation_module

DIGEST = "a" * 64


def _curve_records() -> tuple[ValidationCurve, ValidationRecord]:
    points = tuple(
        ValidationPoint(rate, rate, 0.5) for rate in M0_VALIDATION_RATES
    )
    curve = ValidationCurve(
        episode_id="MOT17-09-FRCNN",
        source_id="09",
        dataset="MOT17",
        split="validation",
        episode_content_sha256=DIGEST,
        condition="full",
        points=points,
        action_sha256=DIGEST,
        metric_input_sha256=DIGEST,
    )
    return curve, ValidationRecord(50_000, 51_200, 0.5, True, curve)


def test_validation_public_surface_imports() -> None:
    assert M0_VALIDATION_RATES == (0.10, 0.25, 0.50, 0.75, 1.00)
    assert ObservationCondition is not None
    assert ValidationAucCallback is not None
    assert ValidationCurve is not None
    assert ValidationPoint is not None
    assert ValidationRecord is not None
    assert ValidationRunner is not None
    assert fixed_support_auc is not None


def test_validation_records_are_frozen_slotted_values() -> None:
    curve, record = _curve_records()

    assert record.curve is curve
    assert record.requested_interactions == 50_000
    assert not hasattr(curve.points[0], "__dict__")
    assert not hasattr(curve, "__dict__")
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.auc = 0.6  # type: ignore[misc]


def test_fixed_support_auc_normalizes_a_linear_curve() -> None:
    points = (
        ValidationPoint(0.25, 0.2, 0.2),
        ValidationPoint(0.75, 0.8, 0.8),
    )

    assert fixed_support_auc(points, support=(0.2, 0.8)) == pytest.approx(0.5)


def test_fixed_support_auc_matches_preregistered_nonlinear_interpolation() -> None:
    points = (
        ValidationPoint(0.10, 0.10, 0.15),
        ValidationPoint(0.50, 0.42, 0.80),
        ValidationPoint(1.00, 0.95, 0.35),
    )
    support = (0.2, 0.8)
    grid = np.linspace(0.2, 0.8, 101)
    expected_y = np.interp(grid, [0.10, 0.42, 0.95], [0.15, 0.80, 0.35])
    expected = float(np.trapezoid(expected_y, grid) / 0.6)

    assert fixed_support_auc(points, support=support) == pytest.approx(expected)


def test_fixed_support_auc_does_not_mutate_or_depend_on_input_order() -> None:
    points = [
        ValidationPoint(1.00, 0.90, 0.30),
        ValidationPoint(0.10, 0.10, 0.20),
        ValidationPoint(0.50, 0.50, 0.80),
    ]
    original = list(points)

    forward = fixed_support_auc(points, support=(0.2, 0.8))
    reverse = fixed_support_auc(tuple(reversed(points)), support=(0.2, 0.8))

    assert forward == pytest.approx(reverse)
    assert points == original


@pytest.mark.parametrize(
    "support",
    [
        [0.0, 1.0],
        (0.0,),
        (0.0, 0.5, 1.0),
        (False, 1.0),
        (0.0, True),
        (np.float64(0.0), 1.0),
        (0.0, np.int64(1)),
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (0.5, 0.5),
        (0.8, 0.2),
        (-0.1, 0.8),
        (0.2, 1.1),
    ],
)
def test_fixed_support_auc_rejects_noncanonical_support(support: Any) -> None:
    points = (ValidationPoint(0.1, 0.0, 0.2), ValidationPoint(1.0, 1.0, 0.8))

    with pytest.raises((TypeError, ValueError)):
        fixed_support_auc(points, support=support)


class _ValidationPointSubclass(ValidationPoint):
    pass


@pytest.mark.parametrize(
    "points",
    [
        (),
        (ValidationPoint(0.1, 0.0, 0.2),),
        (
            _ValidationPointSubclass(0.1, 0.0, 0.2),
            ValidationPoint(1.0, 1.0, 0.8),
        ),
        (object(), ValidationPoint(1.0, 1.0, 0.8)),
        (ValidationPoint(True, 0.0, 0.2), ValidationPoint(1.0, 1.0, 0.8)),
        (
            ValidationPoint(np.float64(0.1), 0.0, 0.2),
            ValidationPoint(1.0, 1.0, 0.8),
        ),
        (ValidationPoint(0.0, 0.0, 0.2), ValidationPoint(1.0, 1.0, 0.8)),
        (ValidationPoint(1.1, 0.0, 0.2), ValidationPoint(1.0, 1.0, 0.8)),
        (
            ValidationPoint(0.1, float("nan"), 0.2),
            ValidationPoint(1.0, 1.0, 0.8),
        ),
        (ValidationPoint(0.1, -0.1, 0.2), ValidationPoint(1.0, 1.0, 0.8)),
        (ValidationPoint(0.1, 0.0, 1.1), ValidationPoint(1.0, 1.0, 0.8)),
        (
            ValidationPoint(0.1, 0.0, float("inf")),
            ValidationPoint(1.0, 1.0, 0.8),
        ),
    ],
)
def test_fixed_support_auc_rejects_malformed_points(points: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        fixed_support_auc(points, support=(0.0, 1.0))


def test_fixed_support_auc_rejects_duplicate_realized_compute() -> None:
    points = (
        ValidationPoint(0.1, 0.2, 0.3),
        ValidationPoint(1.0, 0.2, 0.7),
    )

    with pytest.raises(ValueError, match="duplicate"):
        fixed_support_auc(points, support=(0.2, 0.8))


@pytest.mark.parametrize(
    "points",
    [
        (ValidationPoint(0.1, 0.21, 0.2), ValidationPoint(1.0, 0.9, 0.8)),
        (ValidationPoint(0.1, 0.1, 0.2), ValidationPoint(1.0, 0.79, 0.8)),
    ],
)
def test_fixed_support_auc_rejects_incomplete_measured_support(
    points: tuple[ValidationPoint, ...],
) -> None:
    with pytest.raises(ValueError, match="cover"):
        fixed_support_auc(points, support=(0.2, 0.8))


@pytest.mark.parametrize("bad_result", [float("nan"), float("inf"), -0.1, 1.1])
def test_fixed_support_auc_rejects_invalid_normalized_result(
    monkeypatch: pytest.MonkeyPatch, bad_result: float
) -> None:
    monkeypatch.setattr("research.validation.np.trapezoid", lambda *_args: bad_result)
    points = (ValidationPoint(0.1, 0.0, 0.2), ValidationPoint(1.0, 1.0, 0.8))

    with pytest.raises(ValueError, match="normalized"):
        fixed_support_auc(points, support=(0.0, 1.0))


class _StringSubclass(str):
    pass


class _ValidationCurveSubclass(ValidationCurve):
    pass


def _assert_curve_rejected(curve: object, *, condition: Any = "full") -> None:
    with pytest.raises((TypeError, ValueError)):
        validation_module._validate_m0_curve(
            curve, condition=condition, support=(0.10, 1.00)
        )


def test_validate_m0_curve_accepts_only_exact_curve_type() -> None:
    curve, _ = _curve_records()
    subclass = _ValidationCurveSubclass(
        curve.episode_id,
        curve.source_id,
        curve.dataset,
        curve.split,
        curve.episode_content_sha256,
        curve.condition,
        curve.points,
        curve.action_sha256,
        curve.metric_input_sha256,
    )

    _assert_curve_rejected(object())
    _assert_curve_rejected(subclass)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "MOT17-10-FRCNN"),
        ("episode_id", _StringSubclass("MOT17-09-FRCNN")),
        ("source_id", "10"),
        ("source_id", _StringSubclass("09")),
        ("dataset", "MOT20"),
        ("dataset", _StringSubclass("MOT17")),
        ("split", "train"),
        ("split", _StringSubclass("validation")),
    ],
)
def test_validate_m0_curve_rejects_noncanonical_provenance(
    field: str, value: object
) -> None:
    curve, _ = _curve_records()

    _assert_curve_rejected(replace(curve, **{field: value}))


@pytest.mark.parametrize(
    ("curve_condition", "expected_condition"),
    [
        ("no-scene-change", "full"),
        ("other", "full"),
        (_StringSubclass("full"), "full"),
        ("full", "other"),
        ("full", _StringSubclass("full")),
    ],
)
def test_validate_m0_curve_rejects_condition_mismatch_or_invalid_condition(
    curve_condition: object, expected_condition: object
) -> None:
    curve, _ = _curve_records()

    _assert_curve_rejected(
        replace(curve, condition=curve_condition), condition=expected_condition
    )


@pytest.mark.parametrize(
    "field",
    ["episode_content_sha256", "action_sha256", "metric_input_sha256"],
)
@pytest.mark.parametrize(
    "value",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        _StringSubclass("a" * 64),
        7,
    ],
)
def test_validate_m0_curve_rejects_malformed_digest(
    field: str, value: object
) -> None:
    curve, _ = _curve_records()

    _assert_curve_rejected(replace(curve, **{field: value}))


def test_validate_m0_curve_requires_an_exact_points_tuple() -> None:
    curve, _ = _curve_records()

    _assert_curve_rejected(replace(curve, points=list(curve.points)))


def test_validate_m0_curve_requires_exact_point_types() -> None:
    curve, _ = _curve_records()
    subclass = _ValidationPointSubclass(0.10, 0.10, 0.5)

    _assert_curve_rejected(replace(curve, points=(subclass, *curve.points[1:])))
    _assert_curve_rejected(replace(curve, points=(object(), *curve.points[1:])))


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "reordered", "extra", "altered", "numpy"])
def test_validate_m0_curve_requires_exact_nominal_rate_sequence(mutation: str) -> None:
    curve, _ = _curve_records()
    points = curve.points
    if mutation == "omitted":
        mutated = points[:-1]
    elif mutation == "duplicate":
        mutated = (points[0], replace(points[1], nominal_rate=0.10), *points[2:])
    elif mutation == "reordered":
        mutated = (points[1], points[0], *points[2:])
    elif mutation == "extra":
        mutated = (*points, ValidationPoint(1.0, 0.9, 0.5))
    elif mutation == "altered":
        mutated = (replace(points[0], nominal_rate=0.11), *points[1:])
    else:
        mutated = (replace(points[0], nominal_rate=np.float64(0.10)), *points[1:])

    _assert_curve_rejected(replace(curve, points=mutated))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nominal_rate", float("nan")),
        ("nominal_rate", 0.0),
        ("realized_compute", float("inf")),
        ("realized_compute", -0.1),
        ("realized_compute", 1.1),
        ("hota", float("nan")),
        ("hota", -0.1),
        ("hota", 1.1),
    ],
)
def test_validate_m0_curve_rejects_invalid_point_values(
    field: str, value: object
) -> None:
    curve, _ = _curve_records()
    points = (replace(curve.points[0], **{field: value}), *curve.points[1:])

    _assert_curve_rejected(replace(curve, points=points))


def test_validate_m0_curve_rejects_duplicate_compute_and_incomplete_support() -> None:
    curve, _ = _curve_records()

    duplicate = (
        curve.points[0],
        replace(curve.points[1], realized_compute=curve.points[0].realized_compute),
        *curve.points[2:],
    )
    _assert_curve_rejected(replace(curve, points=duplicate))
    _assert_curve_rejected(
        replace(
            curve,
            points=(replace(curve.points[0], realized_compute=0.11), *curve.points[1:]),
        )
    )
    _assert_curve_rejected(
        replace(
            curve,
            points=(*curve.points[:-1], replace(curve.points[-1], realized_compute=0.99)),
        )
    )


@pytest.mark.parametrize("condition", ["full", "no-scene-change"])
def test_validate_m0_curve_returns_deterministic_auc(condition: Any) -> None:
    curve, _ = _curve_records()
    curve = replace(curve, condition=condition)

    assert validation_module._validate_m0_curve(
        curve, condition=condition, support=(0.10, 1.00)
    ) == pytest.approx(0.5)


class _UnusedRunner:
    def __call__(self, **_kwargs: object) -> ValidationCurve:
        curve, _ = _curve_records()
        return curve


def _empty_checkpoint_dir(path: Path) -> Path:
    path.mkdir()
    return path


def _new_callback(
    checkpoint_dir: Path,
    *,
    runner: object = _UnusedRunner(),
    support: object = (0.10, 1.00),
    recipe: object = PPORecipe(),
    condition: object = "full",
) -> ValidationAucCallback:
    return ValidationAucCallback(
        runner=runner,
        support=support,
        recipe=recipe,
        condition=condition,
        checkpoint_dir=checkpoint_dir,
    )


def test_callback_accepts_only_the_complete_frozen_recipe_before_base_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_init_calls: list[int] = []

    def recording_base_init(_self: object, verbose: int = 0) -> None:
        base_init_calls.append(verbose)

    monkeypatch.setattr(
        validation_module.BaseCallback, "__init__", recording_base_init
    )
    changes = {
        "learning_rate": 1e-3,
        "rollout_steps": 1024,
        "batch_size": 32,
        "gamma": 0.98,
        "gae_lambda": 0.90,
        "clip_range": 0.10,
        "training_steps": 400_000,
        "validation_interval": 25_000,
        "seeds": (9,),
        "net_arch": (32, 32),
    }
    assert set(changes) == {field.name for field in fields(PPORecipe)}
    for name, value in changes.items():
        checkpoint_dir = _empty_checkpoint_dir(tmp_path / name)
        with pytest.raises((TypeError, ValueError), match="recipe"):
            _new_callback(
                checkpoint_dir,
                recipe=replace(PPORecipe(), **{name: value}),
            )
    wrong_type_dir = _empty_checkpoint_dir(tmp_path / "wrong-type")
    with pytest.raises(TypeError, match="recipe"):
        _new_callback(wrong_type_dir, recipe=object())

    assert base_init_calls == []


def test_callback_constructor_accepts_valid_inputs_and_starts_with_no_history(
    tmp_path: Path,
) -> None:
    callback = _new_callback(_empty_checkpoint_dir(tmp_path / "checkpoints"))

    assert callback.verbose == 0
    assert callback.history == ()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("runner", object()),
        ("support", [0.10, 1.00]),
        ("support", (np.float64(0.10), 1.00)),
        ("support", (0.50, 0.50)),
        ("condition", "other"),
        ("condition", _StringSubclass("full")),
    ],
)
def test_callback_constructor_rejects_invalid_boundaries(
    tmp_path: Path, argument: str, value: object
) -> None:
    checkpoint_dir = _empty_checkpoint_dir(tmp_path / argument)

    with pytest.raises((TypeError, ValueError)):
        _new_callback(checkpoint_dir, **{argument: value})


def test_callback_constructor_requires_existing_nonsymlink_empty_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="checkpoint"):
        _new_callback(missing)

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        _new_callback(regular_file)

    target = _empty_checkpoint_dir(tmp_path / "target")
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="checkpoint"):
        _new_callback(symlink)

    nonempty = _empty_checkpoint_dir(tmp_path / "nonempty")
    (nonempty / "entry").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _new_callback(nonempty)

    with_best = _empty_checkpoint_dir(tmp_path / "with-best")
    (with_best / "best.zip").write_bytes(b"old")
    with pytest.raises(ValueError, match="empty"):
        _new_callback(with_best)


class _PolicyModel:
    def __init__(self, action: object) -> None:
        self.action = action
        self.observations: list[dict[str, np.ndarray[Any, Any]]] = []
        self.deterministic: list[bool] = []

    def predict(
        self, observation: dict[str, np.ndarray[Any, Any]], *, deterministic: bool
    ) -> tuple[object, None]:
        self.observations.append(observation)
        self.deterministic.append(deterministic)
        return self.action, None


def _observation() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "tracker_state": np.array([0.1, 0.2], dtype=np.float32),
        "scene_change": np.array([0.3, 0.4], dtype=np.float32),
        "compute_budget": np.array([0.5, 0.6], dtype=np.float32),
    }


@pytest.mark.parametrize(
    "action",
    [
        np.array(0, dtype=np.int8),
        np.array(1, dtype=np.int64),
        np.array(1, dtype=np.uint64),
    ],
)
def test_current_model_policy_accepts_only_scalar_numpy_integer_actions(
    action: np.ndarray[Any, Any],
) -> None:
    model = _PolicyModel(action)
    policy = validation_module._CurrentModelPolicy(model, "full")

    assert policy(_observation()) == int(action)
    assert model.deterministic == [True]


@pytest.mark.parametrize(
    "action",
    [
        1,
        np.int64(1),
        True,
        1.0,
        np.array([1], dtype=np.int64),
        np.array(1.0),
        np.array(True),
        np.array(1, dtype=object),
        np.array(-1, dtype=np.int64),
        np.array(2, dtype=np.int64),
    ],
)
def test_current_model_policy_rejects_malformed_actions(action: object) -> None:
    policy = validation_module._CurrentModelPolicy(_PolicyModel(action), "full")

    with pytest.raises((TypeError, ValueError), match="action"):
        policy(_observation())


def test_current_model_policy_copies_outer_mapping_and_preserves_full_aliases() -> None:
    model = _PolicyModel(np.array(1, dtype=np.int64))
    policy = validation_module._CurrentModelPolicy(model, "full")
    observation = _observation()

    assert policy(observation) == 1
    assert policy(observation) == 1

    first, second = model.observations
    assert first is not observation
    assert second is not observation
    assert second is not first
    assert all(first[name] is value for name, value in observation.items())
    assert all(second[name] is value for name, value in observation.items())


def test_current_model_policy_masks_only_scene_change_without_mutating_input() -> None:
    model = _PolicyModel(np.array(0, dtype=np.int64))
    policy = validation_module._CurrentModelPolicy(model, "no-scene-change")
    observation = _observation()
    original_scene = observation["scene_change"].copy()

    assert policy(observation) == 0

    masked = model.observations[0]
    assert np.array_equal(observation["scene_change"], original_scene)
    assert np.array_equal(masked["scene_change"], np.zeros_like(original_scene))
    assert masked["scene_change"] is not observation["scene_change"]
    assert masked["tracker_state"] is observation["tracker_state"]
    assert masked["compute_budget"] is observation["compute_budget"]


class _FakeVecEnv:
    def __init__(self, num_envs: object = 1) -> None:
        self.num_envs = num_envs


class _FakeModel:
    def __init__(self) -> None:
        self.num_timesteps: object = 0
        self.n_steps: object = 2048
        self.device: object = torch.device("cpu")
        self.env: object = _FakeVecEnv()
        self.action: object = np.array(1, dtype=np.int64)
        self.save_paths: list[Path] = []

    def get_env(self) -> object:
        return self.env

    def predict(
        self, _observation: object, *, deterministic: bool
    ) -> tuple[object, None]:
        assert deterministic is True
        return self.action, None

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        self.save_paths.append(destination)
        destination.write_bytes(b"checkpoint")


class _DirectorySavingModel(_FakeModel):
    def save(self, path: str | Path) -> None:
        destination = Path(path)
        self.save_paths.append(destination)
        destination.unlink()
        destination.mkdir()
        (destination / "payload").write_bytes(b"not a checkpoint")


class _FifoSavingModel(_FakeModel):
    def save(self, path: str | Path) -> None:
        destination = Path(path)
        self.save_paths.append(destination)
        destination.unlink()
        os.mkfifo(destination)


class _RecordingRunner:
    def __init__(self, hotas: tuple[float, ...] = ()) -> None:
        self.calls: list[tuple[object, int, int]] = []
        self.hotas = hotas

    def __call__(
        self,
        *,
        policy: object,
        requested_interactions: int,
        trained_through_interactions: int,
    ) -> ValidationCurve:
        self.calls.append(
            (policy, requested_interactions, trained_through_interactions)
        )
        curve, _ = _curve_records()
        index = len(self.calls) - 1
        hota = self.hotas[index] if self.hotas else 0.5
        return replace(
            curve,
            points=tuple(replace(point, hota=hota) for point in curve.points),
        )


def _bound_callback(
    tmp_path: Path,
    *,
    runner: object | None = None,
    model: _FakeModel | None = None,
) -> tuple[ValidationAucCallback, _FakeModel, _RecordingRunner]:
    actual_runner = _RecordingRunner() if runner is None else runner
    recording_runner = cast(_RecordingRunner, actual_runner)
    actual_model = _FakeModel() if model is None else model
    callback = _new_callback(
        _empty_checkpoint_dir(tmp_path / "checkpoints"), runner=actual_runner
    )
    callback.init_callback(cast(Any, actual_model))
    return callback, actual_model, recording_runner


def _start(callback: ValidationAucCallback) -> None:
    callback.on_training_start(
        {"total_timesteps": 500_000, "reset_num_timesteps": True}, {}
    )


@pytest.mark.parametrize(
    ("local_name", "value"),
    [
        ("total_timesteps", 499_999),
        ("total_timesteps", np.int64(500_000)),
        ("total_timesteps", True),
        ("reset_num_timesteps", False),
        ("reset_num_timesteps", np.bool_(True)),
        ("reset_num_timesteps", 1),
    ],
)
def test_training_start_requires_exact_learn_locals(
    tmp_path: Path, local_name: str, value: object
) -> None:
    callback, _model, _runner = _bound_callback(tmp_path)
    local_values: dict[str, object] = {
        "total_timesteps": 500_000,
        "reset_num_timesteps": True,
    }
    local_values[local_name] = value

    with pytest.raises((TypeError, ValueError), match=local_name):
        callback.on_training_start(local_values, {})


@pytest.mark.parametrize("missing", ["total_timesteps", "reset_num_timesteps"])
def test_training_start_rejects_missing_learn_locals(
    tmp_path: Path, missing: str
) -> None:
    callback, _model, _runner = _bound_callback(tmp_path)
    local_values = {
        "total_timesteps": 500_000,
        "reset_num_timesteps": True,
    }
    del local_values[missing]

    with pytest.raises((TypeError, ValueError), match=missing):
        callback.on_training_start(local_values, {})


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("num_timesteps", 1),
        ("num_timesteps", np.int64(0)),
        ("num_timesteps", False),
        ("n_steps", 1024),
        ("n_steps", np.int64(2048)),
        ("n_steps", True),
        ("device", "cpu"),
        ("device", torch.device("meta")),
    ],
)
def test_training_start_requires_exact_model_state(
    tmp_path: Path, attribute: str, value: object
) -> None:
    model = _FakeModel()
    setattr(model, attribute, value)
    callback, _model, _runner = _bound_callback(tmp_path, model=model)

    with pytest.raises((TypeError, ValueError), match=attribute):
        _start(callback)


@pytest.mark.parametrize("env", [None, object(), _FakeVecEnv(2), _FakeVecEnv(np.int64(1))])
def test_training_start_requires_one_exact_environment(
    tmp_path: Path, env: object
) -> None:
    model = _FakeModel()
    model.env = env
    callback, _model, _runner = _bound_callback(tmp_path, model=model)

    with pytest.raises((TypeError, ValueError), match="environment"):
        _start(callback)


def test_training_start_is_single_use(tmp_path: Path) -> None:
    callback, _model, _runner = _bound_callback(tmp_path)
    _start(callback)

    with pytest.raises(RuntimeError, match="lifecycle"):
        _start(callback)


@pytest.mark.parametrize(
    "hook_name", ["on_step", "on_rollout_start", "on_rollout_end", "on_training_end"]
)
def test_hooks_reject_before_training_start(tmp_path: Path, hook_name: str) -> None:
    callback, _model, _runner = _bound_callback(tmp_path)

    with pytest.raises(RuntimeError, match="lifecycle"):
        getattr(callback, hook_name)()


def test_step_and_rollout_end_are_validation_free_while_open(tmp_path: Path) -> None:
    callback, model, runner = _bound_callback(tmp_path)
    _start(callback)
    model.num_timesteps = 17

    assert callback.on_step() is True
    assert callback.on_rollout_end() is None
    assert runner.calls == []


def test_rollout_starts_follow_exact_boundaries_before_first_candidate(
    tmp_path: Path,
) -> None:
    callback, model, runner = _bound_callback(tmp_path)
    _start(callback)

    for timestep in range(0, 51_200, 2_048):
        model.num_timesteps = timestep
        callback.on_rollout_start()

    model.num_timesteps = 49_152
    with pytest.raises(ValueError, match="timestep"):
        callback.on_rollout_start()
    assert runner.calls == []


@pytest.mark.parametrize("bad_timestep", [True, np.int64(0), -1, 1, 2048])
def test_rollout_start_rejects_nonexact_first_boundary_without_advancing(
    tmp_path: Path, bad_timestep: object
) -> None:
    callback, model, runner = _bound_callback(tmp_path)
    _start(callback)
    model.num_timesteps = bad_timestep

    with pytest.raises((TypeError, ValueError), match="timestep"):
        callback.on_rollout_start()

    model.num_timesteps = 0
    callback.on_rollout_start()
    assert runner.calls == []


def test_rollout_start_rejects_repeated_skipped_and_early_boundaries(
    tmp_path: Path,
) -> None:
    callback, model, _runner = _bound_callback(tmp_path)
    _start(callback)
    model.num_timesteps = 0
    callback.on_rollout_start()

    for bad_timestep in (0, 4_096, 1_024):
        model.num_timesteps = bad_timestep
        with pytest.raises(ValueError, match="timestep"):
            callback.on_rollout_start()

    model.num_timesteps = 2_048
    callback.on_rollout_start()


@pytest.mark.parametrize(
    "bad_timestep", [True, np.int64(501_760), 499_712, 501_760, 503_808]
)
def test_training_end_rejects_nonfinal_or_nonexact_boundary(
    tmp_path: Path, bad_timestep: object
) -> None:
    callback, model, runner = _bound_callback(tmp_path)
    _start(callback)
    model.num_timesteps = bad_timestep

    with pytest.raises((TypeError, ValueError), match="timestep"):
        callback.on_training_end()

    assert runner.calls == []


def test_directory_save_output_preserves_validation_error_and_cleans_temp(
    tmp_path: Path,
) -> None:
    model = _DirectorySavingModel()
    callback, model, _runner = _bound_callback(tmp_path, model=model)
    _start(callback)
    for timestep in range(0, 51_200, 2_048):
        model.num_timesteps = timestep
        callback.on_rollout_start()

    model.num_timesteps = 51_200
    with pytest.raises(
        RuntimeError, match="model save did not produce a nonempty regular file"
    ):
        callback.on_rollout_start()

    assert tuple((tmp_path / "checkpoints").iterdir()) == ()
    assert callback.history == ()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo is unavailable")
def test_fifo_save_output_preserves_validation_error_and_cleans_temp(
    tmp_path: Path,
) -> None:
    model = _FifoSavingModel()
    callback, model, _runner = _bound_callback(tmp_path, model=model)
    _start(callback)
    for timestep in range(0, 51_200, 2_048):
        model.num_timesteps = timestep
        callback.on_rollout_start()

    model.num_timesteps = 51_200
    with pytest.raises(
        RuntimeError, match="model save did not produce a nonempty regular file"
    ):
        callback.on_rollout_start()

    assert tuple((tmp_path / "checkpoints").iterdir()) == ()
    assert callback.history == ()


def test_complete_lifecycle_validates_exact_milestones_and_closes(
    tmp_path: Path,
) -> None:
    hotas = (0.4, 0.6, 0.6, 0.5, 0.55, 0.3, 0.2, 0.1, 0.0, 0.59)
    runner = _RecordingRunner(hotas)
    callback, model, _runner = _bound_callback(tmp_path, runner=runner)
    _start(callback)

    for timestep in range(0, 501_760, 2_048):
        model.num_timesteps = timestep
        callback.on_rollout_start()

    assert len(runner.calls) == 9
    model.num_timesteps = 501_760
    with pytest.raises(ValueError, match="timestep"):
        callback.on_rollout_start()
    callback.on_training_end()

    expected_milestones = (
        (50_000, 51_200),
        (100_000, 100_352),
        (150_000, 151_552),
        (200_000, 200_704),
        (250_000, 251_904),
        (300_000, 301_056),
        (350_000, 350_208),
        (400_000, 401_408),
        (450_000, 450_560),
        (500_000, 501_760),
    )
    assert tuple((requested, actual) for _, requested, actual in runner.calls) == (
        expected_milestones
    )
    assert all(callable(policy) for policy, _, _ in runner.calls)
    assert all(policy(_observation()) == 1 for policy, _, _ in runner.calls)
    first_history = callback.history
    second_history = callback.history
    assert first_history == second_history
    assert first_history is not second_history
    assert tuple(record.auc for record in first_history) == pytest.approx(hotas)
    assert tuple(record.selected for record in first_history) == (
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    assert len(model.save_paths) == 2
    assert (tmp_path / "checkpoints" / "best.zip").read_bytes() == b"checkpoint"
    assert not any(path.exists() for path in model.save_paths)

    for hook_name in ("on_step", "on_rollout_start", "on_rollout_end", "on_training_end"):
        with pytest.raises(RuntimeError, match="lifecycle"):
            getattr(callback, hook_name)()
    assert len(runner.calls) == 10
    with pytest.raises(RuntimeError, match="lifecycle"):
        _start(callback)
