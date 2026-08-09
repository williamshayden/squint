from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import TypeAlias, cast

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]

from .tracker import ObservationScales

JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)

_ROOT_KEYS = {
    "schema_version",
    "episodes",
    "output_dir",
    "budget_rates",
    "seed",
    "observation_scales",
    "tracker",
    "policies",
}
_SCALE_KEYS = {"active_tracks", "age_s", "motion_px_s", "time_since_detector_s"}
_TRACKER_KEYS = {"factory", "parameters"}
_POLICY_KEYS = {"id", "factory", "parameters"}
_EMPTY_PARAMETERS: Mapping[str, JsonValue] = MappingProxyType({})


class ConfigurationError(ValueError):
    """Raised when a benchmark TOML file or factory path is invalid."""


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field} must be a TOML table")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{field} must use string keys")
    return cast(dict[str, object], value)


def _strict_keys(value: dict[str, object], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"{field}.{unknown[0]} is not supported")


def _required(value: dict[str, object], key: str, field: str) -> object:
    if key not in value:
        raise ConfigurationError(f"{field}.{key} is required")
    return value[key]


def _string(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        requirement = "a nonempty string" if nonempty else "a string"
        raise ConfigurationError(f"{field} must be {requirement}")
    return value


def _number(value: object, field: str) -> float:
    if type(value) is int:
        try:
            numeric = float(value)
        except OverflowError as exc:
            raise ConfigurationError(f"{field} must be a finite real number") from exc
    elif type(value) is float:
        numeric = value
    else:
        raise ConfigurationError(f"{field} must be an integer or finite real number")
    if not math.isfinite(numeric):
        raise ConfigurationError(f"{field} must be finite")
    return numeric


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ConfigurationError(f"{field} must be an integer")
    return value


def _config_path(value: object, field: str, base: Path) -> Path:
    path_text = _string(value, field)
    if Path(path_text).is_absolute() or PureWindowsPath(path_text).root:
        raise ConfigurationError(f"{field} must be relative to the TOML file")
    return (base / path_text).resolve()


def _freeze_json(value: object, field: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        numeric = value
        if not math.isfinite(numeric):
            raise ConfigurationError(f"{field} must contain only finite numbers")
        return numeric
    if isinstance(value, list):
        return tuple(_freeze_json(item, f"{field}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{field} must use string object keys")
            frozen[key] = _freeze_json(item, f"{field}.{key}")
        return cast(JsonValue, MappingProxyType(frozen))
    raise ConfigurationError(f"{field} must contain only JSON-compatible values")


def _parameters(value: dict[str, object], field: str) -> Mapping[str, JsonValue]:
    raw = value.get("parameters", _EMPTY_PARAMETERS)
    if raw is _EMPTY_PARAMETERS:
        return _EMPTY_PARAMETERS
    raw_mapping = _mapping(raw, f"{field}.parameters")
    frozen = _freeze_json(raw_mapping, f"{field}.parameters")
    return cast(Mapping[str, JsonValue], frozen)


@dataclass(frozen=True, slots=True)
class TrackerSpec:
    factory: str
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    identifier: str
    factory: str
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    source_path: Path
    episodes: tuple[Path, ...]
    output_dir: Path
    budget_rates: tuple[float, ...]
    seed: int
    observation_scales: ObservationScales
    tracker: TrackerSpec
    policies: tuple[PolicySpec, ...]

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        try:
            source = Path(path).resolve()
            with source.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"config could not be loaded from {path}") from exc

        root = _mapping(raw, "config")
        _strict_keys(root, _ROOT_KEYS, "config")
        schema_version = _integer(_required(root, "schema_version", "config"), "schema_version")
        if schema_version != 1:
            raise ConfigurationError("schema_version must equal integer 1")

        base = source.parent
        raw_episodes = _required(root, "episodes", "config")
        if not isinstance(raw_episodes, list):
            raise ConfigurationError("episodes must be an array")
        if not raw_episodes:
            raise ConfigurationError("episodes must be nonempty")
        episodes = tuple(
            _config_path(item, f"episodes[{index}]", base)
            for index, item in enumerate(raw_episodes)
        )
        seen_episodes: dict[Path, int] = {}
        for index, episode in enumerate(episodes):
            previous = seen_episodes.get(episode)
            if previous is not None:
                raise ConfigurationError(
                    f"episodes[{index}] duplicates episodes[{previous}] after resolution: {episode}"
                )
            seen_episodes[episode] = index
        missing = next(
            ((index, episode) for index, episode in enumerate(episodes) if not episode.is_dir()),
            None,
        )
        if missing is not None:
            index, episode = missing
            raise ConfigurationError(f"episodes[{index}] path does not exist: {episode}")

        output_dir = _config_path(_required(root, "output_dir", "config"), "output_dir", base)

        raw_rates = _required(root, "budget_rates", "config")
        if not isinstance(raw_rates, list) or not raw_rates:
            raise ConfigurationError("budget_rates must be a nonempty array")
        rates = tuple(_number(item, f"budget_rates[{index}]") for index, item in enumerate(raw_rates))
        if any(rate <= 0 or rate > 1 for rate in rates):
            raise ConfigurationError("budget_rates must be in (0, 1]")
        if rates != tuple(sorted(rates)):
            raise ConfigurationError("budget_rates must be sorted in ascending order")
        if len(set(rates)) != len(rates):
            raise ConfigurationError("budget_rates must be unique")

        seed = _integer(_required(root, "seed", "config"), "seed")

        scales_raw = _mapping(_required(root, "observation_scales", "config"), "observation_scales")
        _strict_keys(scales_raw, _SCALE_KEYS, "observation_scales")
        scales_values = {
            name: _number(_required(scales_raw, name, "observation_scales"), f"observation_scales.{name}")
            for name in _SCALE_KEYS
        }
        if any(value <= 0 for value in scales_values.values()):
            invalid = next(name for name, value in scales_values.items() if value <= 0)
            raise ConfigurationError(f"observation_scales.{invalid} must be positive")
        try:
            observation_scales = ObservationScales(**scales_values)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("observation_scales contains invalid values") from exc

        tracker_raw = _mapping(_required(root, "tracker", "config"), "tracker")
        _strict_keys(tracker_raw, _TRACKER_KEYS, "tracker")
        tracker_factory = _string(_required(tracker_raw, "factory", "tracker"), "tracker.factory")
        try:
            load_factory(tracker_factory)
        except ConfigurationError as exc:
            raise ConfigurationError(f"tracker.factory is invalid: {exc}") from exc
        tracker = TrackerSpec(tracker_factory, _parameters(tracker_raw, "tracker"))

        raw_policies = _required(root, "policies", "config")
        if not isinstance(raw_policies, list):
            raise ConfigurationError("policies must be an array")
        if not raw_policies:
            raise ConfigurationError("policies must be nonempty")
        policies: list[PolicySpec] = []
        identifiers: set[str] = set()
        for index, raw_policy in enumerate(raw_policies):
            field = f"policies[{index}]"
            policy_raw = _mapping(raw_policy, field)
            _strict_keys(policy_raw, _POLICY_KEYS, field)
            identifier = _string(_required(policy_raw, "id", field), f"{field}.id")
            if identifier in identifiers:
                raise ConfigurationError(f"{field}.id must be unique")
            identifiers.add(identifier)
            factory = _string(_required(policy_raw, "factory", field), f"{field}.factory")
            try:
                load_factory(factory)
            except ConfigurationError as exc:
                raise ConfigurationError(f"{field}.factory is invalid: {exc}") from exc
            policies.append(PolicySpec(identifier, factory, _parameters(policy_raw, field)))

        return cls(
            source_path=source,
            episodes=episodes,
            output_dir=output_dir,
            budget_rates=rates,
            seed=seed,
            observation_scales=observation_scales,
            tracker=tracker,
            policies=tuple(policies),
        )

    def with_policy_override(self, factory: str) -> BenchmarkConfig:
        load_factory(factory)
        return replace(self, policies=(PolicySpec("external", factory, _EMPTY_PARAMETERS),))

    def with_output_dir(self, output_dir: str | Path) -> BenchmarkConfig:
        return replace(self, output_dir=Path(output_dir).resolve())


def load_factory(path: str) -> Callable[..., object]:
    format_error = ValueError("factory path must use python:module.path:attribute")
    if not isinstance(path, str):
        raise ConfigurationError("factory path must use python:module.path:attribute") from format_error
    parts = path.split(":")
    if (
        len(parts) != 3
        or parts[0] != "python"
        or not parts[1]
        or not parts[2]
        or not all(part.isidentifier() for part in parts[1].split("."))
        or not parts[2].isidentifier()
    ):
        raise ConfigurationError("factory path must use python:module.path:attribute") from format_error

    module_name, attribute = parts[1], parts[2]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigurationError(f"factory import failed for {path}") from exc
    try:
        factory = getattr(module, attribute)
    except Exception as exc:
        raise ConfigurationError(f"factory attribute failed for {path}") from exc
    if not callable(factory):
        cause = TypeError(f"{path} does not resolve to a callable")
        raise ConfigurationError(f"factory target is not callable: {path}") from cause
    return cast(Callable[..., object], factory)
