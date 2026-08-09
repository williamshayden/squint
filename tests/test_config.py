from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from squint_rl.config import BenchmarkConfig, ConfigurationError, load_factory

_POLICY_BLOCK = """[[policies]]
id = "greedy"
factory = "python:squint_rl.policies:greedy_factory"
parameters = { nested = { values = [1, { enabled = true }] }, label = "policy" }
"""
_VALID_CONFIG = f"""schema_version = 1
episodes = ["episodes/a"]
output_dir = "runs/result"
budget_rates = [0.1, 0.25, 0.5, 0.75, 1.0]
seed = 7

[observation_scales]
active_tracks = 256.0
age_s = 2.0
motion_px_s = 200.0
time_since_detector_s = 2.0

[tracker]
factory = "python:squint_rl.synthetic:make_synthetic_episode"
parameters = {{ nested = {{ values = [1, {{ enabled = true }}] }}, label = "tracker" }}

{_POLICY_BLOCK}"""


def _write_config(tmp_path: Path, text: str = _VALID_CONFIG) -> Path:
    (tmp_path / "episodes" / "a").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "benchmark.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _load_invalid(tmp_path: Path, text: str) -> ConfigurationError:
    with pytest.raises(ConfigurationError) as error:
        BenchmarkConfig.load(_write_config(tmp_path, text))
    return error.value


def test_config_resolves_paths_relative_to_toml_and_allows_parent_segments(tmp_path: Path) -> None:
    config = BenchmarkConfig.load(
        _write_config(
            tmp_path,
            _VALID_CONFIG.replace('output_dir = "runs/result"', 'output_dir = "../runs/result"'),
        )
    )

    assert config.source_path == (tmp_path / "benchmark.toml").resolve()
    assert config.episodes == ((tmp_path / "episodes" / "a").resolve(),)
    assert config.output_dir == (tmp_path / "../runs/result").resolve()
    assert config.policies[0].identifier == "greedy"


def test_loaded_value_objects_and_nested_parameters_are_immutable(tmp_path: Path) -> None:
    config = BenchmarkConfig.load(_write_config(tmp_path))

    assert isinstance(config.tracker.parameters, MappingProxyType)
    tracker_nested = config.tracker.parameters["nested"]
    assert isinstance(tracker_nested, MappingProxyType)
    assert tracker_nested["values"] == (1, MappingProxyType({"enabled": True}))
    with pytest.raises(TypeError):
        config.tracker.parameters["label"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        tracker_nested["values"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        tracker_nested["values"][0] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.seed = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("replacement", "field"),
    [
        ("schema_version = true", "schema_version"),
        ('budget_rates = ["0.1"]', "budget_rates[0]"),
        ("seed = true", "seed"),
        ("active_tracks = true", "observation_scales.active_tracks"),
    ],
)
def test_rejects_bool_or_string_numeric_values(tmp_path: Path, replacement: str, field: str) -> None:
    original = {
        "schema_version": "schema_version = 1",
        "budget_rates[0]": "budget_rates = [0.1, 0.25, 0.5, 0.75, 1.0]",
        "seed": "seed = 7",
        "observation_scales.active_tracks": "active_tracks = 256.0",
    }[field]
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace(original, replacement))
    assert field in str(error)


@pytest.mark.parametrize(
    "replacement",
    [
        "budget_rates = [nan]",
        "budget_rates = [inf]",
        "active_tracks = inf",
        "age_s = nan",
    ],
)
def test_rejects_nonfinite_numeric_values(tmp_path: Path, replacement: str) -> None:
    original = (
        "budget_rates = [0.1, 0.25, 0.5, 0.75, 1.0]"
        if replacement.startswith("budget_rates")
        else replacement.split(" = ", 1)[0] + " = 2.0"
        if replacement.startswith(("age_s", "motion_px_s", "time_since_detector_s"))
        else "active_tracks = 256.0"
    )
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace(original, replacement))
    assert "finite" in str(error)


def test_missing_required_nested_key_has_a_field_path(tmp_path: Path) -> None:
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace("age_s = 2.0\n", ""))
    assert "observation_scales.age_s" in str(error)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (
            _VALID_CONFIG.replace(
                'factory = "python:squint_rl.synthetic:make_synthetic_episode"\n', ""
            ),
            "tracker.factory",
        ),
        (_VALID_CONFIG.replace('id = "greedy"\n', ""), "policies[0].id"),
        (
            _VALID_CONFIG.replace('factory = "python:squint_rl.policies:greedy_factory"\n', ""),
            "policies[0].factory",
        ),
    ],
)
def test_missing_required_tracker_and_policy_fields_have_field_paths(
    tmp_path: Path, text: str, field: str
) -> None:
    error = _load_invalid(tmp_path, text)
    assert field in str(error)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (_VALID_CONFIG.replace("seed = 7\n", "seed = 7\nmystery = true\n"), "config.mystery"),
        (_VALID_CONFIG.replace("time_since_detector_s = 2.0\n", "time_since_detector_s = 2.0\nmystery = true\n"), "observation_scales.mystery"),
        (_VALID_CONFIG.replace("factory = \"python:squint_rl.synthetic:make_synthetic_episode\"\n", "factory = \"python:squint_rl.synthetic:make_synthetic_episode\"\nmystery = true\n"), "tracker.mystery"),
        (_VALID_CONFIG.replace("parameters = { nested = { values = [1, { enabled = true }] }, label = \"policy\" }\n", "parameters = { nested = { values = [1, { enabled = true }] }, label = \"policy\" }\nmystery = true\n"), "policies[0].mystery"),
    ],
)
def test_unknown_keys_are_rejected_at_each_config_level(
    tmp_path: Path, text: str, field: str
) -> None:
    error = _load_invalid(tmp_path, text)
    assert field in str(error)


def test_required_root_keys_are_enforced(tmp_path: Path) -> None:
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace('output_dir = "runs/result"\n', ""))
    assert "config.output_dir" in str(error)


@pytest.mark.parametrize(
    "text",
    [
        _VALID_CONFIG.replace("episodes = [\"episodes/a\"]", "episodes = []"),
        _VALID_CONFIG.replace(_POLICY_BLOCK, "").replace(
            "[observation_scales]", "policies = []\n\n[observation_scales]"
        ),
        _VALID_CONFIG.replace("id = \"greedy\"", "id = \"\""),
        _VALID_CONFIG.replace(
            'factory = "python:squint_rl.policies:greedy_factory"', 'factory = ""'
        ),
        _VALID_CONFIG.replace("episodes = [\"episodes/a\"]", "episodes = [\"\"]"),
    ],
)
def test_collections_and_identifiers_must_be_nonempty(tmp_path: Path, text: str) -> None:
    error = _load_invalid(tmp_path, text)
    assert "nonempty" in str(error) or ".id" in str(error)


@pytest.mark.parametrize(
    "rates",
    [
        "[0.0]",
        "[1.1]",
        "[0.5, 0.1]",
        "[0.1, 0.1]",
    ],
)
def test_budget_rates_are_sorted_unique_and_bounded(tmp_path: Path, rates: str) -> None:
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace("[0.1, 0.25, 0.5, 0.75, 1.0]", rates))
    assert "budget_rates" in str(error)


def test_duplicate_policy_ids_are_rejected(tmp_path: Path) -> None:
    duplicate = _POLICY_BLOCK + "\n" + _POLICY_BLOCK
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace(_POLICY_BLOCK, duplicate))
    assert "policies[1].id" in str(error)


def test_scales_must_be_positive(tmp_path: Path) -> None:
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace("age_s = 2.0", "age_s = 0.0"))
    assert "observation_scales.age_s" in str(error)


def test_schema_version_must_be_exact_integer_one(tmp_path: Path) -> None:
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace("schema_version = 1", "schema_version = 2"))
    assert "schema_version" in str(error)
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace("schema_version = 1", "schema_version = 1.0"))
    assert "schema_version" in str(error)


@pytest.mark.parametrize("field", ["episodes", "output_dir"])
def test_config_paths_must_be_relative(tmp_path: Path, field: str) -> None:
    replacement = (
        'episodes = ["/absolute/episode"]'
        if field == "episodes"
        else 'output_dir = "/absolute/output"'
    )
    original = 'episodes = ["episodes/a"]' if field == "episodes" else 'output_dir = "runs/result"'
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace(original, replacement))
    assert field in str(error)


def test_episode_paths_must_exist_as_directories_but_output_need_not_exist(tmp_path: Path) -> None:
    missing_episode = _VALID_CONFIG.replace("episodes/a", "episodes/missing")
    error = _load_invalid(tmp_path, missing_episode)
    assert "episodes[0] path does not exist" in str(error)
    config = BenchmarkConfig.load(_write_config(tmp_path / "second"))
    assert not config.output_dir.exists()


def test_duplicate_resolved_episode_paths_are_rejected(tmp_path: Path) -> None:
    text = _VALID_CONFIG.replace(
        'episodes = ["episodes/a"]',
        'episodes = ["episodes/a", "episodes/../episodes/a"]',
    )
    error = _load_invalid(tmp_path, text)
    assert "episodes[1]" in str(error)
    assert "duplicate" in str(error)


@pytest.mark.parametrize(
    ("field", "original", "replacement"),
    [
        ("episodes", 'episodes = ["episodes/a"]', 'episodes = ["\\\\output"]'),
        ("output_dir", 'output_dir = "runs/result"', 'output_dir = "\\\\output"'),
    ],
)
def test_windows_root_relative_config_paths_are_rejected(
    tmp_path: Path, field: str, original: str, replacement: str
) -> None:
    if field == "episodes":
        (tmp_path / "\\output").mkdir()
    error = _load_invalid(tmp_path, _VALID_CONFIG.replace(original, replacement))
    assert field in str(error)
    assert "relative" in str(error)


def test_huge_budget_rate_integer_is_wrapped_as_configuration_error(tmp_path: Path) -> None:
    huge_integer = "1" + "0" * 400
    text = _VALID_CONFIG.replace(
        "[0.1, 0.25, 0.5, 0.75, 1.0]", f"[{huge_integer}]"
    )
    with pytest.raises(ConfigurationError) as error:
        BenchmarkConfig.load(_write_config(tmp_path, text))
    assert "budget_rates[0]" in str(error.value)
    assert isinstance(error.value.__cause__, OverflowError)


def test_malformed_parameter_values_are_rejected(tmp_path: Path) -> None:
    text = _VALID_CONFIG.replace(
        'parameters = { nested = { values = [1, { enabled = true }] }, label = "tracker" }',
        'parameters = { value = inf }',
    )
    error = _load_invalid(tmp_path, text)
    assert "tracker.parameters.value" in str(error)


def test_factory_path_loads_existing_callable() -> None:
    assert load_factory("python:squint_rl.policies:greedy_factory").__name__ == "greedy_factory"


@pytest.mark.parametrize(
    "path",
    [
        "squint_rl.policies.greedy_factory",
        "python:squint_rl.policies:greedy_factory:extra",
        "python::greedy_factory",
        "python:squint_rl.policies:",
    ],
)
def test_factory_format_is_explicit(path: str) -> None:
    with pytest.raises(ConfigurationError, match="python:module.path:attribute") as error:
        load_factory(path)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "path",
    [
        "python:not_a_real_squint_module:factory",
        "python:squint_rl.policies:not_a_real_attribute",
        "python:squint_rl.env:RUN_DETECTOR",
    ],
)
def test_factory_import_attribute_and_type_failures_are_wrapped(path: str) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_factory(path)
    assert error.value.__cause__ is not None


def test_minimal_override_behaviors_are_preserved(tmp_path: Path) -> None:
    config = BenchmarkConfig.load(_write_config(tmp_path))

    override = config.with_policy_override("python:squint_rl.policies:greedy_factory")
    assert override.policies == (
        config.policies[0].__class__("external", "python:squint_rl.policies:greedy_factory", MappingProxyType({})),
    )
    output_override = config.with_output_dir("runs/override")
    assert output_override.output_dir == (Path.cwd() / "runs/override").resolve()
