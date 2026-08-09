from __future__ import annotations

import csv
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from squint_rl.config import (
    BenchmarkConfig,
    ConfigurationError,
    PolicySpec,
    TrackerSpec,
)
from squint_rl.episode import Episode, EpisodeValidationError, seal_episode
from squint_rl.synthetic import make_synthetic_episode


def _tree_hash(path: Path) -> str:
    digest = sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = item.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _config_text(
    *,
    output_dir: str,
    policy_factory: str = "python:tests.fakes:greedy_factory",
    budget_rates: str = "[0.10, 0.50]",
) -> str:
    return f'''schema_version = 1
episodes = ["episode-a", "episode-b"]
output_dir = "{output_dir}"
budget_rates = {budget_rates}
seed = 17

[observation_scales]
active_tracks = 8.0
age_s = 5.0
motion_px_s = 20.0
time_since_detector_s = 5.0

[tracker]
factory = "python:tests.fakes:tracker_factory"

[[policies]]
id = "configured"
factory = "{policy_factory}"
'''


def _make_episode(
    path: Path,
    *,
    frame_count: int,
    latency_ms: float,
    identifier: str,
    detector_overrides: dict[str, object] | None = None,
    cost_profile_overrides: dict[str, object] | None = None,
    normalization_overrides: dict[str, object] | None = None,
    normalization_to_remove: str | None = None,
) -> Path:
    source = make_synthetic_episode(
        path.with_name(f"{path.name}-source"),
        frame_count=frame_count,
        fps=2.0,
        change_frames=(0, frame_count - 1),
        latency_ms=latency_ms,
    )
    episode = Episode.open(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["episode"]["id"] = identifier
    manifest["detector"]["weights_sha256"] = "f" * 64
    cost_profile: dict[str, object] = {
        "unit": "detector_ms",
        "p95_ms": 10.0,
        "reserve_ms": 10.0,
        "capacity_ms": 20.0,
    }
    profile_payload = json.dumps(
        {
            "cost_profile": cost_profile,
            "normalization": manifest["normalization"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    cost_profile["profile_sha256"] = sha256(profile_payload).hexdigest()
    manifest["cost_profile"] = cost_profile
    if detector_overrides is not None:
        manifest["detector"].update(detector_overrides)
    if cost_profile_overrides is not None:
        manifest["cost_profile"].update(cost_profile_overrides)
    if normalization_overrides is not None:
        manifest["normalization"].update(normalization_overrides)
    if normalization_to_remove is not None:
        manifest["normalization"].pop(normalization_to_remove)
    manifest["artifacts"] = {}
    arrays = {
        name: np.array(value, copy=True) for name, value in episode.arrays.items()
    }
    return seal_episode(path, manifest=manifest, arrays=arrays)


@pytest.fixture
def synthetic_episode(tmp_path: Path) -> Path:
    return _make_episode(
        tmp_path / "episode", frame_count=5, latency_ms=10.0, identifier="single"
    )


@pytest.fixture
def synthetic_config(tmp_path: Path) -> Path:
    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier="a"
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=7, latency_ms=20.0, identifier="b"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")
    return config_path


def test_repeated_baseline_has_identical_actions_and_metric_inputs(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    first_config = BenchmarkConfig.load(synthetic_config)
    first = evaluate(first_config)
    second = evaluate(
        first_config.with_output_dir(first_config.output_dir.with_name("run-2"))
    )

    assert first.action_sha256 == second.action_sha256
    assert first.metric_input_sha256 == second.metric_input_sha256
    assert _tree_hash(first.output_dir / "tracks") == _tree_hash(
        second.output_dir / "tracks"
    )


def test_outputs_have_only_public_artifacts_and_anchor_runs(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    result = evaluate(BenchmarkConfig.load(synthetic_config))

    assert {item.name for item in result.output_dir.iterdir()} == {
        "config.json",
        "provenance.json",
        "results.json",
        "curve.csv",
        "tracks",
    }
    assert result.results["anchors"]["all_frame"]["detector_calls"] == 12
    assert result.results["anchors"]["first_frame_only"]["detector_calls"] == 2
    assert not any("trackeval" in item.name for item in result.output_dir.rglob("*"))


def test_all_frame_anchor_bypasses_budget_admission(tmp_path: Path) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=4, latency_ms=100.0, identifier="a"
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=3, latency_ms=100.0, identifier="b"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    result = evaluate(config_path)

    assert result.results["anchors"]["all_frame"]["detector_calls"] == 7


def test_import_path_override_replaces_configured_policies(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config).with_policy_override(
        "python:tests.fakes:never_detect_factory"
    )
    result = evaluate(config)

    assert set(result.results["policies"]) == {"external"}
    assert all(
        point["detector_calls"] == 0
        for point in result.results["policies"]["external"]["points"]
    )


def test_direct_callable_override_is_truthful_in_config_and_provenance(
    synthetic_config: Path,
) -> None:
    from tests.fakes import never_detect_factory

    from squint_rl.benchmark import evaluate

    result = evaluate(
        BenchmarkConfig.load(synthetic_config), policy_factory=never_detect_factory
    )
    config = json.loads(result.output_dir.joinpath("config.json").read_text())
    provenance = json.loads(result.output_dir.joinpath("provenance.json").read_text())

    assert config["policies"] == [
        {"id": "external", "factory": "<callable>", "parameters": {}}
    ]
    assert provenance["policy_factories"] == ["<callable>"]
    assert (
        provenance["config_sha256"]
        == sha256(result.output_dir.joinpath("config.json").read_bytes()).hexdigest()
    )
    assert (
        provenance["config_sha256"]
        != sha256(
            BenchmarkConfig.load(synthetic_config).source_path.read_bytes()
        ).hexdigest()
    )


def test_effective_config_hash_does_not_require_a_readable_source_file(
    synthetic_config: Path, tmp_path: Path
) -> None:
    from squint_rl.benchmark import evaluate

    config = replace(
        BenchmarkConfig.load(synthetic_config), source_path=tmp_path / "missing.toml"
    )
    result = evaluate(config)
    provenance = json.loads(result.output_dir.joinpath("provenance.json").read_text())

    assert (
        provenance["config_sha256"]
        == sha256(result.output_dir.joinpath("config.json").read_bytes()).hexdigest()
    )


def test_policy_identifier_rejection_happens_before_atomic_publication(
    synthetic_config: Path, tmp_path: Path
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    unsafe = replace(
        config,
        policies=(
            PolicySpec("../../escaped", "python:tests.fakes:never_detect_factory", {}),
        ),
    )

    with pytest.raises(ConfigurationError, match=r"policies\[0\].id"):
        evaluate(unsafe)

    assert not unsafe.output_dir.exists()
    assert not (tmp_path / "escaped").exists()
    assert not list(
        unsafe.output_dir.parent.glob(f".{unsafe.output_dir.name}.*.incomplete")
    )


@pytest.mark.parametrize(
    "identifier", ["all-frame", "first-frame-only", "anchors", "ANCHORS", "/tmp/x"]
)
def test_reserved_and_path_like_policy_identifiers_are_rejected(
    synthetic_config: Path, identifier: str
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    unsafe = replace(
        config,
        policies=(
            PolicySpec(identifier, "python:tests.fakes:never_detect_factory", {}),
        ),
    )

    with pytest.raises(ConfigurationError, match=r"policies\[0\].id"):
        evaluate(unsafe)
    assert not unsafe.output_dir.exists()


@pytest.mark.parametrize(
    "identifier",
    [
        "CON",
        "PRN.log",
        "aux",
        "NUL",
        "COM1.txt",
        "lpt9.csv",
        "foo.",
        "scene:1",
        "name-space ",
    ],
)
def test_windows_reserved_and_ambiguous_policy_identifiers_are_rejected(
    synthetic_config: Path, identifier: str
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    invalid = replace(
        config,
        policies=(
            PolicySpec(identifier, "python:tests.fakes:never_detect_factory", {}),
        ),
    )

    with pytest.raises(ConfigurationError, match=r"policies\[0\].id"):
        evaluate(invalid)
    assert not invalid.output_dir.exists()
    assert not list(
        invalid.output_dir.parent.glob(f".{invalid.output_dir.name}.*.incomplete")
    )


def test_policy_identifier_case_collisions_are_rejected_before_atomic_run(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    collision = replace(
        config,
        policies=(
            PolicySpec("foo", "python:tests.fakes:never_detect_factory", {}),
            PolicySpec("FOO", "python:tests.fakes:never_detect_factory", {}),
        ),
    )

    with pytest.raises(ConfigurationError, match=r"policies\[1\].id"):
        evaluate(collision)
    assert not collision.output_dir.exists()
    assert not list(
        collision.output_dir.parent.glob(f".{collision.output_dir.name}.*.incomplete")
    )


@pytest.mark.parametrize(
    "identifier", ["CON", "COM1.txt", "foo.", "scene:1", "name-space "]
)
def test_unsafe_episode_identifier_is_rejected_before_atomic_run(
    tmp_path: Path, identifier: str
) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier=identifier
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=7, latency_ms=20.0, identifier="other"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="episode.id"):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.*.incomplete"))


def test_episode_identifier_case_collisions_are_rejected_before_atomic_run(
    tmp_path: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier="foo"
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=7, latency_ms=20.0, identifier="FOO"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="episode.id"):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.*.incomplete"))


def test_track_paths_distinguish_constrained_all_frame_named_policy(
    tmp_path: Path,
) -> None:
    from squint_rl.benchmark import _rate_directory, _track_path

    assert _track_path(tmp_path, "all-frame", 0.1, "sequence") == (
        tmp_path / "tracks" / "all-frame" / _rate_directory(0.1) / "sequence.txt"
    )


def test_close_rates_use_distinct_track_directories(tmp_path: Path) -> None:
    from squint_rl.benchmark import _rate_directory

    first = _rate_directory(0.1000001)
    second = _rate_directory(0.1000002)

    assert first != second
    assert first.startswith("rho-") and second.startswith("rho-")


def test_close_rates_keep_distinct_track_files_and_metric_inputs(
    tmp_path: Path,
) -> None:
    from squint_rl.benchmark import _rate_directory, evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier="a"
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=7, latency_ms=20.0, identifier="b"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        _config_text(
            output_dir="result",
            budget_rates="[0.1000001, 0.1000002]",
        ),
        encoding="utf-8",
    )

    result = evaluate(config_path)
    directories = {
        item.name
        for item in result.output_dir.joinpath("tracks", "configured").iterdir()
    }

    assert directories == {_rate_directory(0.1000001), _rate_directory(0.1000002)}
    assert (
        len(list(result.output_dir.joinpath("tracks", "configured").rglob("*.txt")))
        == 4
    )
    assert result.metric_input_sha256
    with result.output_dir.joinpath("curve.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rates = [row["nominal_rate"] for row in csv.DictReader(stream)]
    assert rates == ["0.1000001", "0.1000002"]
    assert {float(rate) for rate in rates} == {0.1000001, 0.1000002}


def test_manual_factory_validation_happens_before_atomic_publication(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    invalid = replace(config, tracker=TrackerSpec("python:not_real_module:factory", {}))

    with pytest.raises(ConfigurationError, match="tracker.factory"):
        evaluate(invalid)
    assert not invalid.output_dir.exists()
    assert not list(
        invalid.output_dir.parent.glob(f".{invalid.output_dir.name}.*.incomplete")
    )


def test_manual_effective_policy_factory_validation_happens_before_atomic_publication(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    invalid = replace(
        config,
        policies=(PolicySpec("valid", "python:not_real_module:factory", {}),),
    )

    with pytest.raises(ConfigurationError, match=r"policies\[0\].factory"):
        evaluate(invalid)
    assert not invalid.output_dir.exists()
    assert not list(
        invalid.output_dir.parent.glob(f".{invalid.output_dir.name}.*.incomplete")
    )


def test_tracker_constructor_failure_retains_incomplete_output(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    runtime = replace(
        config,
        tracker=TrackerSpec("python:tests.fakes:raising_tracker_factory", {}),
    )

    with pytest.raises(RuntimeError, match="tracker construction failed"):
        evaluate(runtime)
    assert not runtime.output_dir.exists()
    assert (
        len(
            list(
                runtime.output_dir.parent.glob(
                    f".{runtime.output_dir.name}.*.incomplete"
                )
            )
        )
        == 1
    )


def test_stateful_factories_are_fresh_and_seeded_per_constrained_unit(
    synthetic_config: Path,
) -> None:
    from tests import fakes

    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    config = replace(
        config,
        tracker=TrackerSpec("python:tests.fakes:stateful_tracker_factory", {}),
        policies=(
            PolicySpec("stateful", "python:tests.fakes:stateful_policy_factory", {}),
        ),
    )
    fakes.reset_stateful_records()
    first = evaluate(config)
    first_seeds = list(fakes.stateful_policy_seeds)
    first_tracker_instances = list(fakes.stateful_tracker_instances)
    first_policy_instances = list(fakes.stateful_policy_instances)

    fakes.reset_stateful_records()
    evaluate(config.with_output_dir(config.output_dir.with_name("repeat")))

    assert (
        len(first_policy_instances)
        == len({id(item) for item in first_policy_instances})
        == 4
    )
    assert (
        len(first_tracker_instances)
        == len({id(item) for item in first_tracker_instances})
        == 8
    )
    assert len(first_seeds) == len(set(first_seeds)) == 4
    assert fakes.stateful_policy_seeds == first_seeds
    assert first.action_sha256


def test_unequal_episode_costs_use_one_summed_denominator_and_common_support(
    synthetic_config: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    result = evaluate(synthetic_config)
    anchors = result.results["anchors"]
    points = result.results["policies"]["configured"]["points"]

    assert anchors["all_frame"]["charged_detector_ms"] == pytest.approx(190.0)
    assert anchors["first_frame_only"]["realized_compute"] == pytest.approx(
        30.0 / 190.0
    )
    assert result.results["curve_areas"]["support"] == pytest.approx(
        [
            min(point["realized_compute"] for point in points),
            max(point["realized_compute"] for point in points),
        ]
    )


@pytest.mark.parametrize(
    "detector_overrides",
    [
        {"threshold": 0.2},
        {"input_size": [64, 64]},
        {"precision": "float16"},
        {"weights_sha256": "a" * 64},
    ],
)
def test_complete_detector_profile_must_match_before_atomic_publication(
    tmp_path: Path, detector_overrides: dict[str, object]
) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier="a"
    )
    _make_episode(
        tmp_path / "episode-b",
        frame_count=7,
        latency_ms=20.0,
        identifier="b",
        detector_overrides=detector_overrides,
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="detector profile"):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "profile_sha256",
    [None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_profile_hash_must_be_lowercase_sha256_before_atomic_publication(
    tmp_path: Path, profile_sha256: object
) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a",
        frame_count=5,
        latency_ms=10.0,
        identifier="a",
        cost_profile_overrides={"profile_sha256": profile_sha256},
    )
    _make_episode(
        tmp_path / "episode-b", frame_count=7, latency_ms=20.0, identifier="b"
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="cost_profile.profile_sha256"):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("cost_profile_overrides", "normalization_overrides", "message"),
    [
        ({"profile_sha256": "b" * 64}, None, "cost_profile.profile_sha256"),
        ({"capacity_ms": 21.0}, None, "cost_profile"),
        (None, {"age_s": 6.0}, "normalization"),
    ],
)
def test_frozen_episode_profiles_must_match_before_atomic_publication(
    tmp_path: Path,
    cost_profile_overrides: dict[str, object] | None,
    normalization_overrides: dict[str, object] | None,
    message: str,
) -> None:
    from squint_rl.benchmark import evaluate

    _make_episode(
        tmp_path / "episode-a", frame_count=5, latency_ms=10.0, identifier="a"
    )
    _make_episode(
        tmp_path / "episode-b",
        frame_count=7,
        latency_ms=20.0,
        identifier="b",
        cost_profile_overrides=cost_profile_overrides,
        normalization_overrides=normalization_overrides,
    )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match=message):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


def test_config_observation_scales_must_match_frozen_normalization(
    tmp_path: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    for name, frame_count, latency_ms in (("a", 5, 10.0), ("b", 7, 20.0)):
        _make_episode(
            tmp_path / f"episode-{name}",
            frame_count=frame_count,
            latency_ms=latency_ms,
            identifier=name,
            normalization_overrides={"age_s": 6.0},
        )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="observation_scales.age_s"):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    ("normalization_overrides", "normalization_to_remove", "message"),
    [
        (
            {"unexpected": 1.0},
            None,
            "manifest normalization.unexpected is not supported",
        ),
        (None, "age_s", "manifest normalization.age_s is required"),
    ],
)
def test_normalization_requires_exact_observation_scale_keys(
    tmp_path: Path,
    normalization_overrides: dict[str, object] | None,
    normalization_to_remove: str | None,
    message: str,
) -> None:
    from squint_rl.benchmark import evaluate

    for name, frame_count, latency_ms in (("a", 5, 10.0), ("b", 7, 20.0)):
        _make_episode(
            tmp_path / f"episode-{name}",
            frame_count=frame_count,
            latency_ms=latency_ms,
            identifier=name,
            normalization_overrides=normalization_overrides,
            normalization_to_remove=normalization_to_remove,
        )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match=message):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "field",
    ["active_tracks", "age_s", "motion_px_s", "time_since_detector_s"],
)
def test_huge_normalization_integer_has_field_specific_validation_error(
    tmp_path: Path, field: str
) -> None:
    from squint_rl.benchmark import evaluate

    for name, frame_count, latency_ms in (("a", 5, 10.0), ("b", 7, 20.0)):
        _make_episode(
            tmp_path / f"episode-{name}",
            frame_count=frame_count,
            latency_ms=latency_ms,
            identifier=name,
            normalization_overrides={field: 10**400},
        )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match=rf"manifest normalization\.{field} must be a positive finite number",
    ):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


def test_huge_reserve_integer_has_field_specific_validation_error(
    tmp_path: Path,
) -> None:
    from squint_rl.benchmark import evaluate

    for name, frame_count, latency_ms in (("a", 5, 10.0), ("b", 7, 20.0)):
        _make_episode(
            tmp_path / f"episode-{name}",
            frame_count=frame_count,
            latency_ms=latency_ms,
            identifier=name,
            cost_profile_overrides={"reserve_ms": 10**400},
        )
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(_config_text(output_dir="result"), encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match="manifest cost_profile.reserve_ms must be a positive finite number",
    ):
        evaluate(config_path)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "policy_factory",
    ["python:tests.fakes:raising_policy_factory"],
)
def test_runtime_failure_retains_only_incomplete_output(
    synthetic_config: Path, policy_factory: str
) -> None:
    from squint_rl.benchmark import evaluate

    config = BenchmarkConfig.load(synthetic_config)
    config = config.with_policy_override(policy_factory)
    with pytest.raises(RuntimeError, match="policy execution failed"):
        evaluate(config)

    assert not config.output_dir.exists()
    incomplete = list(
        config.output_dir.parent.glob(f".{config.output_dir.name}.*.incomplete")
    )
    assert len(incomplete) == 1
    assert not list(
        config.output_dir.parent.glob(f".{config.output_dir.name}.*.publish.lock")
    )


def test_results_are_canonical_json(synthetic_config: Path) -> None:
    from squint_rl.benchmark import evaluate

    result = evaluate(synthetic_config)
    payload = result.output_dir.joinpath("results.json").read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload) == result.results


def test_public_exports_have_the_documented_order() -> None:
    import squint_rl

    assert squint_rl.__all__ == [
        "Episode",
        "Tracker",
        "SquintEnv",
        "evaluate",
        "__version__",
    ]
