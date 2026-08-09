from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from squint_rl.config import BenchmarkConfig
from squint_rl.episode import Episode, seal_episode
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
    *, output_dir: str, policy_factory: str = "python:tests.fakes:greedy_factory"
) -> str:
    return f'''schema_version = 1
episodes = ["episode-a", "episode-b"]
output_dir = "{output_dir}"
budget_rates = [0.10, 0.50]
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
    path: Path, *, frame_count: int, latency_ms: float, identifier: str
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
