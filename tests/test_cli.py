from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from squint_rl.cli import main
from squint_rl.episode import Episode, seal_episode
from squint_rl.synthetic import make_synthetic_episode


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
    config_path.write_text(
        """schema_version = 1
episodes = ["episode-a", "episode-b"]
output_dir = "result"
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
factory = "python:tests.fakes:greedy_factory"
""",
        encoding="utf-8",
    )
    return config_path


def test_validate_and_benchmark_cli(
    synthetic_episode: Path, synthetic_config: Path, capsys
) -> None:
    assert main(["episode", "validate", str(synthetic_episode)]) == 0
    assert "valid" in capsys.readouterr().out.lower()

    assert main(["benchmark", str(synthetic_config)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "complete"


def test_invalid_episode_is_nonzero_and_field_specific(tmp_path: Path, capsys) -> None:
    episode = tmp_path / "empty-episode"
    episode.mkdir()

    assert main(["episode", "validate", str(episode)]) == 2
    assert "manifest.json" in capsys.readouterr().err


def test_python_and_cli_results_bytes_match(synthetic_config: Path, capsys) -> None:
    from squint_rl.benchmark import evaluate
    from squint_rl.config import BenchmarkConfig

    config = BenchmarkConfig.load(synthetic_config)
    python_result = evaluate(config)
    cli_config = config.with_output_dir(config.output_dir.with_name("cli-result"))
    cli_config.source_path.write_text(
        cli_config.source_path.read_text(encoding="utf-8").replace(
            'output_dir = "result"', 'output_dir = "cli-result"'
        ),
        encoding="utf-8",
    )

    assert main(["benchmark", str(cli_config.source_path)]) == 0
    capsys.readouterr()
    assert python_result.output_dir.joinpath("results.json").read_bytes() == (
        cli_config.output_dir.joinpath("results.json").read_bytes()
    )


def test_cli_policy_override_uses_external_policy(
    synthetic_config: Path, capsys
) -> None:
    assert (
        main(
            [
                "benchmark",
                str(synthetic_config),
                "--policy",
                "python:tests.fakes:never_detect_factory",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    results = json.loads(
        Path(receipt["output_dir"]).joinpath("results.json").read_text()
    )
    assert set(results["policies"]) == {"external"}


def test_import_path_override_has_python_cli_results_byte_parity(
    synthetic_config: Path, capsys
) -> None:
    from squint_rl.benchmark import evaluate
    from squint_rl.config import BenchmarkConfig

    policy = "python:tests.fakes:never_detect_factory"
    config = BenchmarkConfig.load(synthetic_config)
    python_result = evaluate(config.with_policy_override(policy))
    config.source_path.write_text(
        config.source_path.read_text(encoding="utf-8").replace(
            'output_dir = "result"', 'output_dir = "cli-override"'
        ),
        encoding="utf-8",
    )

    assert main(["benchmark", str(config.source_path), "--policy", policy]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert python_result.output_dir.joinpath("results.json").read_bytes() == (
        Path(receipt["output_dir"]).joinpath("results.json").read_bytes()
    )


def test_invalid_config_exits_two_before_output_creation(
    tmp_path: Path, capsys
) -> None:
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text("schema_version = 1\n", encoding="utf-8")

    assert main(["benchmark", str(bad_config)]) == 2
    assert "config.episodes" in capsys.readouterr().err
    assert not (tmp_path / "result").exists()
