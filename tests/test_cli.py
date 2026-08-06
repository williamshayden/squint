from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import FakeDetector

from edge_perception import cli


def _install_fake_dfine(
    monkeypatch: pytest.MonkeyPatch,
    detector: FakeDetector | None = None,
    error: Exception | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeDfineDetector:
        @classmethod
        def load(cls, *, threshold: float, device: str) -> FakeDetector:
            calls.append({"threshold": threshold, "device": device})
            if error is not None:
                raise error
            assert detector is not None
            return detector

    module = ModuleType("edge_perception.detectors.dfine")
    module.DfineDetector = FakeDfineDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_perception.detectors.dfine", module)
    return calls


def test_run_command_parses_declared_crops_and_passes_threshold_once(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "run"
    load_calls = _install_fake_dfine(monkeypatch, fake_detector)
    runner_calls: list[tuple[object, object]] = []

    def fake_run(config: object, detector: object) -> dict[str, object]:
        runner_calls.append((config, detector))
        return {"status": "complete", "frames_processed": 60, "inference_count": 180}

    monkeypatch.setattr(cli, "run_checkpoint", fake_run)

    exit_code = cli.main(
        [
            "run",
            str(video_path),
            "--output",
            str(output_dir),
            "--crop",
            "right:120,0,80,50",
            "--crop",
            "lower-left:0,50,100,50",
            "--device",
            "auto",
            "--threshold",
            "0.3",
            "--warmup-runs",
            "2",
            "--max-frames",
            "60",
            "--annotate-every",
            "10",
        ]
    )

    assert exit_code == 0
    assert load_calls == [{"threshold": 0.3, "device": "auto"}]
    config, passed_detector = runner_calls[0]
    assert passed_detector is fake_detector
    assert config.input_path == video_path.resolve()
    assert config.output_dir == output_dir.resolve()
    assert [region.region_id for region in config.regions] == ["right", "lower-left"]
    assert config.threshold == 0.3
    assert config.max_frames == 60
    assert config.warmup_runs == 2
    assert config.annotate_every == 10
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"output: {output_dir.resolve()}",
        "status=complete frames=60 inferences=180",
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--crop", "bad"), "crop must use ID:X,Y,WIDTH,HEIGHT"),
        (("--crop", "right:a,0,10,10"), "crop coordinates must be integers"),
        (("--crop", "right:0,0,0,10"), "crop width and height must be positive"),
        (("--crop", "right:0,0,10,-1"), "crop width and height must be positive"),
        (("--threshold", "-0.1"), "threshold must be between 0 and 1"),
        (("--threshold", "1.1"), "threshold must be between 0 and 1"),
        (("--max-frames", "-1"), "max-frames must not be negative"),
        (("--warmup-runs", "-1"), "warmup-runs must not be negative"),
        (("--annotate-every", "-1"), "annotate-every must not be negative"),
    ],
)
def test_run_command_rejects_invalid_values_with_one_line_error(
    arguments: tuple[str, ...],
    message: str,
    video_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["run", str(video_path), "--output", str(tmp_path / "run"), *arguments])

    assert exit_code != 0
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: ")
    assert message in lines[0]


def test_run_command_rejects_duplicate_crop_ids(
    video_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "run",
            str(video_path),
            "--output",
            str(tmp_path / "run"),
            "--crop",
            "right:0,0,10,10",
            "--crop",
            "right:10,0,10,10",
        ]
    )

    assert exit_code != 0
    assert capsys.readouterr().err.strip().splitlines() == ["error: duplicate crop ID: right"]


def test_run_command_rejects_output_equal_to_input(
    video_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["run", str(video_path), "--output", str(video_path)])

    assert exit_code != 0
    assert capsys.readouterr().err.strip().splitlines() == ["error: output must differ from input"]


def test_run_command_reports_unavailable_requested_cuda_without_traceback(
    video_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_dfine(monkeypatch, error=RuntimeError("CUDA was requested but is not available"))

    exit_code = cli.main(
        ["run", str(video_path), "--output", str(tmp_path / "run"), "--device", "cuda"]
    )

    assert exit_code != 0
    assert capsys.readouterr().err.strip().splitlines() == [
        "error: CUDA was requested but is not available"
    ]


@pytest.mark.parametrize(("equivalent", "exit_code"), [(True, 0), (False, 1)])
def test_compare_command_exit_code_reflects_equivalence(
    equivalent: bool,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report: dict[str, Any] = {
        "equivalent": equivalent,
        "left_detection_count": 1,
        "right_detection_count": 1,
        "mismatch_count": 0 if equivalent else 1,
    }
    monkeypatch.setattr(cli, "compare_runs", lambda *_args, **_kwargs: report)

    result = cli.main(["compare", str(tmp_path / "left"), str(tmp_path / "right")])

    assert result == exit_code
    assert capsys.readouterr().out.strip() == (
        f"equivalent={str(equivalent).lower()} left=1 right=1 mismatches={report['mismatch_count']}"
    )
