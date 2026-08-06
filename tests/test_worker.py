from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from conftest import FakeDetector

from edge_perception.config import RunConfig, write_run_config
from edge_perception.worker import main, run_worker


def _write_config(
    path: Path,
    *,
    video_path: Path,
    output_dir: Path,
    detector_id: str = "dfine-nano-coco",
) -> None:
    write_run_config(
        path,
        RunConfig(
            input_path=video_path,
            output_dir=output_dir,
            regions=(),
            threshold=0.3,
            max_frames=1,
            warmup_runs=0,
            annotate_every=0,
            detector_id=detector_id,
            device="cpu",
        ),
    )


def _events(stream: StringIO) -> list[dict[str, object]]:
    lines = [line for line in stream.getvalue().splitlines() if line]
    events = [json.loads(line) for line in lines]
    assert all(isinstance(event, dict) for event in events)
    assert all("NaN" not in line and "Infinity" not in line for line in lines)
    return events


def test_worker_emits_one_finite_json_object_per_line(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    config_path = tmp_path / "experiment.json"
    _write_config(config_path, video_path=video_path, output_dir=tmp_path / "run")
    stream = StringIO()

    exit_code = run_worker(
        config_path,
        tmp_path / "cancel",
        detector_loader=lambda _detector_id, *, threshold, device: fake_detector,
        stream=stream,
    )

    events = _events(stream)
    assert exit_code == 0
    assert [event["phase"] for event in events] == [
        "validating",
        "warming_up",
        "running",
        "complete",
    ]
    assert sum(event["phase"] in {"complete", "cancelled", "failed"} for event in events) == 1


def test_worker_honors_preexisting_cancel_file(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    run_dir = tmp_path / "cancelled-run"
    config_path = tmp_path / "experiment.json"
    cancel_file = tmp_path / "cancel"
    cancel_file.touch()
    _write_config(config_path, video_path=video_path, output_dir=run_dir)
    stream = StringIO()

    exit_code = run_worker(
        config_path,
        cancel_file,
        detector_loader=lambda _detector_id, *, threshold, device: fake_detector,
        stream=stream,
    )

    events = _events(stream)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert events[-1]["phase"] == "cancelled"
    assert summary["status"] == "cancelled"


def test_worker_reports_detector_load_failure_once(
    video_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "experiment.json"
    _write_config(
        config_path,
        video_path=video_path,
        output_dir=tmp_path / "run",
        detector_id="missing-detector",
    )

    exit_code = main(["--config", str(config_path), "--cancel-file", str(tmp_path / "cancel")])

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line]
    stderr_lines = [line for line in captured.err.splitlines() if line]
    events = [json.loads(line) for line in stdout_lines]
    assert exit_code == 2
    assert [event["phase"] for event in events] == ["failed"]
    assert len(stderr_lines) == 1
    assert "unknown detector ID" in stderr_lines[0]


def test_worker_argument_failure_preserves_jsonl_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line]
    stderr_lines = [line for line in captured.err.splitlines() if line]
    events = [json.loads(line) for line in stdout_lines]
    assert exit_code == 2
    assert [event["phase"] for event in events] == ["failed"]
    assert len(stderr_lines) == 1
