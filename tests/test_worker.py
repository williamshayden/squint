from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from conftest import FakeDetector

from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    write_run_config,
)
from edge_perception.contracts import Region
from edge_perception.worker import main, run_worker


def _write_config(
    path: Path,
    *,
    video_path: Path,
    output_dir: Path,
    detector_id: str = "dfine-nano-coco",
    regions: tuple[Region, ...] = (),
    capture: CaptureResult | None = None,
) -> None:
    write_run_config(
        path,
        RunConfig(
            input_path=video_path,
            output_dir=output_dir,
            regions=regions,
            threshold=0.3,
            max_frames=1,
            warmup_runs=0,
            annotate_every=0,
            detector_id=detector_id,
            device="cpu",
            capture=capture,
        ),
    )


def _events(stream: StringIO) -> list[dict[str, object]]:
    lines = [line for line in stream.getvalue().splitlines() if line]
    events = [json.loads(line) for line in lines]
    assert all(isinstance(event, dict) for event in events)
    assert all("NaN" not in line and "Infinity" not in line for line in lines)
    return events


def _capture_provenance(video_path: Path, *, sha256: str) -> CaptureResult:
    return CaptureResult(
        request=CaptureRequest("camera-1", "Test camera", 200, 100, 30.0, True),
        selected_width=200,
        selected_height=100,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=200,
        actual_height=100,
        actual_fps=30.0,
        container="mp4",
        codec="mpeg4",
        duration_seconds=0.1,
        has_audio=False,
        file_size_bytes=video_path.stat().st_size,
        path=video_path,
        sha256=sha256,
    )


@pytest.mark.parametrize(
    "invalid_input",
    ["deleted_source", "out_of_bounds_roi", "nonempty_output", "capture_sha_mismatch"],
)
def test_worker_preflight_rejects_invalid_input_before_loading_detector(
    invalid_input: str,
    video_path: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    regions: tuple[Region, ...] = ()
    capture = None
    if invalid_input == "out_of_bounds_roi":
        regions = (Region("outside", 199, 0, 2, 1),)
    elif invalid_input == "nonempty_output":
        output_dir.mkdir()
        (output_dir / "keep-me.txt").write_text("user data", encoding="utf-8")
    elif invalid_input == "capture_sha_mismatch":
        capture = _capture_provenance(video_path, sha256="0" * 64)

    config_path = tmp_path / "experiment.json"
    _write_config(
        config_path,
        video_path=video_path,
        output_dir=output_dir,
        regions=regions,
        capture=capture,
    )
    if invalid_input == "deleted_source":
        video_path.unlink()

    load_calls: list[str] = []

    def forbidden_loader(detector_id: str, **_kwargs: object) -> object:
        load_calls.append(detector_id)
        raise AssertionError("detector loader must not run before preflight succeeds")

    protocol = StringIO()
    diagnostics = StringIO()
    exit_code = main(
        ["--config", str(config_path), "--cancel-file", str(tmp_path / "cancel")],
        detector_loader=forbidden_loader,  # type: ignore[arg-type]
        protocol_stream=protocol,
        diagnostic_stream=diagnostics,
    )

    assert exit_code == 2
    assert [event["phase"] for event in _events(protocol)] == ["failed"]
    assert diagnostics.getvalue().startswith("error: ")
    assert diagnostics.getvalue().count("\n") == 1
    assert "Traceback" not in diagnostics.getvalue()
    assert load_calls == []


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


def test_worker_redirects_noisy_dependency_stdout_away_from_jsonl_protocol(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    config_path = tmp_path / "experiment.json"
    _write_config(config_path, video_path=video_path, output_dir=tmp_path / "run")
    protocol = StringIO()
    diagnostics = StringIO()
    original_predict = fake_detector.predict

    def noisy_predict(images: tuple[object, ...]) -> object:
        print("detector noise")
        return original_predict(images)  # type: ignore[arg-type]

    def noisy_loader(_detector_id: str, *, threshold: float, device: str) -> FakeDetector:
        print("loader noise")
        fake_detector.predict = noisy_predict  # type: ignore[method-assign]
        return fake_detector

    with redirect_stdout(protocol):
        exit_code = run_worker(
            config_path,
            tmp_path / "cancel",
            detector_loader=noisy_loader,
            stream=protocol,
            diagnostic_stream=diagnostics,
        )

    events = _events(protocol)
    assert exit_code == 0
    assert events[-1]["phase"] == "complete"
    assert diagnostics.getvalue().splitlines() == ["loader noise", "detector noise"]


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


def test_worker_main_reports_runner_failure_without_duplicate_terminal_event(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
) -> None:
    config_path = tmp_path / "experiment.json"
    _write_config(config_path, video_path=video_path, output_dir=tmp_path / "run")
    protocol = StringIO()
    diagnostics = StringIO()

    def fail(_images: tuple[object, ...]) -> object:
        raise RuntimeError("runner failed")

    fake_detector.predict = fail  # type: ignore[method-assign]

    exit_code = main(
        ["--config", str(config_path), "--cancel-file", str(tmp_path / "cancel")],
        detector_loader=lambda _detector_id, *, threshold, device: fake_detector,
        protocol_stream=protocol,
        diagnostic_stream=diagnostics,
    )

    events = _events(protocol)
    assert exit_code == 2
    assert [event["phase"] for event in events] == ["validating", "warming_up", "failed"]
    assert sum(event["phase"] == "failed" for event in events) == 1
    assert diagnostics.getvalue().splitlines() == ["error: runner failed"]


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
