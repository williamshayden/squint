from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from conftest import FakeDetector

from edge_perception import cli
from edge_perception.capture import CameraDeviceInfo, CameraFormatInfo
from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    write_run_config,
)


def _camera_device() -> CameraDeviceInfo:
    return CameraDeviceInfo(
        "camera-1",
        "Desk camera",
        (
            CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object()),
            CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),
        ),
        object(),
    )


def _camera_result(request: CaptureRequest, output: Path) -> CaptureResult:
    return CaptureResult(
        request=request,
        selected_width=1920,
        selected_height=1080,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="YUYV",
        actual_width=1920,
        actual_height=1080,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=0.05,
        has_audio=False,
        file_size_bytes=7,
        path=output,
        sha256="b" * 64,
    )


def _install_camera_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: tuple[CameraDeviceInfo, ...] | None = None,
) -> list[tuple[CaptureRequest, float, Path | None]]:
    module = ModuleType("edge_perception.camera_cli")
    reported_devices = devices or (_camera_device(),)
    calls: list[tuple[CaptureRequest, float, Path | None]] = []
    module.list_cameras = lambda: reported_devices  # type: ignore[attr-defined]

    def capture(request: CaptureRequest, *, duration_seconds: float, output: Path | None) -> CaptureResult:
        calls.append((request, duration_seconds, output))
        return _camera_result(request, output or Path("controller-choice.mp4"))

    module.capture_camera = capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_perception.camera_cli", module)
    return calls


def test_camera_list_renders_discovered_device_and_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_camera_boundary(monkeypatch)

    assert cli.main(["camera", "list"]) == 0

    output = capsys.readouterr().out
    assert "camera-1" in output
    assert "Desk camera" in output
    assert "1280x720" in output
    assert "1920x1080" in output
    assert "15-60 fps" in output
    assert "30-30 fps" in output


def test_camera_capture_parses_constraints_uses_discovered_description_and_renders_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_camera_boundary(monkeypatch)
    output = tmp_path / "capture.mp4"

    assert (
        cli.main(
            [
                "camera",
                "capture",
                "--device",
                "camera-1",
                "--duration",
                "0.05",
                "--output",
                str(output),
                "--width",
                "1920",
                "--height",
                "1080",
                "--fps",
                "30",
                "--strict",
            ]
        )
        == 0
    )

    request = CaptureRequest("camera-1", "Desk camera", 1920, 1080, 30.0, True)
    assert calls == [(request, 0.05, output.resolve())]
    rendered = capsys.readouterr().out
    assert f"Capture: {output.resolve()}" in rendered
    assert "SHA-256: " + "b" * 64 in rendered
    assert (
        "Capture request: device=camera-1 description=Desk camera "
        "width=1920 px height=1080 px fps=30 fps strict=true"
    ) in rendered
    assert "Applied camera format: 1920x1080 30-30 fps YUYV" in rendered
    assert (
        "Recorded format: width=1920 px height=1080 px fps=30 fps "
        "container=mp4 codec=h264 duration=0.05 s audio=false size=7 bytes"
    ) in rendered


def test_camera_capture_omits_output_for_shared_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_camera_boundary(monkeypatch)

    assert cli.main(["camera", "capture", "--device", "camera-1", "--duration", "0.05"]) == 0

    assert calls[0][2] is None
    assert (
        "Capture request: device=camera-1 description=Desk camera "
        "width=None height=None fps=None strict=false"
    ) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--device", "missing", "--duration", "0.05"), "camera device is unavailable: missing"),
        (
            ("--device", "camera-1", "--duration", "0"),
            "argument --duration: duration must be finite and positive",
        ),
        (
            ("--device", "camera-1", "--duration", "nan"),
            "argument --duration: duration must be finite and positive",
        ),
    ],
)
def test_camera_capture_rejects_invalid_device_or_duration_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
    message: str,
) -> None:
    calls = _install_camera_boundary(monkeypatch)
    output = tmp_path / "capture.mp4"

    assert cli.main(["camera", "capture", *arguments, "--output", str(output)]) == 2

    assert capsys.readouterr().err == f"error: {message}\n"
    assert calls == []
    assert not output.exists()


def test_camera_capture_rejects_existing_output_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_camera_boundary(monkeypatch)
    output = tmp_path / "capture.mp4"
    output.write_text("user capture", encoding="utf-8")

    assert (
        cli.main(
            ["camera", "capture", "--device", "camera-1", "--duration", "0.05", "--output", str(output)]
        )
        == 2
    )

    assert capsys.readouterr().err == f"error: capture destination already exists: {output.resolve()}\n"
    assert calls == []
    assert output.read_text(encoding="utf-8") == "user capture"


def test_camera_command_reports_missing_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import edge_perception

    original_import = __import__
    monkeypatch.delitem(sys.modules, "edge_perception.camera_cli", raising=False)
    monkeypatch.delattr(edge_perception, "camera_cli", raising=False)

    def import_without_camera_extra(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "edge_perception.camera_cli":
            raise ImportError("No module named 'PySide6'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", import_without_camera_extra)

    assert cli.main(["camera", "list"]) == 2
    assert capsys.readouterr().err == (
        "error: camera support is unavailable; install adaptive-edge-perception[camera]\n"
    )


def test_camera_command_preserves_unrelated_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_perception

    original_import = __import__
    monkeypatch.delitem(sys.modules, "edge_perception.camera_cli", raising=False)
    monkeypatch.delattr(edge_perception, "camera_cli", raising=False)

    def import_with_unrelated_failure(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "edge_perception.camera_cli":
            raise ImportError("camera adapter has a broken dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", import_with_unrelated_failure)

    with pytest.raises(ImportError, match="camera adapter has a broken dependency"):
        cli.main(["camera", "list"])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ModuleNotFoundError("No module named 'PySide6Tools'", name="PySide6Tools"),
            False,
        ),
        (
            ImportError("cannot import name 'camera' from 'PySide6Shim'", name="PySide6Shim"),
            False,
        ),
        (ModuleNotFoundError("No module named 'PySide6'", name="PySide6"), True),
        (
            ImportError(
                "cannot import name 'QTimer' from 'PySide6.QtCore'",
                name="PySide6.QtCore",
            ),
            True,
        ),
    ],
)
def test_pyside6_import_classifier_requires_exact_module_boundary(
    error: ImportError,
    expected: bool,
) -> None:
    assert cli._is_pyside6_import_error(error) is expected


def test_gui_command_lazily_launches_native_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path | None] = []
    module = ModuleType("edge_perception.gui.app")
    module.launch_gui = lambda run_dir=None: calls.append(run_dir) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_perception.gui.app", module)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    assert cli.main(["gui", "--run", str(tmp_path)]) == 0
    assert calls == [tmp_path.resolve()]


def test_gui_command_reports_missing_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = __import__

    def import_without_gui_extra(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "edge_perception.gui.app":
            raise ImportError("No module named 'PySide6'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", import_without_gui_extra)

    assert cli.main(["gui"]) == 2
    captured = capsys.readouterr()
    assert captured.err == (
        "error: native GUI dependencies are unavailable; install adaptive-edge-perception[gui]\n"
    )
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("missing", ["manifest.json", "summary.json"])
def test_gui_command_requires_run_artifacts(
    tmp_path: Path,
    missing: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in {"manifest.json", "summary.json"} - {missing}:
        (run_dir / name).write_text("{}", encoding="utf-8")

    assert cli.main(["gui", "--run", str(run_dir)]) == 2
    assert capsys.readouterr().err == f"error: run directory is missing {missing}: {run_dir.resolve()}\n"


def _install_fake_detector(
    monkeypatch: pytest.MonkeyPatch,
    detector: FakeDetector | None = None,
    error: Exception | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_load_detector(detector_id: str, *, threshold: float, device: str) -> object:
        calls.append({"detector_id": detector_id, "threshold": threshold, "device": device})
        if error is not None:
            raise error
        assert detector is not None
        return detector

    monkeypatch.setattr(cli, "load_detector", fake_load_detector)
    return calls


def test_run_command_parses_declared_crops_and_passes_threshold_once(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    load_calls = _install_fake_detector(monkeypatch, fake_detector)
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
    assert load_calls == [{"detector_id": "dfine-nano-coco", "threshold": 0.3, "device": "auto"}]
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


def test_run_command_rejects_nonempty_output_before_loading_detector(
    video_path: Path,
    tmp_path: Path,
    fake_detector: FakeDetector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    marker = output_dir / "keep-me.txt"
    marker.write_text("user data", encoding="utf-8")
    load_calls = _install_fake_detector(monkeypatch, fake_detector)

    exit_code = cli.main(
        [
            "run",
            str(video_path),
            "--output",
            str(output_dir),
            "--warmup-runs",
            "0",
            "--max-frames",
            "0",
        ]
    )

    assert exit_code != 0
    assert load_calls == []
    assert capsys.readouterr().err.strip().splitlines() == [
        f"error: output directory must be empty: {output_dir.resolve()}"
    ]
    assert marker.read_text(encoding="utf-8") == "user data"
    assert list(output_dir.iterdir()) == [marker]


def test_run_command_reports_unavailable_requested_cuda_without_traceback(
    video_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_detector(monkeypatch, error=RuntimeError("CUDA was requested but is not available"))

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


def test_run_config_loads_selected_detector_once(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "experiment.json"
    write_run_config(config_path, RunConfig(video_path, tmp_path / "run", (), 0.3, 1, 0, 0))
    calls: list[tuple[str, float, str]] = []
    monkeypatch.setattr(
        cli,
        "load_detector",
        lambda detector_id, *, threshold, device: calls.append((detector_id, threshold, device))
        or object(),
    )
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda _config, _detector: {
            "status": "complete",
            "frames_processed": 1,
            "inference_count": 1,
        },
    )

    assert cli.main(["run", "--config", str(config_path)]) == 0
    assert calls == [("dfine-nano-coco", 0.3, "auto")]


def test_run_accepts_config_without_positional_input(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig(video_path, tmp_path / "run", (), 0.6, 2, 4, 8, device="cpu")
    config_path = tmp_path / "experiment.json"
    write_run_config(config_path, config)
    runner_calls: list[tuple[RunConfig, object]] = []
    detector = object()
    monkeypatch.setattr(cli, "load_detector", lambda *_args, **_kwargs: detector)
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda loaded, selected: runner_calls.append((loaded, selected))
        or {"status": "complete", "frames_processed": 1, "inference_count": 1},
    )

    assert cli.main(["run", "--config", str(config_path)]) == 0
    assert runner_calls == [(config, detector)]


def test_explicit_run_flags_keep_existing_defaults(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_calls: list[RunConfig] = []
    monkeypatch.setattr(cli, "load_detector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda config, _detector: config_calls.append(config)
        or {"status": "complete", "frames_processed": 1, "inference_count": 1},
    )

    assert cli.main(["run", str(video_path), "--output", str(tmp_path / "run")]) == 0
    assert config_calls == [
        RunConfig(video_path, tmp_path / "run", (), 0.3, None, 2, 10, device="auto")
    ]


def test_output_flag_overrides_config_output_only(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig(video_path, tmp_path / "configured-run", (), 0.6, 2, 4, 8, device="cpu")
    config_path = tmp_path / "experiment.json"
    output_dir = tmp_path / "overridden-run"
    write_run_config(config_path, config)
    config_calls: list[RunConfig] = []
    monkeypatch.setattr(cli, "load_detector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda loaded, _detector: config_calls.append(loaded)
        or {"status": "complete", "frames_processed": 1, "inference_count": 1},
    )

    assert cli.main(["run", "--config", str(config_path), "--output", str(output_dir)]) == 0
    assert config_calls == [replace(config, output_dir=output_dir)]


def test_present_flags_override_config_and_omitted_flags_do_not(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig(video_path, tmp_path / "run", (), 0.3, 1, 7, 9, device="auto")
    config_path = tmp_path / "experiment.json"
    write_run_config(config_path, config)
    config_calls: list[RunConfig] = []
    monkeypatch.setattr(cli, "load_detector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda loaded, _detector: config_calls.append(loaded)
        or {"status": "complete", "frames_processed": 1, "inference_count": 1},
    )

    assert (
        cli.main(
            [
                "run",
                "--config",
                str(config_path),
                "--threshold",
                "0.8",
                "--device",
                "cpu",
                "--max-frames",
                "2",
            ]
        )
        == 0
    )
    assert config_calls == [replace(config, threshold=0.8, device="cpu", max_frames=2)]


def test_run_requires_input_or_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run"]) == 2
    assert capsys.readouterr().err.strip().splitlines() == ["error: run requires INPUT or --config"]


def test_input_and_config_are_mutually_exclusive(
    tmp_path: Path,
    video_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["run", str(video_path), "--config", str(tmp_path / "experiment.json")]) == 2
    assert capsys.readouterr().err.strip().splitlines() == [
        "error: INPUT cannot be combined with --config"
    ]


def test_unknown_detector_id_fails_before_model_import(
    tmp_path: Path,
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "experiment.json"
    write_run_config(
        config_path,
        RunConfig(video_path, tmp_path / "run", (), 0.3, 1, 0, 0, detector_id="unknown"),
    )
    monkeypatch.delitem(sys.modules, "edge_perception.detectors.dfine", raising=False)

    assert cli.main(["run", "--config", str(config_path)]) == 2
    assert capsys.readouterr().err.strip().splitlines() == ["error: unknown detector ID: unknown"]
    assert "edge_perception.detectors.dfine" not in sys.modules


def test_malformed_config_is_one_line_and_model_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "malformed.json"
    config_path.write_text("{", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "edge_perception.detectors.dfine", raising=False)

    assert cli.main(["run", "--config", str(config_path)]) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1
    assert "edge_perception.detectors.dfine" not in sys.modules
