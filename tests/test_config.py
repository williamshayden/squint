from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_perception.config import (
    CaptureRequest,
    CaptureResult,
    RunConfig,
    load_run_config,
    render_run_cli,
    write_run_config,
)
from edge_perception.contracts import Region


def test_capture_request_treats_dimensions_and_fps_independently() -> None:
    assert CaptureRequest("camera-1", "EMEET", 1920, None, 15.0, False).requested_height is None
    assert CaptureRequest("camera-1", "EMEET", None, 1080, None, True).requested_width is None


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0.0, -1.0])
def test_capture_request_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="FPS"):
        CaptureRequest("camera-1", "EMEET", None, None, fps, False)


def test_capture_request_from_dict_requires_a_json_boolean_for_strict() -> None:
    payload = {
        "device_id": "camera-1",
        "device_description": "EMEET",
        "requested_width": 1920,
        "requested_height": None,
        "requested_fps": 30.0,
        "strict": 1,
    }

    with pytest.raises(TypeError, match="strict"):
        CaptureRequest.from_dict(payload)


def test_capture_result_from_dict_rejects_unknown_fields() -> None:
    payload = _capture_result_payload()
    payload["unexpected"] = "value"

    with pytest.raises(ValueError, match="unknown"):
        CaptureResult.from_dict(payload, base_dir=Path.cwd())


def test_load_run_config_resolves_relative_paths_from_config_parent(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "experiment.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "source": {
                    "path": "media/capture.mp4",
                    "capture": {
                        **_capture_result_payload(),
                        "path": "media/capture.mp4",
                    },
                },
                "output": {"directory": "runs/one"},
                "detector": {"id": "dfine-nano-coco", "device": "cuda", "threshold": 0.3},
                "regions": [],
                "execution": {"max_frames": None, "warmup_runs": 1, "annotate_every": 1},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_run_config(config_path)

    assert loaded.input_path == config_path.parent / "media" / "capture.mp4"
    assert loaded.output_dir == config_path.parent / "runs" / "one"
    assert loaded.capture is not None
    assert loaded.capture.path == config_path.parent / "media" / "capture.mp4"


def test_run_config_file_round_trips_capture_provenance(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video")
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 30.0, True)
    capture = CaptureResult(
        request=request,
        selected_width=1920,
        selected_height=1080,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=1920,
        actual_height=1080,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=5.0,
        has_audio=False,
        file_size_bytes=5,
        path=video,
        sha256="a" * 64,
    )
    config = RunConfig(
        input_path=video,
        output_dir=tmp_path / "run",
        regions=(Region("roi", 1, 2, 30, 40),),
        threshold=0.3,
        max_frames=3,
        warmup_runs=1,
        annotate_every=1,
        detector_id="dfine-nano-coco",
        device="cuda",
        capture=capture,
    )
    path = tmp_path / "experiment.json"

    write_run_config(path, config)

    assert load_run_config(path) == config


def test_render_run_cli_returns_resolved_argument_tuple(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    output = tmp_path / "run"

    assert render_run_cli(config_path, output) == (
        "edge-perception",
        "run",
        "--config",
        str(config_path.resolve()),
        "--output",
        str(output.resolve()),
    )


def _capture_result_payload() -> dict[str, object]:
    return {
        "request": {
            "device_id": "camera-1",
            "device_description": "EMEET",
            "requested_width": 1920,
            "requested_height": 1080,
            "requested_fps": 30.0,
            "strict": True,
        },
        "selected_width": 1920,
        "selected_height": 1080,
        "selected_min_fps": 30.0,
        "selected_max_fps": 30.0,
        "selected_pixel_format": "NV12",
        "actual_width": 1920,
        "actual_height": 1080,
        "actual_fps": 30.0,
        "container": "mp4",
        "codec": "h264",
        "duration_seconds": 5.0,
        "has_audio": False,
        "file_size_bytes": 5,
        "path": "capture.mp4",
        "sha256": "a" * 64,
    }
