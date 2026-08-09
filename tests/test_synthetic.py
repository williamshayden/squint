from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _synthetic_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.synthetic")
    except ModuleNotFoundError:
        return None


def _episode_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.episode")
    except ModuleNotFoundError:
        return None


def test_synthetic_fixture_module_is_available() -> None:
    assert _synthetic_module() is not None, "Task 3 synthetic fixture builder must exist"


def test_synthetic_fixture_has_fixed_geometry_and_fixture_metadata(tmp_path: Path) -> None:
    synthetic_module = _synthetic_module()
    episode_module = _episode_module()
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    assert episode_module is not None, "Task 3 episode loader must exist"
    path = synthetic_module.make_synthetic_episode(
        tmp_path / "episode", frame_count=5, change_frames=(0, 3)
    )
    episode = episode_module.Episode.open(path)

    assert episode.manifest["schema"] == {"name": "squint.replay", "version": 1}
    assert episode.manifest["detector"]["id"] == "synthetic"
    assert episode.manifest["hardware"]["id"] == "synthetic"
    assert episode.manifest["cost_profile"]["p95_ms"] == 10.0
    assert episode.manifest["cost_profile"]["reserve_ms"] == 10.0
    assert episode.manifest["cost_profile"]["capacity_ms"] == 20.0
    assert episode.manifest["normalization"]["active_tracks"] == 8
    assert episode.manifest["normalization"]["age_s"] == 5
    assert episode.manifest["normalization"]["motion_px_s"] == 20
    assert episode.manifest["normalization"]["time_since_detector_s"] == 5
    assert episode.manifest["telemetry"] == {
        "latency_mean_ms": None,
        "gpu_utilization_percent": None,
        "vram_bytes": None,
    }
    for index in range(5):
        frame = episode.frame(index)
        np.testing.assert_array_equal(frame.detections.boxes_xyxy, [[10 + 2 * index, 10, 30 + 2 * index, 40]])
        np.testing.assert_array_equal(frame.ground_truth.boxes_xyxy, [[10 + 2 * index, 10, 30 + 2 * index, 40], [70, 70, 80, 90]])
        np.testing.assert_array_equal(frame.ground_truth.track_ids, [7, 99])
        np.testing.assert_array_equal(frame.ground_truth.class_ids, [1, 7])
        np.testing.assert_array_equal(frame.ground_truth.valid, [True, False])
        np.testing.assert_array_equal(frame.ground_truth.ignore, [False, True])
        np.testing.assert_array_equal(frame.scene_change, np.full((3, 3), index in (0, 3), np.float32))


def test_synthetic_fixture_sealing_is_deterministic(tmp_path: Path) -> None:
    synthetic_module = _synthetic_module()
    episode_module = _episode_module()
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    assert episode_module is not None, "Task 3 episode loader must exist"

    first = episode_module.Episode.open(synthetic_module.make_synthetic_episode(tmp_path / "first"))
    second = episode_module.Episode.open(synthetic_module.make_synthetic_episode(tmp_path / "second"))

    assert first.content_sha256 == second.content_sha256
    assert (first.path / "arrays.npz").read_bytes() == (second.path / "arrays.npz").read_bytes()


@pytest.mark.parametrize("change_frames", [(), (1,), (0, 5), (-1, 0)])
def test_synthetic_fixture_requires_zero_change_frame_in_range(
    tmp_path: Path, change_frames: tuple[int, ...]
) -> None:
    synthetic_module = _synthetic_module()
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"

    with pytest.raises(ValueError, match="change_frames"):
        synthetic_module.make_synthetic_episode(
            tmp_path / "episode", frame_count=5, change_frames=change_frames
        )
