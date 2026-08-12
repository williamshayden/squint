import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_cold_import_does_not_load_training_dependencies() -> None:
    code = (
        "import research.e4, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'stable_baselines3' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _e4():
    from research import e4

    return e4


def _expected_action_hash(actions: tuple[int, ...]) -> str:
    payload = json.dumps({"actions": list(actions)}, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"squint.e4.actions.v1\0" + payload).hexdigest()


def test_stage_a_defines_only_private_apis() -> None:
    tree = ast.parse(Path(__file__).parents[2].joinpath("research/e4.py").read_text())
    definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    assert definitions
    assert all(name.startswith("_") for name in definitions)
    assert "run_e4" not in definitions


def test_fixture_has_literal_manifest_and_all_replay_contracts(tmp_path: Path) -> None:
    from squint_rl.episode import Episode

    path = _e4()._build_episode(tmp_path / "episode")
    episode = Episode.open(path)
    assert episode.frame_count == 64
    assert episode.fps == 4.0
    assert set(episode.arrays) == {
        "timestamps_s", "detector_latency_ms", "scene_change",
        "det_boxes_xyxy", "det_scores", "det_class_ids", "det_frame_offsets",
        "gt_boxes_xyxy", "gt_track_ids", "gt_class_ids", "gt_visibility",
        "gt_valid", "gt_ignore", "gt_frame_offsets",
    }
    expected_shapes = {
        "timestamps_s": ((64,), np.float64),
        "detector_latency_ms": ((64,), np.float32),
        "scene_change": ((64, 3, 3), np.float32),
        "det_boxes_xyxy": ((58, 4), np.float32),
        "det_scores": ((58,), np.float32),
        "det_class_ids": ((58,), np.int64),
        "det_frame_offsets": ((65,), np.int64),
        "gt_boxes_xyxy": ((58, 4), np.float32),
        "gt_track_ids": ((58,), np.int64),
        "gt_class_ids": ((58,), np.int64),
        "gt_visibility": ((58,), np.float32),
        "gt_valid": ((58,), np.bool_),
        "gt_ignore": ((58,), np.bool_),
        "gt_frame_offsets": ((65,), np.int64),
    }
    for name, (shape, dtype) in expected_shapes.items():
        assert episode.arrays[name].shape == shape
        assert episode.arrays[name].dtype == dtype

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    record = source["record"]
    assert record == {
        "box_a": [10.0, 10.0, 30.0, 30.0],
        "box_b": [70.0, 70.0, 90.0, 90.0],
        "class_id": 1,
        "budget_reserve_ms": 10.0,
        "detector_latency_ms": 10.0,
        "detector_score": 0.9,
        "evaluation_view": {"pulses": [3, 11, 17, 21, 29], "start": 32, "stop": 64},
        "fps": 4.0,
        "frame_count": 64,
        "height": 100,
        "identity": 7,
        "ignore_rule": "no ignored records",
        "scene_feature_shape": [3, 3],
        "train_view": {"pulses": [3, 7, 13, 21, 25, 31], "start": 0, "stop": 32},
        "width": 100,
    }
    record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert source["sha256"] == hashlib.sha256(record_bytes).hexdigest()
    assert source["class_mapping"] == {"1": "target"}
    assert source["ignore_region_rules"] == "no ignored records"
    assert manifest["episode"] == {"id": "squint.e4.change-pulse.v1"}
    assert manifest["artifacts"]["arrays.npz_sha256"]
    assert manifest["artifacts"]["content_sha256"] == episode.content_sha256

    expected_pulses = (3, 7, 13, 21, 25, 31, 35, 43, 49, 53, 61)
    expected_boxes = []
    for frame in range(64):
        if frame < 3 or 32 <= frame < 35:
            expected_boxes.append(None)
        else:
            pulse_index = sum(frame >= pulse for pulse in expected_pulses)
            expected_boxes.append((10.0, 10.0, 30.0, 30.0) if pulse_index % 2 else
                                  (70.0, 70.0, 90.0, 90.0))
    expected_boxes[3] = (10.0, 10.0, 30.0, 30.0)
    expected_boxes[35] = (10.0, 10.0, 30.0, 30.0)
    assert np.array_equal(episode.arrays["timestamps_s"], np.arange(64, dtype=np.float64) / 4.0)
    assert np.array_equal(episode.arrays["detector_latency_ms"], np.full(64, 10.0, np.float32))
    expected_counts = np.array([0 if box is None else 1 for box in expected_boxes], np.int64)
    expected_offsets = np.concatenate(([0], np.cumsum(expected_counts, dtype=np.int64)))
    assert np.array_equal(episode.arrays["det_frame_offsets"], expected_offsets)
    assert np.array_equal(episode.arrays["gt_frame_offsets"], episode.arrays["det_frame_offsets"])
    assert np.array_equal(episode.arrays["det_scores"], np.full(58, 0.9, np.float32))
    assert np.array_equal(episode.arrays["det_class_ids"], np.ones(58, np.int64))
    assert np.array_equal(episode.arrays["gt_track_ids"], np.full(58, 7, np.int64))
    assert np.array_equal(episode.arrays["gt_class_ids"], np.ones(58, np.int64))
    assert np.array_equal(episode.arrays["gt_visibility"], np.ones(58, np.float32))
    assert np.array_equal(episode.arrays["gt_valid"], np.ones(58, np.bool_))
    assert np.array_equal(episode.arrays["gt_ignore"], np.zeros(58, np.bool_))
    assert np.array_equal(episode.arrays["scene_change"].sum(axis=(1, 2)),
                          np.array([9 if frame in expected_pulses else 0 for frame in range(64)], np.float32))
    assert [episode.frame(i).detections.boxes_xyxy.tolist() for i in range(64)] == [
        [] if box is None else [list(box)] for box in expected_boxes
    ]
    assert [episode.frame(i).ground_truth.track_ids.tolist() for i in range(64)] == [
        [] if box is None else [7] for box in expected_boxes
    ]
    assert [episode.frame(i).ground_truth.boxes_xyxy.tolist() for i in range(64)] == [
        [] if box is None else [list(box)] for box in expected_boxes
    ]


def test_fixture_build_is_byte_deterministic(tmp_path: Path) -> None:
    from squint_rl.episode import Episode

    first = _e4()._build_episode(tmp_path / "first")
    second = _e4()._build_episode(tmp_path / "second")
    assert Episode.open(first).content_sha256 == Episode.open(second).content_sha256
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()
    first_episode = Episode.open(first)
    train = first_episode.slice(0, 32)
    evaluation = first_episode.slice(32, 64)
    assert (train.start, train.stop, train.frame_count) == (0, 32, 32)
    assert (evaluation.start, evaluation.stop, evaluation.frame_count) == (32, 64, 32)
    assert train.frame(0).index == 0 and train.frame(0).timestamp_s == 0.0
    assert evaluation.frame(0).index == 0 and evaluation.frame(0).timestamp_s == 8.0
    assert len(train.frame(3).ground_truth) == 1
    assert len(evaluation.frame(3).ground_truth) == 1
    assert len(evaluation.frame(0).ground_truth) == 0


def test_hold_last_tracker_lifecycle_identity_and_summary() -> None:
    from squint_rl.tracker import DetectionBatch

    e4 = _e4()
    tracker = e4._HoldLastTracker()
    assert tracker.summary().active_tracks == 0
    assert len(tracker.step(None, 0.0)) == 0
    measurement = DetectionBatch(
        np.array([[10, 10, 30, 30]], np.float32), np.array([0.9], np.float32), np.array([1], np.int64)
    )
    first = tracker.step(measurement, 0.25)
    assert first.track_ids.dtype == np.int64 and first.track_ids.tolist() == [1]
    assert first.class_ids.tolist() == [1] and np.isclose(first.scores[0], 0.9)
    retained = tracker.step(None, 0.5)
    assert np.array_equal(retained.boxes_xyxy, first.boxes_xyxy)
    summary = tracker.summary()
    assert (summary.active_tracks, summary.confirmed_tracks, summary.stale_tracks) == (1, 1, 0)
    assert (summary.mean_age_s, summary.mean_motion_px_s) == (0.0, 0.0)
    assert np.isclose(summary.mean_confidence, 0.9)
    replacement = DetectionBatch(
        np.array([[70, 70, 90, 90]], np.float32), np.array([0.8], np.float32), np.array([1], np.int64)
    )
    replaced = tracker.step(replacement, 0.75)
    assert replaced.boxes_xyxy.tolist() == [[70.0, 70.0, 90.0, 90.0]]
    assert np.isclose(replaced.scores[0], 0.8)
    assert len(tracker.step(DetectionBatch.empty(), 1.0)) == 0
    assert tracker.summary().active_tracks == 0
    identity = {
        "id": "squint.e4.tracker.v1",
        "implementation": "hold-last-v1",
        "measurement_replaces_state": True,
        "skip_retains_state": True,
        "track_id": 1,
    }
    assert e4._tracker_identity() == identity
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert e4._tracker_hash() == hashlib.sha256(
        b"squint.e4.tracker.v1\0" + identity_bytes
    ).hexdigest()


def test_action_hash_has_literal_hand_derived_fixture() -> None:
    assert _e4()._action_hash((0, 1, 0, 1)) == "5713b8549a6b975dd71f2f90d6ebbda5b4e20e01fe4076287e65737f15ca0589"


def test_six_real_environment_rollouts_are_literal_and_exact(tmp_path: Path) -> None:
    from squint_rl.episode import Episode

    episode = Episode.open(_e4()._build_episode(tmp_path / "episode"))
    rollouts = _e4()._run_rollouts(episode.slice(32, 64))
    assert [(item["policy"], item["rho"]) for item in rollouts] == [
        ("greedy-affordable-v1", 0.25), ("periodic-v1", 0.25), ("scene-change-v1", 0.25),
        ("greedy-affordable-v1", 0.5), ("periodic-v1", 0.5), ("scene-change-v1", 0.5),
    ]
    schedules = [
        (0, 4, 8, 12, 16, 20, 24, 28), (0, 4, 8, 12, 16, 20, 24, 28), (3, 11, 17, 21, 29),
        (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
        (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), (3, 11, 17, 21, 29),
    ]
    totals = (11.0, 11.0, 32.0, 23.0, 23.0, 32.0)
    means = (0.34375, 0.34375, 1.0, 0.71875, 0.71875, 1.0)
    for item, schedule, total, mean in zip(rollouts, schedules, totals, means, strict=True):
        expected_actions = [1 if frame in schedule else 0 for frame in range(32)]
        assert item["requested_actions"] == expected_actions
        assert item["applied_actions"] == expected_actions
        assert item["requested_frames"] == list(schedule)
        assert item["applied_frames"] == list(schedule)
        assert item["denied_frames"] == []
        assert len(item["rewards"]) == 32
        assert all(np.isfinite(item["rewards"]))
        assert item["total_reward"] == total
        assert item["mean_reward"] == mean
        assert item["terminated"] is True and item["truncated"] is False
        assert item["action_sha256"] == _expected_action_hash(tuple(expected_actions))
