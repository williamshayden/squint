from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

_ID = "squint.e4.change-pulse.v1"
_PULSES = (3, 7, 13, 21, 25, 31, 35, 43, 49, 53, 61)
_BOX_A: list[float] = [10.0, 10.0, 30.0, 30.0]
_BOX_B: list[float] = [70.0, 70.0, 90.0, 90.0]
_RECORD = {
    "box_a": _BOX_A, "box_b": _BOX_B,
    "budget_reserve_ms": 10.0, "class_id": 1, "detector_latency_ms": 10.0,
    "detector_score": 0.9,
    "evaluation_view": {"pulses": [3, 11, 17, 21, 29], "start": 32, "stop": 64},
    "fps": 4.0, "frame_count": 64, "height": 100, "identity": 7,
    "ignore_rule": "no ignored records", "scene_feature_shape": [3, 3],
    "train_view": {"pulses": [3, 7, 13, 21, 25, 31], "start": 0, "stop": 32},
    "width": 100,
}


def _compact(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _build_episode(path: str | Path) -> Path:
    from squint_rl.episode import seal_episode
    from squint_rl.synthetic import synthetic_manifest

    record = json.loads(_compact(_RECORD))
    source = {
        "id": _ID, "sha256": sha256(_compact(record)).hexdigest(), "record": record,
        "width": 100, "height": 100, "frame_count": 64, "fps": 4.0,
        "duration_s": 15.75, "dataset": "synthetic", "split": "e4-stage-a",
        "class_mapping": {"1": "target"}, "ignore_region_rules": "no ignored records",
    }
    manifest = synthetic_manifest(frame_count=64, fps=4.0, change_frames=(0,), latency_ms=10.0)
    manifest.update({
        "episode": {"id": _ID}, "source": source,
        "detector": {"id": "squint.e4.detector.v1", "family": "fixture", "model_id": "fixture",
                      "revision": "1", "weights_sha256": source["sha256"], "threshold": 0.0,
                      "input_size": [100, 100], "precision": "float32", "backend": "numpy"},
        "cost_profile": {"unit": "detector_ms", "p95_ms": 10.0, "reserve_ms": 10.0,
                         "capacity_ms": 20.0},
        "scene_feature": {"name": "change-pulse", "shape": [3, 3]},
        "normalization": {"active_tracks": 8, "age_s": 5, "motion_px_s": 20,
                           "time_since_detector_s": 5}, "artifacts": {},
    })
    boxes: list[list[float]] = []
    counts: list[int] = []
    scene = np.zeros((64, 3, 3), np.float32)
    for frame in range(64):
        empty = frame < 3 or 32 <= frame < 35
        if frame in _PULSES:
            scene[frame] = 1.0
        if empty:
            counts.append(0)
        else:
            boxes.append(_BOX_A if sum(frame >= pulse for pulse in _PULSES) % 2 else _BOX_B)
            counts.append(1)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    arrays: dict[str, Any] = {
        "timestamps_s": np.arange(64, dtype=np.float64) / 4.0,
        "detector_latency_ms": np.full(64, 10.0, np.float32), "scene_change": scene,
        "det_boxes_xyxy": box_array, "det_scores": np.full(len(boxes), 0.9, np.float32),
        "det_class_ids": np.ones(len(boxes), np.int64), "det_frame_offsets": offsets,
        "gt_boxes_xyxy": box_array.copy(), "gt_track_ids": np.full(len(boxes), 7, np.int64),
        "gt_class_ids": np.ones(len(boxes), np.int64), "gt_visibility": np.ones(len(boxes), np.float32),
        "gt_valid": np.ones(len(boxes), np.bool_), "gt_ignore": np.zeros(len(boxes), np.bool_),
        "gt_frame_offsets": offsets.copy(),
    }
    return seal_episode(path, manifest=manifest, arrays=arrays)


class _HoldLastTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        from squint_rl.tracker import TrackBatch

        self._tracks = TrackBatch.empty()

    def step(self, detections: Any, timestamp_s: float) -> Any:
        from squint_rl.tracker import TrackBatch

        del timestamp_s
        if detections is not None:
            if len(detections) > 1:
                raise ValueError("hold-last-v1 accepts at most one detection")
            self._tracks = TrackBatch(
                detections.boxes_xyxy, np.full(len(detections), 1, np.int64),
                detections.class_ids, detections.scores,
            )
        return self._tracks

    def summary(self) -> Any:
        from squint_rl.tracker import TrackerSummary

        confidence = float(np.mean(self._tracks.scores)) if len(self._tracks) else 0.0
        count = len(self._tracks)
        return TrackerSummary(count, count, 0, 0.0, 0.0, confidence)


def _tracker_identity() -> dict[str, object]:
    return {"id": "squint.e4.tracker.v1", "implementation": "hold-last-v1",
            "measurement_replaces_state": True, "skip_retains_state": True, "track_id": 1}


def _tracker_hash() -> str:
    return sha256(b"squint.e4.tracker.v1\0" + _compact(_tracker_identity())).hexdigest()


def _action_hash(actions: tuple[int, ...]) -> str:
    return sha256(b"squint.e4.actions.v1\0" + _compact({"actions": list(actions)})).hexdigest()


def _rollout(view: Any, rho: float, policy_name: str) -> dict[str, object]:
    from squint_rl.budget import BudgetConfig
    from squint_rl.env import RUN_DETECTOR, SKIP, SquintEnv
    from squint_rl.policies import GreedyAffordable, Periodic, Policy, SceneChange
    from squint_rl.tracker import ObservationScales

    policy: Policy
    if policy_name == "greedy-affordable-v1":
        policy = GreedyAffordable()
    elif policy_name == "periodic-v1":
        policy = Periodic(interval_s=1.0 / (rho * 4.0), elapsed_scale_s=5.0)
    elif policy_name == "scene-change-v1":
        policy = SceneChange(threshold=1.0)
    else:
        raise ValueError(policy_name)
    env = SquintEnv(episode=view, tracker=_HoldLastTracker(),
                    budget=BudgetConfig.for_rate(reserve_ms=10.0, source_fps=4.0, nominal_rate=rho),
                    observation_scales=ObservationScales(8.0, 5.0, 20.0, 5.0))
    observation, _ = env.reset(seed=0)
    requested: list[int] = []
    applied: list[int] = []
    requested_frames: list[int] = []
    applied_frames: list[int] = []
    denied_frames: list[int] = []
    rewards: list[float] = []
    terminated = truncated = False
    for frame in range(32):
        action = int(policy(observation))
        if action not in (SKIP, RUN_DETECTOR):
            raise ValueError("policy must return a binary action")
        observation, reward, terminated, truncated, info = env.step(action)
        requested.append(action); applied.append(int(info["applied_action"])); rewards.append(float(reward))
        if action: requested_frames.append(frame)
        if info["applied_action"]: applied_frames.append(frame)
        if info["denied"]: denied_frames.append(frame)
    return {"policy": policy_name, "rho": rho, "requested_actions": requested,
            "applied_actions": applied, "requested_frames": requested_frames,
            "applied_frames": applied_frames, "denied_frames": denied_frames,
            "rewards": rewards, "total_reward": sum(rewards), "mean_reward": sum(rewards) / 32.0,
            "terminated": terminated, "truncated": truncated,
            "action_sha256": _action_hash(tuple(requested))}


def _run_rollouts(view: Any) -> tuple[dict[str, object], ...]:
    return tuple(_rollout(view, rho, name) for rho, name in (
        (0.25, "greedy-affordable-v1"), (0.25, "periodic-v1"), (0.25, "scene-change-v1"),
        (0.5, "greedy-affordable-v1"), (0.5, "periodic-v1"), (0.5, "scene-change-v1")))
