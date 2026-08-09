from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .episode import SCHEMA_NAME, SCHEMA_VERSION, seal_episode


def _validate_parameters(
    *, frame_count: int, fps: float, change_frames: tuple[int, ...], latency_ms: float
) -> None:
    if frame_count < 2 or fps <= 0 or latency_ms <= 0:
        raise ValueError("synthetic episode requires frame_count >= 2, fps > 0, and latency_ms > 0")
    if 0 not in change_frames or any(frame not in range(frame_count) for frame in change_frames):
        raise ValueError("change_frames must include zero and remain within the episode")


def synthetic_manifest(
    *, frame_count: int, fps: float, change_frames: tuple[int, ...], latency_ms: float
) -> dict[str, object]:
    _validate_parameters(
        frame_count=frame_count, fps=fps, change_frames=change_frames, latency_ms=latency_ms
    )
    literal_parameters = json.dumps(
        {
            "change_frames": change_frames,
            "fps": fps,
            "frame_count": frame_count,
            "latency_ms": latency_ms,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    source_hash = sha256(literal_parameters.encode()).hexdigest()
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "episode": {"id": "synthetic"},
        "source": {
            "id": "synthetic",
            "sha256": source_hash,
            "width": 100,
            "height": 100,
            "frame_count": frame_count,
            "fps": fps,
            "duration_s": (frame_count - 1) / fps,
            "dataset": "synthetic",
            "split": "fixture",
            "class_mapping": {"1": "valid", "7": "ignored"},
            "ignore_region_rules": "ignored identities are explicit records",
        },
        "detector": {
            "id": "synthetic",
            "family": "synthetic",
            "model_id": "synthetic",
            "revision": "synthetic",
            "weights_sha256": source_hash,
            "threshold": 0.0,
            "input_size": [100, 100],
            "precision": "float32",
            "backend": "numpy",
        },
        "hardware": {
            "id": "synthetic",
            "accelerator_backend": "synthetic",
            "driver_version": "synthetic",
            "runtime_version": "synthetic",
            "timing_protocol": "synthetic deterministic fixture",
        },
        "cost_profile": {
            "unit": "detector_ms",
            "p95_ms": latency_ms,
            "reserve_ms": latency_ms,
            "capacity_ms": 2.0 * latency_ms,
        },
        "scene_feature": {"name": "synthetic_scene_change", "shape": [3, 3]},
        "normalization": {
            "active_tracks": 8,
            "age_s": 5,
            "motion_px_s": 20,
            "time_since_detector_s": 5,
        },
        "telemetry": {
            "latency_mean_ms": None,
            "gpu_utilization_percent": None,
            "vram_bytes": None,
        },
        "artifacts": {},
    }


def make_synthetic_episode(
    path: str | Path,
    *,
    frame_count: int = 12,
    fps: float = 2.0,
    change_frames: tuple[int, ...] = (0, 4, 8),
    latency_ms: float = 10.0,
) -> Path:
    _validate_parameters(
        frame_count=frame_count, fps=fps, change_frames=change_frames, latency_ms=latency_ms
    )
    valid_boxes = np.array(
        [[10 + 2 * frame, 10, 30 + 2 * frame, 40] for frame in range(frame_count)],
        np.float32,
    )
    ignored_boxes = np.tile(np.array([[70, 70, 80, 90]], np.float32), (frame_count, 1))
    gt_boxes = np.stack((valid_boxes, ignored_boxes), axis=1).reshape(-1, 4)
    scene = np.zeros((frame_count, 3, 3), np.float32)
    scene[list(change_frames)] = 1.0
    arrays: dict[str, NDArray[Any]] = {
        "timestamps_s": np.arange(frame_count, dtype=np.float64) / fps,
        "detector_latency_ms": np.full(frame_count, latency_ms, np.float32),
        "scene_change": scene,
        "det_boxes_xyxy": valid_boxes,
        "det_scores": np.full(frame_count, 0.9, np.float32),
        "det_class_ids": np.ones(frame_count, np.int64),
        "det_frame_offsets": np.arange(frame_count + 1, dtype=np.int64),
        "gt_boxes_xyxy": gt_boxes,
        "gt_track_ids": np.tile(np.array([7, 99], np.int64), frame_count),
        "gt_class_ids": np.tile(np.array([1, 7], np.int64), frame_count),
        "gt_visibility": np.ones(frame_count * 2, np.float32),
        "gt_valid": np.tile(np.array([True, False]), frame_count),
        "gt_ignore": np.tile(np.array([False, True]), frame_count),
        "gt_frame_offsets": np.arange(0, 2 * frame_count + 1, 2, dtype=np.int64),
    }
    manifest = synthetic_manifest(
        frame_count=frame_count,
        fps=fps,
        change_frames=change_frames,
        latency_ms=latency_ms,
    )
    return seal_episode(path, manifest=manifest, arrays=arrays)
