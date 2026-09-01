"""Pinned ByteTrack adapter that retains prediction-only tracker output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
from numpy.typing import NDArray

from squint_rl.episode import Episode
from squint_rl.tracker import DetectionBatch, TrackBatch, TrackerSummary


@dataclass(slots=True)
class _TrackState:
    class_id: int
    confidence: float
    first_seen_s: float
    last_measurement_s: float
    previous_center: NDArray[np.float32] | None = None
    previous_output_s: float | None = None
    motion_px_s: float = 0.0


class ByteTrackAdapter:
    """Expose ByteTrack's confirmed Kalman predictions through Squint's protocol."""

    def __init__(
        self,
        *,
        frame_rate: float,
        backend: object | None = None,
        lost_track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
    ) -> None:
        if not math.isfinite(frame_rate) or frame_rate <= 0.0:
            raise ValueError("frame_rate must be finite and positive")
        if backend is None:
            from trackers import ByteTrackTracker

            backend = ByteTrackTracker(
                frame_rate=frame_rate,
                lost_track_buffer=lost_track_buffer,
                track_activation_threshold=track_activation_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                high_conf_det_threshold=high_conf_det_threshold,
            )
        self._backend: Any = backend
        self._frame_rate = frame_rate
        self._states: dict[int, _TrackState] = {}
        self._tracks = TrackBatch.empty()
        self._timestamp_s: float | None = None

    def reset(self) -> None:
        self._backend.reset()
        self._states.clear()
        self._tracks = TrackBatch.empty()
        self._timestamp_s = None

    def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
        """Advance ByteTrack and return its full confirmed prediction view."""
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._timestamp_s is not None and timestamp_s < self._timestamp_s:
            raise ValueError("timestamps must be monotonic")
        measurement = self._measurement(detections)
        returned = self._backend.update(measurement, timestamp=timestamp_s)
        self._remember_measurements(returned, timestamp_s)
        self._timestamp_s = timestamp_s
        self._tracks = self._read_predictions(timestamp_s)
        return self._tracks

    def summary(self) -> TrackerSummary:
        if self._timestamp_s is None or len(self._tracks) == 0:
            return TrackerSummary.empty()
        states = [self._states[int(track_id)] for track_id in self._tracks.track_ids]
        active = len(states)
        stale = sum(state.last_measurement_s < self._timestamp_s for state in states)
        return TrackerSummary(
            active_tracks=active,
            confirmed_tracks=active,
            stale_tracks=stale,
            mean_age_s=float(
                np.mean([self._timestamp_s - state.first_seen_s for state in states])
            ),
            mean_motion_px_s=float(np.mean([state.motion_px_s for state in states])),
            mean_confidence=float(np.mean([state.confidence for state in states])),
        )

    @staticmethod
    def _measurement(detections: DetectionBatch | None) -> sv.Detections:
        if detections is None:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=detections.boxes_xyxy.copy(),
            confidence=detections.scores.copy(),
            class_id=detections.class_ids.copy(),
        )

    def _remember_measurements(self, returned: sv.Detections, timestamp_s: float) -> None:
        tracker_ids = returned.tracker_id
        confidences = returned.confidence
        class_ids = returned.class_id
        if tracker_ids is None or confidences is None or class_ids is None:
            return
        for tracker_id, confidence, class_id in zip(
            tracker_ids, confidences, class_ids, strict=True
        ):
            identifier = int(tracker_id)
            if identifier < 0:
                continue
            state = self._states.get(identifier)
            if state is None:
                self._states[identifier] = _TrackState(
                    class_id=int(class_id),
                    confidence=float(confidence),
                    first_seen_s=timestamp_s,
                    last_measurement_s=timestamp_s,
                )
            else:
                state.class_id = int(class_id)
                state.confidence = float(confidence)
                state.last_measurement_s = timestamp_s

    def _read_predictions(self, timestamp_s: float) -> TrackBatch:
        tracked = self._backend.tracked_objects
        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            return TrackBatch.empty()
        items = sorted(
            (
                (int(identifier), np.asarray(box, dtype=np.float32))
                for identifier, box in zip(tracker_ids, tracked.xyxy, strict=True)
                if int(identifier) >= 0
            ),
            key=lambda item: item[0],
        )
        if not items:
            return TrackBatch.empty()

        boxes: list[NDArray[np.float32]] = []
        class_ids: list[int] = []
        confidences: list[float] = []
        identifiers: list[int] = []
        for identifier, box in items:
            state = self._states.get(identifier)
            if state is None:
                state = _TrackState(
                    class_id=1,
                    confidence=0.0,
                    first_seen_s=timestamp_s,
                    last_measurement_s=timestamp_s,
                )
                self._states[identifier] = state
            center = np.asarray(
                ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0), dtype=np.float32
            )
            if state.previous_center is not None and state.previous_output_s is not None:
                elapsed = timestamp_s - state.previous_output_s
                if elapsed > 0.0:
                    state.motion_px_s = float(np.linalg.norm(center - state.previous_center) / elapsed)
            state.previous_center = center
            state.previous_output_s = timestamp_s
            boxes.append(box)
            identifiers.append(identifier)
            class_ids.append(state.class_id)
            confidences.append(state.confidence)
        return TrackBatch(
            np.asarray(boxes, dtype=np.float32),
            np.asarray(identifiers, dtype=np.int64),
            np.asarray(class_ids, dtype=np.int64),
            np.asarray(confidences, dtype=np.float32),
        )


def tracker_factory(*, episode: Episode, **parameters: Any) -> ByteTrackAdapter:
    """Build the pinned ByteTrack reference tracker for one replay episode."""
    return ByteTrackAdapter(frame_rate=episode.fps, **parameters)
