from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Literal
from uuid import uuid4

from .episode import Episode
from .tracker import TrackBatch

_IGNORED_MOT17_CLASSES = frozenset({2, 7, 8, 12})


class AtomicRun:
    """Stage and publish without overwriting another AtomicRun publication."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination).absolute()
        owner = uuid4().hex
        self.working = self.destination.parent / (
            f".{self.destination.name}.{owner}.incomplete"
        )
        self._claim = self.destination.parent / f".{self.destination.name}.publish.lock"
        self._claim_marker = self._claim / owner
        self._owns_claim = False

    def __enter__(self) -> Path:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.destination):
            raise FileExistsError(f"completed run already exists: {self.destination}")
        self._acquire_claim()
        try:
            if os.path.lexists(self.destination):
                raise FileExistsError(f"completed run already exists: {self.destination}")
            self.working.mkdir()
        except BaseException:
            self._release_claim()
            raise
        return self.working

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_value, traceback
        try:
            if exc_type is not None:
                return False
            for item in sorted(path for path in self.working.rglob("*") if path.is_file()):
                with item.open("rb") as stream:
                    os.fsync(stream.fileno())
            if os.path.lexists(self.destination):
                raise FileExistsError(f"completed run already exists: {self.destination}")
            os.replace(self.working, self.destination)
            return False
        finally:
            self._release_claim()

    def _acquire_claim(self) -> None:
        try:
            self._claim.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"run publication already in progress: {self.destination}"
            ) from error
        try:
            self._claim_marker.touch(exist_ok=False)
        except BaseException:
            self._claim.rmdir()
            raise
        self._owns_claim = True

    def _release_claim(self) -> None:
        if not self._owns_claim:
            return
        try:
            try:
                self._claim_marker.unlink()
            except FileNotFoundError:
                return
            self._claim.rmdir()
        finally:
            self._owns_claim = False


@dataclass(frozen=True, slots=True)
class CurveRow:
    policy_id: str
    nominal_rate: float
    realized_compute: float
    hota: float

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be nonempty")
        if not math.isfinite(self.nominal_rate) or not 0.0 < self.nominal_rate <= 1.0:
            raise ValueError("nominal_rate must be finite and in (0, 1]")
        if not math.isfinite(self.realized_compute) or self.realized_compute < 0.0:
            raise ValueError("realized_compute must be finite and nonnegative")
        if not math.isfinite(self.hota) or not 0.0 <= self.hota <= 1.0:
            raise ValueError("hota must be finite and in [0, 1]")


def write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_hashed(path: Path, rows: Sequence[Sequence[str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    payload = output.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def write_mot_tracks(path: Path, frames: Sequence[TrackBatch]) -> str:
    rows: list[list[str]] = []
    for frame_index, tracks in enumerate(frames, start=1):
        for index in sorted(range(len(tracks)), key=lambda item: int(tracks.track_ids[item])):
            if int(tracks.class_ids[index]) != 1:
                raise ValueError("MOT17 tracker output must use pedestrian class 1")
            x1, y1, x2, y2 = tracks.boxes_xyxy[index]
            rows.append(
                [
                    str(frame_index),
                    str(int(tracks.track_ids[index]) + 1),
                    f"{x1 + 1:.6f}",
                    f"{y1 + 1:.6f}",
                    f"{x2 - x1:.6f}",
                    f"{y2 - y1:.6f}",
                    f"{tracks.scores[index]:.6f}",
                    "1",
                    "-1",
                    "-1",
                ]
            )
    return _write_hashed(path, rows)


def write_mot_ground_truth(path: Path, episode: Episode) -> str:
    rows: list[list[str]] = []
    for frame_index in range(episode.frame_count):
        truth = episode.frame(frame_index).ground_truth
        for index in sorted(range(len(truth)), key=lambda item: int(truth.track_ids[item])):
            valid = bool(truth.valid[index])
            ignored = bool(truth.ignore[index])
            if not valid and not ignored:
                continue
            class_id = int(truth.class_ids[index])
            if valid and class_id != 1:
                raise ValueError("valid MOT17 ground truth must use pedestrian class 1")
            if ignored and class_id not in _IGNORED_MOT17_CLASSES:
                raise ValueError("ignored MOT17 ground truth must use a supported distractor class")
            track_id = int(truth.track_ids[index])
            if track_id <= 0:
                raise ValueError("MOT ground-truth track IDs must be positive")
            x1, y1, x2, y2 = truth.boxes_xyxy[index]
            rows.append(
                [
                    str(frame_index + 1),
                    str(track_id),
                    f"{x1 + 1:.6f}",
                    f"{y1 + 1:.6f}",
                    f"{x2 - x1:.6f}",
                    f"{y2 - y1:.6f}",
                    "1",
                    str(class_id),
                    f"{truth.visibility[index]:.6f}",
                    "-1",
                ]
            )
    return _write_hashed(path, rows)


def write_curve_csv(path: Path, points: Sequence[CurveRow]) -> str:
    ordered = sorted(
        points,
        key=lambda point: (
            point.policy_id,
            point.nominal_rate,
            point.realized_compute,
            point.hota,
        ),
    )
    rows = [["policy_id", "nominal_rate", "realized_compute", "hota"]]
    rows.extend(
        [
            point.policy_id,
            f"{point.nominal_rate:.6f}",
            f"{point.realized_compute:.9f}",
            f"{point.hota:.9f}",
        ]
        for point in ordered
    )
    return _write_hashed(path, rows)
