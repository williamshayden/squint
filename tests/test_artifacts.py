from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from squint_rl.tracker import GroundTruthBatch


class _GroundTruthEpisode:
    def __init__(self, ground_truth: GroundTruthBatch) -> None:
        self._ground_truth = ground_truth
        self.frame_count = 1

    def frame(self, index: int) -> object:
        assert index == 0
        return type("Frame", (), {"ground_truth": self._ground_truth})()


def test_atomic_run_publishes_json_only_after_clean_exit(tmp_path: Path) -> None:
    from squint_rl.artifacts import AtomicRun, write_json

    destination = tmp_path / "run"
    with AtomicRun(destination) as work:
        write_json(work / "result.json", {"b": 2, "a": 1})

    assert destination.joinpath("result.json").read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'


def test_atomic_run_serializes_concurrent_publishers_without_overwriting_winner(
    tmp_path: Path,
) -> None:
    from squint_rl.artifacts import AtomicRun, write_json

    destination = tmp_path / "run"
    first_entered = Event()
    release_first = Event()
    outcome_lock = Lock()
    outcomes: dict[str, str | OSError] = {}

    def publish(label: str) -> None:
        try:
            with AtomicRun(destination) as work:
                write_json(work / "result.json", {"publisher": label})
                if label == "first":
                    first_entered.set()
                    if not release_first.wait(timeout=15):
                        raise TimeoutError("first publisher was not released")
            outcome: str | OSError = "success"
        except OSError as error:
            outcome = error
        with outcome_lock:
            outcomes[label] = outcome

    first = Thread(target=publish, args=("first",))
    first.start()
    assert first_entered.wait(timeout=5)
    assert not destination.exists()

    second = Thread(target=publish, args=("second",))
    second.start()
    second.join(timeout=5)
    second_finished_while_first_staged = not second.is_alive()

    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_finished_while_first_staged
    assert not first.is_alive()
    assert not second.is_alive()
    assert outcomes["first"] == "success"
    assert isinstance(outcomes["second"], FileExistsError)
    assert destination.joinpath("result.json").read_bytes() == (
        b'{\n  "publisher": "first"\n}\n'
    )
    assert {item.name for item in tmp_path.iterdir()} == {"run"}


def test_atomic_run_retains_interrupted_work(tmp_path: Path) -> None:
    from squint_rl.artifacts import AtomicRun, write_json

    destination = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt), AtomicRun(destination) as work:
        write_json(work / "result.json", {})
        raise KeyboardInterrupt

    assert not destination.exists()
    incomplete = list(tmp_path.glob(".run.*.incomplete"))
    assert len(incomplete) == 1
    assert incomplete[0].joinpath("result.json").exists()

    with AtomicRun(destination) as work:
        write_json(work / "result.json", {"retry": True})
    assert destination.joinpath("result.json").exists()
    assert incomplete[0].joinpath("result.json").exists()


def test_atomic_run_rejects_existing_destination_and_broken_symlink(tmp_path: Path) -> None:
    from squint_rl.artifacts import AtomicRun

    destination = tmp_path / "run"
    destination.mkdir()
    with pytest.raises(FileExistsError), AtomicRun(destination):
        pass

    destination.rmdir()
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError), AtomicRun(destination):
        pass


def test_atomic_run_retains_work_when_publish_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from squint_rl import artifacts

    destination = tmp_path / "run"
    real_replace = artifacts.os.replace

    def fail_publish(source: object, target: object) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failed"), artifacts.AtomicRun(destination) as work:
        artifacts.write_json(work / "result.json", {})

    assert not destination.exists()
    incomplete = list(tmp_path.glob(".run.*.incomplete"))
    assert len(incomplete) == 1

    monkeypatch.setattr(artifacts.os, "replace", real_replace)
    with artifacts.AtomicRun(destination) as work:
        artifacts.write_json(work / "result.json", {"retry": True})
    assert destination.joinpath("result.json").exists()
    assert incomplete[0].joinpath("result.json").exists()


def test_write_json_is_canonical_and_rejects_nonfinite_values(tmp_path: Path) -> None:
    from squint_rl.artifacts import write_json

    path = tmp_path / "result.json"
    write_json(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    with pytest.raises(ValueError, match="Out of range"):
        write_json(path, {"value": float("nan")})


def test_write_mot_tracks_uses_one_based_coordinates_ids_and_sorted_rows(tmp_path: Path) -> None:
    from squint_rl.artifacts import write_mot_tracks
    from squint_rl.tracker import TrackBatch

    frames = (
        TrackBatch(
            np.array([[4, 5, 14, 25], [0, 1, 2, 4]], dtype=np.float32),
            np.array([4, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([0.25, 0.5], dtype=np.float32),
        ),
        TrackBatch(
            np.array([[1, 2, 3, 5]], dtype=np.float32),
            np.array([2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0.75], dtype=np.float32),
        ),
    )
    path = tmp_path / "tracks.txt"

    digest = write_mot_tracks(path, frames)

    expected = (
        b"1,1,1.000000,2.000000,2.000000,3.000000,0.500000,1,-1,-1\n"
        b"1,5,5.000000,6.000000,10.000000,20.000000,0.250000,1,-1,-1\n"
        b"2,3,2.000000,3.000000,2.000000,3.000000,0.750000,1,-1,-1\n"
    )
    assert path.read_bytes() == expected
    assert digest == sha256(expected).hexdigest()


def test_write_mot_tracks_rejects_nonpedestrian_output(tmp_path: Path) -> None:
    from squint_rl.artifacts import write_mot_tracks
    from squint_rl.tracker import TrackBatch

    tracks = TrackBatch(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([1], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="class 1"):
        write_mot_tracks(tmp_path / "tracks.txt", (tracks,))


def test_write_mot_ground_truth_keeps_only_supported_records(tmp_path: Path) -> None:
    from squint_rl.artifacts import write_mot_ground_truth
    from squint_rl.tracker import GroundTruthBatch

    truth = GroundTruthBatch(
        np.array([[4, 5, 14, 25], [0, 1, 2, 4], [9, 9, 10, 10]], dtype=np.float32),
        np.array([2, 3, 4], dtype=np.int64),
        np.array([1, 7, 1], dtype=np.int64),
        np.array([1, 0.5, 1], dtype=np.float32),
        np.array([True, False, False]),
        np.array([False, True, False]),
    )
    path = tmp_path / "gt.txt"

    digest = write_mot_ground_truth(path, _GroundTruthEpisode(truth))

    expected = (
        b"1,2,5.000000,6.000000,10.000000,20.000000,1,1,1.000000,-1\n"
        b"1,3,1.000000,2.000000,2.000000,3.000000,1,7,0.500000,-1\n"
    )
    assert path.read_bytes() == expected
    assert digest == sha256(expected).hexdigest()


@pytest.mark.parametrize(
    ("class_id", "valid", "ignore", "message"),
    [(7, True, False, "valid"), (3, False, True, "ignored"), (1, False, True, "ignored")],
)
def test_write_mot_ground_truth_rejects_inconsistent_classes(
    tmp_path: Path, class_id: int, valid: bool, ignore: bool, message: str
) -> None:
    from squint_rl.artifacts import write_mot_ground_truth
    from squint_rl.tracker import GroundTruthBatch

    truth = GroundTruthBatch(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([1], dtype=np.int64),
        np.array([class_id], dtype=np.int64),
        np.array([1], dtype=np.float32),
        np.array([valid]),
        np.array([ignore]),
    )
    with pytest.raises(ValueError, match=message):
        write_mot_ground_truth(tmp_path / "gt.txt", _GroundTruthEpisode(truth))


def test_write_mot_ground_truth_rejects_nonpositive_ids(tmp_path: Path) -> None:
    from squint_rl.artifacts import write_mot_ground_truth
    from squint_rl.tracker import GroundTruthBatch

    truth = GroundTruthBatch(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.float32),
        np.array([True]),
        np.array([False]),
    )
    with pytest.raises(ValueError, match="positive"):
        write_mot_ground_truth(tmp_path / "gt.txt", _GroundTruthEpisode(truth))


def test_write_curve_csv_sorts_full_rows_and_hashes_emitted_bytes(tmp_path: Path) -> None:
    from squint_rl.artifacts import CurveRow, write_curve_csv

    path = tmp_path / "curve.csv"
    digest = write_curve_csv(
        path,
        (
            CurveRow("b", 0.8, 0.2, 0.5),
            CurveRow("a", 0.8, 0.2, 0.6),
            CurveRow("a", 0.4, 0.2, 0.6),
        ),
    )
    expected = (
        b"policy_id,nominal_rate,realized_compute,hota\n"
        b"a,0.400000,0.200000000,0.600000000\n"
        b"a,0.800000,0.200000000,0.600000000\n"
        b"b,0.800000,0.200000000,0.500000000\n"
    )
    assert path.read_bytes() == expected
    assert digest == sha256(expected).hexdigest()


@pytest.mark.parametrize(
    ("nominal_rate", "realized_compute", "hota"),
    [
        (0.0, 0.1, 0.5),
        (1.1, 0.1, 0.5),
        (float("nan"), 0.1, 0.5),
        (0.5, float("inf"), 0.5),
        (0.5, -0.1, 0.5),
        (0.5, 0.1, float("nan")),
        (0.5, 0.1, 1.1),
    ],
)
def test_curve_row_rejects_invalid_numeric_values(
    nominal_rate: float, realized_compute: float, hota: float
) -> None:
    from squint_rl.artifacts import CurveRow

    with pytest.raises(ValueError):
        CurveRow("policy", nominal_rate, realized_compute, hota)
