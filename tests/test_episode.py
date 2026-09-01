from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st


def _episode_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.episode")
    except ModuleNotFoundError:
        return None


def _synthetic_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.synthetic")
    except ModuleNotFoundError:
        return None


def _episode_at(path: Path) -> object:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    return episode_module.Episode.open(synthetic_module.make_synthetic_episode(path))


def _reseal_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    arrays_path = path / "arrays.npz"
    np.savez(arrays_path, **arrays)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    artifacts["arrays.npz_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    artifacts.pop("content_sha256", None)
    normalized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    artifacts["content_sha256"] = hashlib.sha256(
        normalized + artifacts["arrays.npz_sha256"].encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_episode_modules_are_available() -> None:
    assert _episode_module() is not None, "Task 3 episode loader must exist"
    assert _synthetic_module() is not None, "Task 3 synthetic fixture builder must exist"


def test_generated_episode_round_trips_without_raw_media(tmp_path: Path) -> None:
    episode = _episode_at(tmp_path / "episode")

    assert episode.frame_count == 12
    assert episode.frame(0).timestamp_s == 0.0
    assert {item.name for item in episode.path.iterdir()} == {"manifest.json", "arrays.npz"}


def test_loader_rejects_hash_mismatch_before_frame_access(tmp_path: Path) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(
        tmp_path / "episode", frame_count=6, change_frames=(0, 4)
    )
    with np.load(path / "arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["timestamps_s"] = arrays["timestamps_s"].copy()
    arrays["timestamps_s"][1] += 0.01
    np.savez(path / "arrays.npz", **arrays)

    with pytest.raises(episode_module.EpisodeValidationError, match="arrays.npz sha256"):
        episode_module.Episode.open(path)


def test_loader_parses_the_verified_array_bytes_when_archive_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(tmp_path / "episode")
    arrays_path = path / "arrays.npz"
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(b"not an NPZ archive")
    original_read_bytes = Path.read_bytes

    def replace_after_read(candidate: Path) -> bytes:
        result = original_read_bytes(candidate)
        if candidate == arrays_path:
            os.replace(replacement, arrays_path)
        return result

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    assert episode_module.Episode.open(path).frame_count == 12


def test_loader_checks_arrays_hash_before_manifest_semantics(tmp_path: Path) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(tmp_path / "episode")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["fps"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (path / "arrays.npz").write_bytes(b"tampered")

    with pytest.raises(episode_module.EpisodeValidationError, match="arrays.npz sha256"):
        episode_module.Episode.open(path)


def _drop_detection_scores(arrays: dict[str, np.ndarray]) -> None:
    del arrays["det_scores"]


def _add_unexpected_array(arrays: dict[str, np.ndarray]) -> None:
    arrays["unexpected"] = np.empty(0, np.float32)


def _change_detection_score_dtype(arrays: dict[str, np.ndarray]) -> None:
    arrays["det_scores"] = arrays["det_scores"].astype(np.float64)


def _change_scene_shape(arrays: dict[str, np.ndarray]) -> None:
    arrays["scene_change"] = arrays["scene_change"][:, :, :2]


def _make_offsets_nonmonotonic(arrays: dict[str, np.ndarray]) -> None:
    arrays["det_frame_offsets"] = arrays["det_frame_offsets"].copy()
    arrays["det_frame_offsets"][1:3] = (2, 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_detection_scores, "array set mismatch"),
        (_add_unexpected_array, "array set mismatch"),
        (_change_detection_score_dtype, "det_scores must have dtype"),
        (_change_scene_shape, "scene_change must have shape"),
        (_make_offsets_nonmonotonic, "offsets must be monotonic"),
    ],
)
def test_loader_rejects_sealed_malformed_arrays(
    tmp_path: Path,
    mutate: Callable[[dict[str, np.ndarray]], None],
    message: str,
) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(tmp_path / "episode")
    with np.load(path / "arrays.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    mutate(arrays)
    _reseal_arrays(path, arrays)

    with pytest.raises(episode_module.EpisodeValidationError, match=message):
        episode_module.Episode.open(path)


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([1, 1, 1], np.int64),
        np.array([0, 2, 1], np.int64),
        np.array([0, 1, 3], np.int64),
    ],
)
def test_offsets_must_start_zero_be_monotonic_and_end_at_value_count(
    offsets: np.ndarray,
) -> None:
    episode_module = _episode_module()
    assert episode_module is not None, "Task 3 episode loader must exist"

    with pytest.raises(episode_module.EpisodeValidationError, match="offset"):
        episode_module.validate_offsets(
            "det_frame_offsets", offsets, frame_count=2, value_count=2
        )


@st.composite
def _valid_offsets(draw: st.DrawFn) -> tuple[int, int, np.ndarray]:
    frame_count = draw(st.integers(min_value=1, max_value=20))
    counts = draw(st.lists(st.integers(min_value=0, max_value=20), min_size=frame_count, max_size=frame_count))
    offsets = np.array([0, *np.cumsum(counts, dtype=np.int64)], dtype=np.int64)
    return frame_count, int(offsets[-1]), offsets


@given(_valid_offsets())
def test_offset_property_accepts_exactly_valid_non_decreasing_offsets(
    values: tuple[int, int, np.ndarray],
) -> None:
    episode_module = _episode_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    frame_count, value_count, offsets = values

    episode_module.validate_offsets(
        "det_frame_offsets", offsets, frame_count=frame_count, value_count=value_count
    )
    with pytest.raises(episode_module.EpisodeValidationError, match="length"):
        episode_module.validate_offsets(
            "det_frame_offsets", offsets[:-1], frame_count=frame_count, value_count=value_count
        )
    wrong_final = offsets.copy()
    wrong_final[-1] += 1
    with pytest.raises(episode_module.EpisodeValidationError, match="final offset"):
        episode_module.validate_offsets(
            "det_frame_offsets", wrong_final, frame_count=frame_count, value_count=value_count
        )


def test_episode_arrays_scene_and_nested_manifest_are_irreversibly_immutable(tmp_path: Path) -> None:
    episode = _episode_at(tmp_path / "episode")

    with pytest.raises(TypeError):
        episode.arrays["timestamps_s"] = np.array([], np.float64)
    for array in (episode.arrays["timestamps_s"], episode.frame(0).scene_change):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        episode.manifest["source"] = {}
    with pytest.raises(TypeError):
        episode.manifest["source"]["frame_count"] = 1


def test_replay_frame_batches_are_irreversibly_immutable(tmp_path: Path) -> None:
    frame = _episode_at(tmp_path / "episode").frame(0)

    for batch in (frame.detections, frame.ground_truth):
        for field in batch.__dataclass_fields__:
            array = getattr(batch, field)
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE"):
                array.setflags(write=True)


def test_slice_rebases_index_without_copying_arrays_or_timestamps(tmp_path: Path) -> None:
    episode = _episode_at(tmp_path / "episode")
    view = episode.slice(3, 7)

    assert view.parent is episode
    assert view.frame_count == 4
    assert view.frame(0).index == 0
    assert view.frame(0).timestamp_s == episode.frame(3).timestamp_s
    assert view.frame(3).timestamp_s == episode.frame(6).timestamp_s
    expected_hash = hashlib.sha256(f"{episode.content_sha256}:3:7".encode()).hexdigest()
    assert view.content_sha256 == expected_hash
    for start, stop in ((-1, 1), (0, 0), (4, 4), (0, episode.frame_count + 1)):
        with pytest.raises(ValueError, match="nonempty in-range"):
            episode.slice(start, stop)


def test_manifest_content_hash_rejects_tampering(tmp_path: Path) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(tmp_path / "episode")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["fps"] = 3.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(episode_module.EpisodeValidationError, match="content_sha256"):
        episode_module.Episode.open(path)


def test_loader_checks_schema_before_other_required_manifest_objects(tmp_path: Path) -> None:
    episode_module = _episode_module()
    synthetic_module = _synthetic_module()
    assert episode_module is not None, "Task 3 episode loader must exist"
    assert synthetic_module is not None, "Task 3 synthetic fixture builder must exist"
    path = synthetic_module.make_synthetic_episode(tmp_path / "episode")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"]["name"] = "unsupported"
    del manifest["source"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(episode_module.EpisodeValidationError, match="unsupported schema name"):
        episode_module.Episode.open(path)
