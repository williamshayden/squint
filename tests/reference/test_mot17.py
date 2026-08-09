from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squint_rl.reference.mot17 import (
    Mot17FormatError,
    Mot17MissingFileError,
    load_sequence,
    sequence_ids,
)


def _write_sequence(
    root: Path,
    *,
    identifier: str = "02",
    length: int = 3,
    rows: tuple[str, ...] = (
        "1,1,10,20,30,40,1,1,1",
        "1,2,10,20,30,40,1,7,0.5",
        "1,3,10,20,30,40,0,1,0",
    ),
    width: int = 100,
    height: int = 80,
    image_extension: str = ".jpg",
    image_dir: str = "img1",
) -> Path:
    source = root / "train" / f"MOT17-{identifier}-FRCNN"
    images = source / image_dir
    images.mkdir(parents=True, exist_ok=True)
    (source / "gt").mkdir(parents=True, exist_ok=True)
    (source / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name=MOT17-{identifier}-FRCNN\n"
        f"imDir={image_dir}\n"
        "frameRate=30\n"
        f"seqLength={length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        f"imExt={image_extension}\n",
        encoding="utf-8",
    )
    for frame in range(1, length + 1):
        (images / f"{frame:06d}{image_extension}").write_bytes(b"frame")
    (source / "gt" / "gt.txt").write_text(
        "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    return source


@pytest.fixture
def mot17_fixture(tmp_path: Path) -> Path:
    _write_sequence(tmp_path)
    return tmp_path


def test_whole_scene_partitions_are_frozen() -> None:
    assert sequence_ids("train") == ("02", "04", "05", "10")
    assert sequence_ids("validation") == ("09",)
    assert sequence_ids("test") == ("11", "13")
    partitions = (sequence_ids("train"), sequence_ids("validation"), sequence_ids("test"))
    assert all(set(left).isdisjoint(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :])
    with pytest.raises(ValueError, match="partition"):
        sequence_ids("official-test")


def test_only_canonical_frcnn_copy_is_read(mot17_fixture: Path) -> None:
    train = mot17_fixture / "train"
    for variant in ("DPM", "SDP"):
        poisoned = train / f"MOT17-02-{variant}"
        (poisoned / "img1").mkdir(parents=True)
        (poisoned / "seqinfo.ini").write_text("not a valid sequence", encoding="utf-8")
    sequence = load_sequence(mot17_fixture, "02")
    assert sequence.source_dir.name == "MOT17-02-FRCNN"


def test_parser_converts_coordinates_and_preserves_provenance(mot17_fixture: Path) -> None:
    sequence = load_sequence(mot17_fixture, "02")
    frame = sequence.ground_truth[0]
    np.testing.assert_array_equal(frame.boxes_xyxy[0], [9, 19, 39, 59])
    assert frame.valid.tolist() == [True, False, False]
    assert frame.ignore.tolist() == [False, True, False]
    assert frame.class_ids.tolist() == [1, 7, 1]
    np.testing.assert_array_equal(frame.track_ids, [1, 2, 3])
    np.testing.assert_array_equal(frame.visibility, [1, 0.5, 0])
    assert len(sequence.ground_truth[1]) == 0


def test_lower_right_one_pixel_box_is_valid(tmp_path: Path) -> None:
    _write_sequence(
        tmp_path,
        length=1,
        width=100,
        height=80,
        rows=("1,1,100,80,1,1,1,1,1",),
    )
    frame = load_sequence(tmp_path, "02").ground_truth[0]
    np.testing.assert_array_equal(frame.boxes_xyxy[0], [99, 79, 100, 80])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame", "1.0"),
        ("identity", "1e0"),
        ("mark", ""),
        ("class_id", "nan"),
    ],
)
def test_gt_integer_fields_are_strict_decimal(
    mot17_fixture: Path, field: str, value: str
) -> None:
    values = ["1", "1", "10", "20", "30", "40", "1", "1", "1"]
    index = {"frame": 0, "identity": 1, "mark": 6, "class_id": 7}[field]
    values[index] = value
    gt = mot17_fixture / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt"
    gt.write_text(",".join(values) + "\n", encoding="utf-8")
    with pytest.raises(Mot17FormatError, match=field):
        load_sequence(mot17_fixture, "02")


@pytest.mark.parametrize(
    ("values", "field"),
    [
        (("0", "1", "10", "20", "30", "40", "1", "1", "1"), "frame"),
        (("1", "0", "10", "20", "30", "40", "1", "1", "1"), "identity"),
        (("1", "1", "10", "20", "30", "40", "2", "1", "1"), "mark"),
        (("1", "1", "10", "20", "30", "40", "1", "14", "1"), "class_id"),
        (("1", "1", "10", "20", "30", "40", "1", "1", "1.1"), "visibility"),
    ],
)
def test_gt_semantics_are_strict(mot17_fixture: Path, values: tuple[str, ...], field: str) -> None:
    gt = mot17_fixture / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt"
    gt.write_text(",".join(values) + "\n", encoding="utf-8")
    with pytest.raises(Mot17FormatError, match=field):
        load_sequence(mot17_fixture, "02")


@pytest.mark.parametrize(
    "row",
    [
        "1,1,0,20,30,40,1,1,1",
        "1,1,10,20,0,40,1,1,1",
        "1,1,10,20,30,40,1,1,inf",
        "1,1,10,20,30,40,1,1,nan",
        "1,1,10,20,30,40,1,1,0.1,extra",
    ],
)
def test_geometry_visibility_and_column_count_are_strict(mot17_fixture: Path, row: str) -> None:
    gt = mot17_fixture / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt"
    gt.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(Mot17FormatError):
        load_sequence(mot17_fixture, "02")


def test_duplicate_frame_identity_reports_both_source_lines(mot17_fixture: Path) -> None:
    gt = mot17_fixture / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt"
    gt.write_text(
        "1,4,10,20,30,40,1,1,1\n1,4,11,20,30,40,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(Mot17FormatError, match=r"line 1.*line 2|line 2.*line 1"):
        load_sequence(mot17_fixture, "02")


@pytest.mark.parametrize(
    "kind",
    ("missing", "extra", "wrong-extension", "directory"),
)
def test_image_inventory_is_exact(mot17_fixture: Path, kind: str) -> None:
    images = mot17_fixture / "train" / "MOT17-02-FRCNN" / "img1"
    if kind == "missing":
        (images / "000002.jpg").unlink()
    elif kind == "extra":
        (images / "000004.jpg").write_bytes(b"extra")
    elif kind == "wrong-extension":
        (images / "000002.png").write_bytes(b"wrong")
    else:
        (images / "000002.jpg").unlink()
        (images / "000002.jpg").mkdir()
    with pytest.raises(Mot17MissingFileError if kind == "missing" else Mot17FormatError):
        load_sequence(mot17_fixture, "02")


def test_missing_sequence_and_gt_files_have_context(tmp_path: Path) -> None:
    source = _write_sequence(tmp_path)
    (source / "gt" / "gt.txt").unlink()
    with pytest.raises(Mot17MissingFileError, match=r"MOT17-02-FRCNN.*gt.txt"):
        load_sequence(tmp_path, "02")
    source.joinpath("gt").mkdir(exist_ok=True)
    (source / "seqinfo.ini").unlink()
    with pytest.raises(Mot17MissingFileError, match=r"seqinfo.ini"):
        load_sequence(tmp_path, "02")


def test_metadata_is_strict_and_paths_are_contained(tmp_path: Path) -> None:
    source = _write_sequence(tmp_path)
    seqinfo = source / "seqinfo.ini"
    original = seqinfo.read_text(encoding="utf-8")
    for old, replacement, field in (
        ("name=MOT17-02-FRCNN", "name=MOT17-04-FRCNN", "name"),
        ("imDir=img1", "imDir=/tmp", "imDir"),
        ("imExt=.jpg", "imExt=jpg", "imExt"),
    ):
        seqinfo.write_text(original.replace(old, replacement), encoding="utf-8")
        with pytest.raises(Mot17FormatError, match=field):
            load_sequence(tmp_path, "02")
        seqinfo.write_text(original, encoding="utf-8")


def test_noncanonical_identifiers_are_rejected(tmp_path: Path) -> None:
    _write_sequence(tmp_path)
    with pytest.raises(Mot17FormatError, match="identifier"):
        load_sequence(tmp_path, "2")
    with pytest.raises(Mot17FormatError, match="identifier"):
        load_sequence(tmp_path, "02-FRCNN")
