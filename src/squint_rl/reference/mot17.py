from __future__ import annotations

import configparser
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import numpy as np

from squint_rl.tracker import FloatArray, GroundTruthBatch

TRAIN: Final[tuple[str, ...]] = ("02", "04", "05", "10")
VALIDATION: Final[tuple[str, ...]] = ("09",)
TEST: Final[tuple[str, ...]] = ("11", "13")
DISTRACTOR_CLASSES: Final[frozenset[int]] = frozenset({2, 7, 8, 12})
_PARTITIONS: Final[dict[str, tuple[str, ...]]] = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}
_IDENTIFIERS: Final[frozenset[str]] = frozenset(TRAIN + VALIDATION + TEST)
_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_EXTENSION = re.compile(r"^\.[A-Za-z0-9]+$")
_REQUIRED_SEQUENCE_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "imDir",
    "frameRate",
    "seqLength",
    "imWidth",
    "imHeight",
    "imExt",
)


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


class _Mot17ErrorMixin:
    sequence: str
    path: Path
    line: int | None
    field: str | None
    value: str | None

    def _set_context(
        self,
        *,
        sequence: str,
        path: Path,
        line: int | None,
        field: str | None,
        value: str | None,
    ) -> None:
        self.sequence = sequence
        self.path = path
        self.line = line
        self.field = field
        self.value = value


class Mot17FormatError(_Mot17ErrorMixin, ValueError):
    """A MOT17 file violates the strict replay-import contract."""

    def __init__(
        self,
        message: str,
        *,
        sequence: str,
        path: Path,
        line: int | None = None,
        field: str | None = None,
        value: str | None = None,
    ) -> None:
        self._set_context(
            sequence=sequence, path=path, line=line, field=field, value=value
        )
        context = _context(sequence, path, line, field, value)
        super().__init__(f"{message} ({context})")


class Mot17MissingFileError(_Mot17ErrorMixin, FileNotFoundError):
    """A required MOT17 file or exact source frame is absent."""

    def __init__(
        self,
        message: str,
        *,
        sequence: str,
        path: Path,
        line: int | None = None,
        field: str | None = None,
        value: str | None = None,
    ) -> None:
        self._set_context(
            sequence=sequence, path=path, line=line, field=field, value=value
        )
        context = _context(sequence, path, line, field, value)
        super().__init__(f"{message} ({context})")


@dataclass(frozen=True, slots=True)
class Mot17Sequence:
    identifier: str
    source_dir: Path
    image_paths: tuple[Path, ...]
    width: int
    height: int
    fps: float
    ground_truth: tuple[GroundTruthBatch, ...]


@dataclass(frozen=True, slots=True)
class _GroundTruthRow:
    identity: int
    class_id: int
    box: FloatArray
    visibility: float
    valid: bool
    ignore: bool


def _context(
    sequence: str,
    path: Path,
    line: int | None,
    field: str | None,
    value: str | None,
) -> str:
    parts = [f"sequence={sequence}", f"path={path}"]
    if line is not None:
        parts.append(f"line={line}")
    if field is not None:
        parts.append(f"field={field}")
    if value is not None:
        parts.append(f"value={value!r}")
    return ", ".join(parts)


def sequence_ids(partition: str) -> tuple[str, ...]:
    try:
        return _PARTITIONS[partition]
    except KeyError as error:
        raise ValueError("partition must be train, validation, or test") from error


def _format(
    message: str,
    *,
    sequence: str,
    path: Path,
    line: int | None = None,
    field: str | None = None,
    value: str | None = None,
) -> Mot17FormatError:
    return Mot17FormatError(
        message,
        sequence=sequence,
        path=path,
        line=line,
        field=field,
        value=value,
    )


def _missing(
    message: str,
    *,
    sequence: str,
    path: Path,
    line: int | None = None,
    field: str | None = None,
    value: str | None = None,
) -> Mot17MissingFileError:
    return Mot17MissingFileError(
        message,
        sequence=sequence,
        path=path,
        line=line,
        field=field,
        value=value,
    )


def _parse_integer(
    raw: str,
    *,
    sequence: str,
    path: Path,
    line: int | None,
    field: str,
) -> int:
    if not _DECIMAL_INTEGER.fullmatch(raw):
        raise _format(
            "expected a strict decimal integer",
            sequence=sequence,
            path=path,
            line=line,
            field=field,
            value=raw,
        )
    try:
        return int(raw, 10)
    except ValueError as error:  # pragma: no cover - guarded by the regex
        raise _format(
            "invalid decimal integer",
            sequence=sequence,
            path=path,
            line=line,
            field=field,
            value=raw,
        ) from error


def _parse_float(
    raw: str,
    *,
    sequence: str,
    path: Path,
    line: int,
    field: str,
) -> float:
    if raw == "":
        raise _format(
            "expected a finite decimal value",
            sequence=sequence,
            path=path,
            line=line,
            field=field,
            value=raw,
        )
    try:
        value = float(raw)
    except ValueError as error:
        raise _format(
            "expected a finite decimal value",
            sequence=sequence,
            path=path,
            line=line,
            field=field,
            value=raw,
        ) from error
    if not math.isfinite(value):
        raise _format(
            "value must be finite",
            sequence=sequence,
            path=path,
            line=line,
            field=field,
            value=raw,
        )
    return value


def _read_sequence_info(
    source_dir: Path, identifier: str
) -> tuple[int, int, int, float, Path, str]:
    path = source_dir / "seqinfo.ini"
    if not path.is_file():
        raise _missing(
            "missing seqinfo.ini",
            sequence=identifier,
            path=path,
        )
    parser = _CaseSensitiveConfigParser(interpolation=None, strict=True)
    try:
        with path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
    except FileNotFoundError as error:
        raise _missing(
            "missing seqinfo.ini",
            sequence=identifier,
            path=path,
        ) from error
    except (configparser.Error, OSError, UnicodeError) as error:
        raise _format(
            "invalid seqinfo.ini",
            sequence=identifier,
            path=path,
        ) from error
    if parser.sections() != ["Sequence"]:
        raise _format(
            "seqinfo.ini must contain exactly one [Sequence] section",
            sequence=identifier,
            path=path,
        )
    defaults = parser.defaults()
    if defaults:
        raise _format(
            "seqinfo.ini must not define inherited DEFAULT values",
            sequence=identifier,
            path=path,
            field="DEFAULT",
            value=",".join(sorted(defaults)),
        )
    info = parser["Sequence"]
    for field in _REQUIRED_SEQUENCE_FIELDS:
        if field not in info:
            raise _format(
                "missing required sequence field",
                sequence=identifier,
                path=path,
                field=field,
            )
    name = info["name"]
    expected_name = f"MOT17-{identifier}-FRCNN"
    if name != expected_name:
        raise _format(
            "sequence name does not match canonical FRCNN identifier",
            sequence=identifier,
            path=path,
            field="name",
            value=name,
        )
    image_dir_value = info["imDir"]
    posix_dir = PurePosixPath(image_dir_value)
    windows_dir = PureWindowsPath(image_dir_value)
    if (
        not image_dir_value
        or posix_dir.is_absolute()
        or windows_dir.is_absolute()
        or bool(windows_dir.drive)
        or bool(windows_dir.root)
        or ".." in posix_dir.parts
        or ".." in windows_dir.parts
    ):
        raise _format(
            "imDir must be a non-empty relative path",
            sequence=identifier,
            path=path,
            field="imDir",
            value=image_dir_value,
        )
    image_dir = source_dir / Path(image_dir_value)
    try:
        image_dir.relative_to(source_dir)
        source_dir_resolved = source_dir.resolve(strict=False)
        image_dir_resolved = image_dir.resolve(strict=False)
        image_dir_resolved.relative_to(source_dir_resolved)
    except (ValueError, RuntimeError, OSError) as error:
        raise _format(
            "imDir must remain contained by the sequence directory",
            sequence=identifier,
            path=path,
            field="imDir",
            value=image_dir_value,
        ) from error
    if image_dir_resolved == source_dir_resolved:
        raise _format(
            "imDir must name a contained image directory",
            sequence=identifier,
            path=path,
            field="imDir",
            value=image_dir_value,
        )
    try:
        image_dir_exists = image_dir.is_dir()
    except FileNotFoundError as error:
        raise _missing(
            "missing image directory",
            sequence=identifier,
            path=image_dir,
            field="imDir",
            value=image_dir_value,
        ) from error
    except (RuntimeError, OSError) as error:
        raise _format(
            "unable to inspect image directory",
            sequence=identifier,
            path=image_dir,
            field="imDir",
            value=image_dir_value,
        ) from error
    if not image_dir_exists:
        raise _missing(
            "missing image directory",
            sequence=identifier,
            path=image_dir,
            field="imDir",
            value=image_dir_value,
        )
    extension = info["imExt"]
    if not _EXTENSION.fullmatch(extension):
        raise _format(
            "imExt must be a simple dot-prefixed extension",
            sequence=identifier,
            path=path,
            field="imExt",
            value=extension,
        )
    sequence_length = _parse_positive_metadata_integer(info["seqLength"], identifier, path, "seqLength")
    width = _parse_positive_metadata_integer(info["imWidth"], identifier, path, "imWidth")
    height = _parse_positive_metadata_integer(info["imHeight"], identifier, path, "imHeight")
    fps = _parse_metadata_float(info["frameRate"], identifier, path, "frameRate")
    if fps <= 0:
        raise _format(
            "frameRate must be positive",
            sequence=identifier,
            path=path,
            field="frameRate",
            value=info["frameRate"],
        )
    return sequence_length, width, height, fps, image_dir, extension


def _parse_positive_metadata_integer(raw: str, sequence: str, path: Path, field: str) -> int:
    value = _parse_integer(raw, sequence=sequence, path=path, line=None, field=field)
    if value <= 0:
        raise _format(
            "metadata integer must be positive",
            sequence=sequence,
            path=path,
            field=field,
            value=raw,
        )
    return value


def _parse_metadata_float(raw: str, sequence: str, path: Path, field: str) -> float:
    value = _parse_float(raw, sequence=sequence, path=path, line=0, field=field)
    return value


def _read_images(
    image_dir: Path,
    *,
    sequence: str,
    length: int,
    extension: str,
) -> tuple[Path, ...]:
    expected_names = {f"{frame:06d}{extension}" for frame in range(1, length + 1)}
    try:
        entries = tuple(image_dir.iterdir())
    except FileNotFoundError as error:
        raise _missing(
            "missing image directory",
            sequence=sequence,
            path=image_dir,
            field="imDir",
        ) from error
    except (RuntimeError, OSError) as error:
        raise _format(
            "unable to enumerate image directory",
            sequence=sequence,
            path=image_dir,
            field="imDir",
        ) from error
    actual_names = {entry.name for entry in entries}
    extras = sorted(actual_names - expected_names)
    if extras:
        extra = image_dir / extras[0]
        raise _format(
            "image directory contains an unexpected entry",
            sequence=sequence,
            path=extra,
            field="images",
            value=extras[0],
        )
    missing = sorted(expected_names - actual_names)
    if missing:
        missing_path = image_dir / missing[0]
        raise _missing(
            "missing expected source frame",
            sequence=sequence,
            path=missing_path,
            field="images",
            value=missing[0],
        )
    for entry in entries:
        try:
            is_file = entry.is_file()
        except FileNotFoundError as error:
            raise _missing(
                "missing expected source frame",
                sequence=sequence,
                path=entry,
                field="images",
                value=entry.name,
            ) from error
        except (RuntimeError, OSError) as error:
            raise _format(
                "unable to inspect source image",
                sequence=sequence,
                path=entry,
                field="images",
                value=entry.name,
            ) from error
        if not is_file:
            raise _format(
                "source image inventory entry is not a file",
                sequence=sequence,
                path=entry,
                field="images",
                value=entry.name,
            )
    return tuple(image_dir / f"{frame:06d}{extension}" for frame in range(1, length + 1))


def _read_ground_truth(
    path: Path,
    *,
    identifier: str,
    length: int,
    frame_width: int,
    frame_height: int,
) -> tuple[GroundTruthBatch, ...]:
    try:
        gt_exists = path.is_file()
    except (RuntimeError, OSError) as error:
        raise _format(
            "unable to inspect gt.txt",
            sequence=identifier,
            path=path,
            field="gt",
        ) from error
    if not gt_exists:
        raise _missing("missing gt.txt", sequence=identifier, path=path)
    rows: list[list[_GroundTruthRow]] = [[] for _ in range(length)]
    seen: dict[tuple[int, int], int] = {}
    last_line = 0
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            for row in reader:
                line = reader.line_num
                last_line = line
                if len(row) != 9:
                    raise _format(
                        "gt.txt rows must contain exactly 9 columns",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="columns",
                        value=str(len(row)),
                    )
                frame = _parse_integer(
                    row[0], sequence=identifier, path=path, line=line, field="frame"
                )
                identity = _parse_integer(
                    row[1], sequence=identifier, path=path, line=line, field="identity"
                )
                mark = _parse_integer(
                    row[6], sequence=identifier, path=path, line=line, field="mark"
                )
                class_id = _parse_integer(
                    row[7], sequence=identifier, path=path, line=line, field="class_id"
                )
                if not 1 <= frame <= length:
                    raise _format(
                        "frame is outside seqLength",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="frame",
                        value=row[0],
                    )
                if identity < 1:
                    raise _format(
                        "identity must be positive",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="identity",
                        value=row[1],
                    )
                if identity > _INT64_MAX:
                    raise _format(
                        "identity exceeds the int64 output representation",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="identity",
                        value=row[1],
                    )
                if mark not in (0, 1):
                    raise _format(
                        "mark must be 0 or 1",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="mark",
                        value=row[6],
                    )
                if not 1 <= class_id <= 13:
                    raise _format(
                        "class_id must be in the MOT17 catalog 1..13",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="class_id",
                        value=row[7],
                    )
                x = _parse_float(row[2], sequence=identifier, path=path, line=line, field="x")
                y = _parse_float(row[3], sequence=identifier, path=path, line=line, field="y")
                width = _parse_float(row[4], sequence=identifier, path=path, line=line, field="width")
                height = _parse_float(row[5], sequence=identifier, path=path, line=line, field="height")
                visibility = _parse_float(
                    row[8], sequence=identifier, path=path, line=line, field="visibility"
                )
                if width <= 0 or height <= 0:
                    raise _format(
                        "box width and height must be positive",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="geometry",
                        value=f"{row[4]},{row[5]}",
                    )
                if not 0 <= visibility <= 1:
                    raise _format(
                        "visibility must be in [0, 1]",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="visibility",
                        value=row[8],
                    )
                x1, y1 = x - 1, y - 1
                x2, y2 = x1 + width, y1 + height
                if not 0 <= x1 < x2 <= frame_width:
                    raise _format(
                        "x geometry is outside source dimensions",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="geometry",
                        value=f"{row[2]},{row[4]}",
                    )
                if not 0 <= y1 < y2 <= frame_height:
                    raise _format(
                        "y geometry is outside source dimensions",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="geometry",
                        value=f"{row[3]},{row[5]}",
                    )
                key = (frame, identity)
                previous_line = seen.get(key)
                if previous_line is not None:
                    raise _format(
                        f"duplicate frame/identity at line {line}; first occurrence was line {previous_line}",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="identity",
                        value=f"{frame},{identity}; line {previous_line}",
                    )
                seen[key] = line
                with np.errstate(over="ignore", invalid="ignore"):
                    box = cast(
                        FloatArray,
                        np.array([x1, y1, x2, y2], dtype=np.float32),
                    )
                if not np.all(np.isfinite(box)) or not (
                    box[0] < box[2] and box[1] < box[3]
                ):
                    raise _format(
                        "geometry cannot be represented as a finite float32 box",
                        sequence=identifier,
                        path=path,
                        line=line,
                        field="geometry",
                        value=",".join(row[2:6]),
                    )
                rows[frame - 1].append(
                    _GroundTruthRow(
                        identity=identity,
                        class_id=class_id,
                        box=box,
                        visibility=visibility,
                        valid=mark == 1 and class_id == 1,
                        ignore=mark == 1 and class_id in DISTRACTOR_CLASSES,
                    )
                )
    except FileNotFoundError as error:
        raise _missing("missing gt.txt", sequence=identifier, path=path) from error
    except OSError as error:
        raise _format(
            "unable to read gt.txt",
            sequence=identifier,
            path=path,
            line=last_line + 1,
            field="gt",
        ) from error
    except csv.Error as error:
        raise _format(
            "invalid CSV in gt.txt",
            sequence=identifier,
            path=path,
            line=last_line + 1,
            field="gt",
        ) from error
    except UnicodeError as error:
        raise _format("gt.txt is not valid UTF-8", sequence=identifier, path=path) from error
    return tuple(_batch(frame_rows) for frame_rows in rows)


def _batch(rows: list[_GroundTruthRow]) -> GroundTruthBatch:
    if not rows:
        return GroundTruthBatch.empty()
    return GroundTruthBatch(
        boxes_xyxy=cast(FloatArray, np.stack([row.box for row in rows]).astype(np.float32)),
        track_ids=np.array([row.identity for row in rows], dtype=np.int64),
        class_ids=np.array([row.class_id for row in rows], dtype=np.int64),
        visibility=np.array([row.visibility for row in rows], dtype=np.float32),
        valid=np.array([row.valid for row in rows], dtype=np.bool_),
        ignore=np.array([row.ignore for row in rows], dtype=np.bool_),
    )


def load_sequence(root: str | Path, identifier: str) -> Mot17Sequence:
    if identifier not in _IDENTIFIERS or not re.fullmatch(r"[0-9]{2}", identifier):
        raise Mot17FormatError(
            "identifier must be one of the frozen canonical two-digit MOT17 IDs",
            sequence=identifier,
            path=Path(root),
            field="identifier",
            value=identifier,
        )
    source_dir = Path(root) / "train" / f"MOT17-{identifier}-FRCNN"
    length, width, height, fps, image_dir, extension = _read_sequence_info(
        source_dir, identifier
    )
    image_paths = _read_images(
        image_dir,
        sequence=identifier,
        length=length,
        extension=extension,
    )
    ground_truth = _read_ground_truth(
        source_dir / "gt" / "gt.txt",
        identifier=identifier,
        length=length,
        frame_width=width,
        frame_height=height,
    )
    return Mot17Sequence(
        identifier=identifier,
        source_dir=source_dir,
        image_paths=image_paths,
        width=width,
        height=height,
        fps=fps,
        ground_truth=ground_truth,
    )
