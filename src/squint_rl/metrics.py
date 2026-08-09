from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class MetricSummary:
    hota: float
    deta: float
    assa: float
    idf1: float
    false_positives: int
    false_negatives: int
    identity_switches: int

    def __post_init__(self) -> None:
        for name in ("hota", "deta", "assa", "idf1"):
            _unit_interval(float(getattr(self, name)), name)
        for name in ("false_positives", "false_negatives", "identity_switches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class MetricReport:
    combined: MetricSummary
    per_sequence: Mapping[str, MetricSummary]

    def __post_init__(self) -> None:
        if not isinstance(self.combined, MetricSummary):
            raise TypeError("combined must be a MetricSummary")
        values = dict(self.per_sequence)
        if not values:
            raise ValueError("per_sequence must be nonempty")
        if any(not isinstance(name, str) or not name or not isinstance(value, MetricSummary) for name, value in values.items()):
            raise ValueError("per_sequence must map nonempty names to MetricSummary values")
        object.__setattr__(self, "per_sequence", MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class CurvePoint:
    realized_compute: float
    hota: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.realized_compute) or self.realized_compute < 0.0:
            raise ValueError("realized_compute must be finite and nonnegative")
        _unit_interval(self.hota, "hota")


@dataclass(frozen=True, slots=True)
class CurveAreas:
    support: tuple[float, float]
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        lower, upper = self.support
        if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or lower >= upper:
            raise ValueError("support must be a finite, nonempty nonnegative interval")
        values = dict(self.values)
        if not values:
            raise ValueError("values must be nonempty")
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise ValueError("curve names must be nonempty strings")
            _unit_interval(float(value), f"{name} area")
        object.__setattr__(self, "values", MappingProxyType(values))


def common_support_areas(
    curves: Mapping[str, Sequence[CurvePoint]],
    *,
    grid_size: int = 101,
) -> CurveAreas:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not curves:
        raise ValueError("curves must be nonempty")
    ordered: dict[str, list[CurvePoint]] = {}
    for name, points in curves.items():
        if not isinstance(name, str) or not name:
            raise ValueError("curve names must be nonempty strings")
        if any(not isinstance(point, CurvePoint) for point in points):
            raise ValueError(f"{name} points must be CurvePoint records")
        copied = sorted(points, key=lambda point: point.realized_compute)
        if len(copied) < 2:
            raise ValueError(f"{name} must have at least two points")
        if len({point.realized_compute for point in copied}) != len(copied):
            raise ValueError(f"{name} has duplicate compute points")
        ordered[name] = copied
    lower = max(points[0].realized_compute for points in ordered.values())
    upper = min(points[-1].realized_compute for points in ordered.values())
    if lower >= upper:
        raise ValueError("curves have no common measured support")
    grid = np.linspace(lower, upper, grid_size)
    values: dict[str, float] = {}
    for name, points in ordered.items():
        compute = np.array([point.realized_compute for point in points], dtype=float)
        hota = np.array([point.hota for point in points], dtype=float)
        interpolated = np.interp(grid, compute, hota)
        values[name] = float(np.trapezoid(interpolated, grid) / (upper - lower))
    return CurveAreas((lower, upper), values)


def run_trackeval(
    *,
    gt_root: Path,
    trackers_root: Path,
    output_root: Path,
    sequence_lengths: Mapping[str, int],
) -> MetricReport:
    evaluator, dataset, metrics = _trackeval_objects(
        gt_root=gt_root,
        trackers_root=trackers_root,
        output_root=output_root,
        sequence_lengths=sequence_lengths,
    )
    raw, messages = evaluator.evaluate([dataset], metrics)
    dataset_name = dataset.get_name()
    tracker_message = messages[dataset_name]["squint"]
    if tracker_message != "Success":
        raise RuntimeError(str(tracker_message))
    tracker_result = raw[dataset_name]["squint"]
    combined = _metric_summary(tracker_result["COMBINED_SEQ"]["pedestrian"])
    per_sequence = {
        sequence: _metric_summary(tracker_result[sequence]["pedestrian"])
        for sequence in sorted(sequence_lengths)
    }
    return MetricReport(combined, per_sequence)


def _trackeval_objects(
    *,
    gt_root: Path,
    trackers_root: Path,
    output_root: Path,
    sequence_lengths: Mapping[str, int],
) -> tuple[Any, Any, list[Any]]:
    sequence_info = dict(sequence_lengths)
    if not sequence_info or any(
        not isinstance(name, str)
        or not name
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        for name, length in sequence_info.items()
    ):
        raise ValueError("sequence_lengths must map nonempty names to positive integers")
    import trackeval  # type: ignore[import-untyped]

    evaluator = trackeval.Evaluator(
        {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "LOG_ON_ERROR": None,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )
    dataset = trackeval.datasets.MotChallenge2DBox(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": ["squint"],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "eval",
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": sequence_info,
            "PRINT_CONFIG": False,
        }
    )
    metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return (
        evaluator,
        dataset,
        [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ],
    )


def _metric_summary(raw: Mapping[str, Mapping[str, Any]]) -> MetricSummary:
    summary = MetricSummary(
        hota=float(np.mean(raw["HOTA"]["HOTA"])),
        deta=float(np.mean(raw["HOTA"]["DetA"])),
        assa=float(np.mean(raw["HOTA"]["AssA"])),
        idf1=float(raw["Identity"]["IDF1"]),
        false_positives=int(raw["CLEAR"]["CLR_FP"]),
        false_negatives=int(raw["CLEAR"]["CLR_FN"]),
        identity_switches=int(raw["CLEAR"]["IDSW"]),
    )
    return summary
