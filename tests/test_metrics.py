from __future__ import annotations

from pathlib import Path

import pytest


def test_curve_area_uses_only_common_realized_support() -> None:
    from squint_rl.metrics import CurvePoint, common_support_areas

    curves = {
        "a": [CurvePoint(0.1, 0.2), CurvePoint(0.5, 0.6), CurvePoint(0.9, 0.8)],
        "b": [CurvePoint(0.2, 0.3), CurvePoint(0.6, 0.7), CurvePoint(0.8, 0.75)],
    }
    areas = common_support_areas(curves, grid_size=101)

    assert areas.support == pytest.approx((0.2, 0.8))
    assert areas.values["a"] == pytest.approx(0.5625, abs=1e-3)
    assert areas.values["b"] == pytest.approx(0.575, abs=1e-3)


def test_curve_area_sorts_copies_and_returns_immutable_values() -> None:
    from squint_rl.metrics import CurvePoint, common_support_areas

    first = [CurvePoint(1.0, 0.8), CurvePoint(0.0, 0.2)]
    second = [CurvePoint(0.0, 0.3), CurvePoint(1.0, 0.7)]
    areas = common_support_areas({"first": first, "second": second}, grid_size=2)

    assert first[0].realized_compute == 1.0
    assert areas.values["first"] == pytest.approx(0.5)
    with pytest.raises(TypeError):
        areas.values["first"] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    "curves",
    [
        {},
        {"empty": []},
        {"single": [object()]},
    ],
)
def test_curve_area_rejects_empty_or_short_curves(curves: object) -> None:
    from squint_rl.metrics import common_support_areas

    with pytest.raises(ValueError):
        common_support_areas(curves)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "points",
    [
        [(0.2, 0.3), (0.2, 0.4)],
        [(-0.1, 0.3), (0.2, 0.4)],
        [(float("nan"), 0.3), (0.2, 0.4)],
        [(0.1, float("inf")), (0.2, 0.4)],
        [(0.1, -0.01), (0.2, 0.4)],
        [(0.1, 0.3), (0.2, 1.01)],
    ],
)
def test_curve_area_rejects_invalid_points(points: list[tuple[float, float]]) -> None:
    from squint_rl.metrics import CurvePoint, common_support_areas

    with pytest.raises(ValueError):
        common_support_areas({"curve": [CurvePoint(compute, hota) for compute, hota in points]})


def test_metric_summary_rejects_invalid_values() -> None:
    from squint_rl.metrics import MetricSummary

    with pytest.raises(ValueError):
        MetricSummary(1.1, 1.0, 1.0, 1.0, 0, 0, 0)
    with pytest.raises(ValueError):
        MetricSummary(1.0, 1.0, 1.0, 1.0, -1, 0, 0)


def test_curve_area_rejects_invalid_grid_and_degenerate_common_support() -> None:
    from squint_rl.metrics import CurvePoint, common_support_areas

    curves = {
        "first": [CurvePoint(0.0, 0.2), CurvePoint(1.0, 0.8)],
        "second": [CurvePoint(1.0, 0.3), CurvePoint(2.0, 0.7)],
    }
    with pytest.raises(ValueError, match="grid_size"):
        common_support_areas({"first": curves["first"]}, grid_size=1)
    with pytest.raises(ValueError, match="common"):
        common_support_areas(curves)


def test_run_trackeval_reports_perfect_cpu_only_tracks(tmp_path: Path) -> None:
    from squint_rl.metrics import run_trackeval

    sequence = "MOT17-perfect"
    gt = tmp_path / "gt" / sequence / "gt" / "gt.txt"
    tracks = tmp_path / "trackers" / "squint" / "data" / f"{sequence}.txt"
    gt.parent.mkdir(parents=True)
    tracks.parent.mkdir(parents=True)
    gt.write_text(
        "1,1,1,1,10,10,1,1,1,-1\n2,1,2,1,10,10,1,1,1,-1\n",
        encoding="utf-8",
        newline="\n",
    )
    tracks.write_text(
        "1,1,1,1,10,10,1,1,-1,-1\n2,1,2,1,10,10,1,1,-1,-1\n",
        encoding="utf-8",
        newline="\n",
    )

    output_root = tmp_path / "output"
    report = run_trackeval(
        gt_root=tmp_path / "gt",
        trackers_root=tmp_path / "trackers",
        output_root=output_root,
        sequence_lengths={sequence: 2},
    )

    for summary in (report.combined, report.per_sequence[sequence]):
        assert summary.hota == pytest.approx(1.0)
        assert summary.deta == pytest.approx(1.0)
        assert summary.assa == pytest.approx(1.0)
        assert summary.idf1 == pytest.approx(1.0)
        assert summary.false_positives == 0
        assert summary.false_negatives == 0
        assert summary.identity_switches == 0
    with pytest.raises(TypeError):
        report.per_sequence[sequence] = report.combined  # type: ignore[index]
    assert not output_root.exists()
