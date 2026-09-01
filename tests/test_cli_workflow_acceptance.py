from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import FakeDetector

from edge_perception import cli
from edge_perception.video import iter_video, probe_video


def _write_three_frame_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 200
        stream.height = 100
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)

        for frame_index, red_value in enumerate((20, 120, 220)):
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[..., 0] = red_value
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_local_video_run_inspect_rerun_compare_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video_path = tmp_path / "three-frames.mp4"
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_three_frame_video(video_path)
    metadata = probe_video(video_path)
    assert metadata.width == 200
    assert metadata.height == 100
    assert metadata.average_fps == pytest.approx(30.0)
    decoded_frames = list(iter_video(video_path))
    assert [frame.frame_index for frame in decoded_frames] == [0, 1, 2]
    assert [frame.image.shape for frame in decoded_frames] == [
        (100, 200, 3),
        (100, 200, 3),
        (100, 200, 3),
    ]
    monkeypatch.setattr(
        cli,
        "load_detector",
        lambda _detector_id, *, threshold, device: FakeDetector(),
    )
    run_arguments = [
        "run",
        str(video_path),
        "--detector",
        "dfine-nano-coco",
        "--device",
        "cpu",
        "--max-frames",
        "3",
        "--warmup-runs",
        "0",
        "--annotate-every",
        "1",
        "--crop",
        "left:0,0,100,100",
        "--crop",
        "right:100,0,100,100",
    ]

    assert cli.main([*run_arguments[:2], "--output", str(run_a), *run_arguments[2:]]) == 0
    capsys.readouterr()

    assert cli.main(["inspect", str(run_a)]) == 0
    inspection = capsys.readouterr()
    assert inspection.err == ""
    assert "Run\n  status: Completed" in inspection.out
    assert "Metrics\n  frames processed: 3" in inspection.out
    assert "  inference count: 9" in inspection.out

    assert cli.main([*run_arguments[:2], "--output", str(run_b), *run_arguments[2:]]) == 0
    capsys.readouterr()

    expected_schedule = [
        (0, "full-frame"),
        (0, "left"),
        (0, "right"),
        (1, "full-frame"),
        (1, "left"),
        (1, "right"),
        (2, "full-frame"),
        (2, "left"),
        (2, "right"),
    ]
    repeated_schedules: list[list[tuple[int, str]]] = []
    for run_dir in (run_a, run_b):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        inference_rows = [
            json.loads(line)
            for line in (run_dir / "inferences.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert summary["frames_processed"] == 3
        assert summary["inference_count"] == 9
        schedule = sorted((row["frame_index"], row["region_id"]) for row in inference_rows)
        assert schedule == expected_schedule
        repeated_schedules.append(schedule)
    assert repeated_schedules == [expected_schedule, expected_schedule]

    assert cli.main(["compare", str(run_a), str(run_b)]) == 0
    comparison = capsys.readouterr()
    assert comparison.err == ""
    assert comparison.out == "equivalent=true left=9 right=9 mismatches=0\n"

    for run_dir in (run_a, run_b):
        for artifact in (
            "manifest.json",
            "summary.json",
            "inferences.jsonl",
            "detections.jsonl",
            "hardware.jsonl",
        ):
            assert (run_dir / artifact).is_file()
        assert [path.name for path in sorted((run_dir / "annotated").glob("*.png"))] == [
            "000000.png",
            "000001.png",
            "000002.png",
        ]
