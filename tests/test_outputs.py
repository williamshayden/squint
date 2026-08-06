import json
import math
from pathlib import Path

import numpy as np

from edge_perception.contracts import Box, Detection, Region
from edge_perception.outputs import RunOutputs, summarize_latencies
from edge_perception.telemetry import TelemetrySample


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_outputs_write_linked_json_records_and_annotated_frame(tmp_path: Path) -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    region = Region("right", 10, 4, 12, 16)
    detection = Detection(Box(10.25, 4.5, 20.75, 18.0), 1, 0.875, "person")

    with RunOutputs(
        tmp_path / "run",
        run_id="run-001",
        manifest={"detector": {"backend": "plain"}},
    ) as outputs:
        outputs.write_inference(
            {
                "frame_id": "frame-000000",
                "inference_id": "inference-000000-001",
                "region_id": region.region_id,
                "timing": {"inference_ms": 1.25},
            }
        )
        outputs.write_detections(
            frame_id="frame-000000",
            inference_id="inference-000000-001",
            region_id=region.region_id,
            detections=(detection,),
        )
        outputs.write_hardware(TelemetrySample(123, 10, 20, None, None, None, None).to_dict())
        outputs.flush_frame()
        outputs.annotate(0, frame, regions=(region,), detections=(detection,))
        outputs.write_summary({"complete_frame": {"count": 1}})

    run_dir = tmp_path / "run"
    for relative_path in (
        "manifest.json",
        "inferences.jsonl",
        "detections.jsonl",
        "hardware.jsonl",
        "summary.json",
        "annotated/000000.png",
    ):
        assert (run_dir / relative_path).is_file()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    inference = _records(run_dir / "inferences.jsonl")[0]
    detection_record = _records(run_dir / "detections.jsonl")[0]
    hardware = _records(run_dir / "hardware.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert all(record["schema_version"] == "0.1.0" for record in (manifest, inference, detection_record, hardware, summary))
    assert {manifest["run_id"], inference["run_id"], detection_record["run_id"], hardware["run_id"], summary["run_id"]} == {"run-001"}
    assert detection_record["frame_id"] == inference["frame_id"] == "frame-000000"
    assert detection_record["inference_id"] == inference["inference_id"] == "inference-000000-001"
    assert detection_record["box"] == [10.25, 4.5, 20.75, 18.0]
    serialized = "\n".join((run_dir / name).read_text(encoding="utf-8") for name in ("manifest.json", "inferences.jsonl", "detections.jsonl", "hardware.jsonl", "summary.json"))
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert "tensor(" not in serialized


def test_summarize_latencies_uses_linear_percentiles() -> None:
    summary = summarize_latencies([1, 2, 3, 4, 100])

    assert summary["count"] == 5
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == np.percentile([1, 2, 3, 4, 100], 95, method="linear")
    assert summary["p99_ms"] == np.percentile([1, 2, 3, 4, 100], 99, method="linear")
    assert all(math.isfinite(value) for key, value in summary.items() if key != "count")
