import json

import pytest

from edge_perception.contracts import (
    BatchPrediction,
    Box,
    Detection,
    DetectorIdentity,
    Region,
    StageTiming,
)


def test_box_rejects_non_finite_and_degenerate_values() -> None:
    with pytest.raises(ValueError):
        Box(float("nan"), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        Box(1.0, 0.0, 1.0, 1.0)


def test_detection_is_backend_neutral_and_json_serializable() -> None:
    detection = Detection(Box(1.25, 2.5, 10.75, 20.0), 3, 0.875, "car")
    encoded = json.dumps(detection.to_dict(), sort_keys=True)
    assert "torch" not in encoded.lower()
    assert json.loads(encoded)["box"] == [1.25, 2.5, 10.75, 20.0]


def test_region_requires_positive_integer_extent() -> None:
    with pytest.raises(ValueError):
        Region("bad", 0, 0, 0, 10)


def test_batch_prediction_serializes_nested_contracts() -> None:
    prediction = BatchPrediction(
        detections=((Detection(Box(0.0, 1.0, 2.0, 3.0), 1, 0.5),), ()),
        timing=StageTiming(1.0, 2.0, 3.0, 6.0),
    )

    assert prediction.to_dict() == {
        "detections": [[{"box": [0.0, 1.0, 2.0, 3.0], "class_id": 1, "score": 0.5, "label": None}], []],
        "timing": {
            "preprocess_ms": 1.0,
            "inference_ms": 2.0,
            "postprocess_ms": 3.0,
            "total_ms": 6.0,
        },
    }


def test_timing_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        StageTiming(-0.1, 0.0, 0.0, 0.0)


def test_detector_identity_serializes_all_provenance_fields() -> None:
    identity = DetectorIdentity(
        adapter="transformers",
        model_id="dfine-n",
        revision="abc123",
        weights_sha256="a" * 64,
        backend="torch",
        backend_version="2.11.0",
        device="cpu",
        dtype="float32",
    )

    assert identity.to_dict() == {
        "adapter": "transformers",
        "model_id": "dfine-n",
        "revision": "abc123",
        "weights_sha256": "a" * 64,
        "backend": "torch",
        "backend_version": "2.11.0",
        "device": "cpu",
        "dtype": "float32",
    }
