from __future__ import annotations

import math
import os

import numpy as np
import pytest

from edge_perception.contracts import Detection
from edge_perception.detectors.dfine import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DfineDetector,
)


def _model_test_device() -> str:
    return os.environ.get("MODEL_TEST_DEVICE", "cpu")


def test_model_test_device_respects_requested_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_TEST_DEVICE", "cuda")

    assert _model_test_device() == "cuda"


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
def test_pinned_dfine_model_returns_backend_neutral_batch() -> None:
    device = _model_test_device()
    detector = DfineDetector.load(device=device)
    image = np.arange(640 * 640 * 3, dtype=np.uint32).reshape(640, 640, 3).astype(np.uint8)

    result = detector.predict((image,))

    assert detector.identity.model_id == DEFAULT_MODEL_ID
    assert detector.identity.revision == DEFAULT_REVISION
    assert detector.identity.device == device
    assert detector.identity.dtype == "float32"
    assert len(detector.identity.weights_sha256) == 64
    assert len(result.detections) == 1
    assert all(isinstance(detection, Detection) for detection in result.detections[0])
    assert all(math.isfinite(value) for value in result.timing.to_dict().values())
    assert "torch" not in repr(result.to_dict()).lower()
