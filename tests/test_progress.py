from __future__ import annotations

from typing import Any

import pytest

from edge_perception.progress import ProgressEvent


def test_progress_event_is_finite_json_native() -> None:
    event = ProgressEvent("running", 2, 6, 123.5, None)

    assert event.to_dict() == {
        "phase": "running",
        "frames_processed": 2,
        "inference_count": 6,
        "elapsed_ms": 123.5,
        "error": None,
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("unknown", 0, 0, 0.0, None), "phase"),
        (("running", -1, 0, 0.0, None), "frames_processed"),
        (("running", True, 0, 0.0, None), "frames_processed"),
        (("running", 0, -1, 0.0, None), "inference_count"),
        (("running", 0, 0, float("nan"), None), "elapsed_ms"),
        (("running", 0, 0, float("inf"), None), "elapsed_ms"),
        (("running", 0, 0, -0.1, None), "elapsed_ms"),
        (("failed", 0, 0, 0.0, 2), "error"),
    ],
)
def test_progress_event_rejects_non_protocol_values(
    values: tuple[Any, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ProgressEvent(*values)
