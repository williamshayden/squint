"""Process-neutral progress and cancellation contracts for checkpoint runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

ProgressPhase = Literal[
    "validating",
    "warming_up",
    "running",
    "complete",
    "cancelled",
    "failed",
]
ProgressCallback = Callable[["ProgressEvent"], None]
CancelCheck = Callable[[], bool]

_PROGRESS_PHASES = {
    "validating",
    "warming_up",
    "running",
    "complete",
    "cancelled",
    "failed",
}


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: ProgressPhase
    frames_processed: int
    inference_count: int
    elapsed_ms: float
    error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or self.phase not in _PROGRESS_PHASES:
            raise ValueError(f"unsupported progress phase: {self.phase}")
        for field_name in ("frames_processed", "inference_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError("elapsed_ms must be a finite number")
        elapsed_ms = float(self.elapsed_ms)
        if not isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        object.__setattr__(self, "elapsed_ms", elapsed_ms)
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "frames_processed": self.frames_processed,
            "inference_count": self.inference_count,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }
