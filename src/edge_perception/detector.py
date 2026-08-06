"""Dependency-neutral detector interface."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from edge_perception.contracts import BatchPrediction, DetectorIdentity


class Detector(Protocol):
    @property
    def identity(self) -> DetectorIdentity: ...

    def warmup(self, image: np.ndarray, runs: int) -> None: ...

    def predict(self, images: Sequence[np.ndarray]) -> BatchPrediction: ...

    def peak_device_memory_bytes(self) -> int | None: ...
