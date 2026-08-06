"""Dependency-light detector discovery and on-demand adapter loading."""

from __future__ import annotations

from dataclasses import dataclass

from edge_perception.detector import Detector


@dataclass(frozen=True, slots=True)
class DetectorDescriptor:
    detector_id: str
    display_name: str
    model_id: str
    revision: str


_DFINE = DetectorDescriptor(
    detector_id="dfine-nano-coco",
    display_name="D-FINE Nano (COCO)",
    model_id="ustc-community/dfine-nano-coco",
    revision="066438d3d8f0da137a37b38fdf3368fd4afceced",
)


def detector_descriptors() -> tuple[DetectorDescriptor, ...]:
    """Return supported detectors without importing their model runtimes."""

    return (_DFINE,)


def load_detector(detector_id: str, *, threshold: float, device: str) -> Detector:
    """Load one supported detector after its identifier has been validated."""

    if detector_id != _DFINE.detector_id:
        raise ValueError(f"unknown detector ID: {detector_id}")
    from edge_perception.detectors.dfine import DfineDetector

    return DfineDetector.load(threshold=threshold, device=device)
