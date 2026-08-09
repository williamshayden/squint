from importlib.metadata import version

from .tracker import (
    DetectionBatch,
    GroundTruthBatch,
    ObservationScales,
    PolicyContext,
    TrackBatch,
    Tracker,
    TrackerSummary,
)

__version__ = version("squint-rl")
__all__ = [
    "__version__",
    "DetectionBatch",
    "GroundTruthBatch",
    "TrackBatch",
    "TrackerSummary",
    "ObservationScales",
    "PolicyContext",
    "Tracker",
]
