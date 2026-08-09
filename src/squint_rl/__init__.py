from importlib.metadata import version

from .episode import Episode
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
    "Episode",
    "DetectionBatch",
    "GroundTruthBatch",
    "TrackBatch",
    "TrackerSummary",
    "ObservationScales",
    "PolicyContext",
    "Tracker",
]
