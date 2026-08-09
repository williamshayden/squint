from importlib.metadata import version

from gymnasium.envs.registration import register, registry

from .env import SquintEnv
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

if "SquintReplay-v0" not in registry:
    register(id="SquintReplay-v0", entry_point="squint_rl.env:SquintEnv")

__version__ = version("squint-rl")
__all__ = [
    "DetectionBatch",
    "Episode",
    "GroundTruthBatch",
    "ObservationScales",
    "PolicyContext",
    "SquintEnv",
    "TrackBatch",
    "Tracker",
    "TrackerSummary",
    "__version__",
]
