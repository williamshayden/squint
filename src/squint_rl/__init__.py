from importlib.metadata import version

from gymnasium.envs.registration import register, registry

from .benchmark import evaluate
from .env import SquintEnv
from .episode import Episode
from .tracker import Tracker

if "SquintReplay-v0" not in registry:
    register(id="SquintReplay-v0", entry_point="squint_rl.env:SquintEnv")

__version__ = version("squint-rl")
__all__ = [
    "Episode",
    "SquintEnv",
    "Tracker",
    "__version__",
    "evaluate",
]
