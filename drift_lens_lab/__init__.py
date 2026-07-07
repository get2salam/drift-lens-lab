"""DriftLens Lab — simulate, detect, and report concept drift."""

__version__ = "0.1.0"

from .evaluation import DetectorScore, score_detector
from .runner import run  # convenient programmatic entry point

__all__ = ["run", "score_detector", "DetectorScore", "__version__"]
