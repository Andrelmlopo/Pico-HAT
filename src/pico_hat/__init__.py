"""Pico-HAT: Hypothesis-Anchored Tracking."""

from .config import shared_config
from .tracker import Tracker

__all__ = ["Tracker", "shared_config"]
__version__ = "0.1.0"
