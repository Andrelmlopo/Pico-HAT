"""The shared causal configuration from HAT (https://arxiv.org/abs/2609.21597)."""

import json
from importlib.resources import files


def shared_config():
    """Return a fresh copy of the fixed benchmark settings."""
    return json.loads(files("pico_hat").joinpath("shared.json").read_text())
