"""Load the published PicoPose checkpoint or a tensor-only state dictionary."""

import hashlib
from pathlib import Path

PICOPOSE_SHA256 = "9d1173d3db55e690e468edde453b06d47402d7b993748e4d2fe8f34a203014c0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pico_state(path):
    import torch

    # The published Lightning checkpoint includes OmegaConf metadata, which
    # PyTorch 2.0 cannot load with its tensor-only unpickler. Permit that format
    # only when the file matches the published checkpoint used for validation.
    published = sha256(path) == PICOPOSE_SHA256
    checkpoint = torch.load(path, map_location="cpu", weights_only=not published)
    state = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("network.") for k in state):
        state = {k[len("network.") :]: v for k, v in state.items() if k.startswith("network.")}
    return state
