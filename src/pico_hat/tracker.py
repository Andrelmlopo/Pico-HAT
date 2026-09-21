"""Incremental temporal processing of PicoPose candidates and DROID poses."""

import numpy as np

from .config import shared_config
from .fusion import _fusion_steps
from .geometry import valid_pose
from .selector import Selector


class Tracker:
    """Emit one object-to-camera pose per input frame, without revising past output.

    ``slam`` is a frozen camera-to-world pose. Candidate translations are metres.
    Missing motion is an all-NaN 4x4 array. Candidates are supplied only when
    ``anchor_due`` is true and a detection is available.
    """

    def __init__(self, capacity=512):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.config = shared_config()
        self.selector = Selector(self.config)
        self.anchors = np.full((capacity, 4, 4), np.nan)
        self.slam = np.full_like(self.anchors, np.nan)
        self.anchor_valid = np.zeros(capacity, dtype=bool)
        self.slam_valid = np.zeros(capacity, dtype=bool)
        self.areas = np.ones(capacity)
        self.frame = 0
        self.last_attempt = -self.config["anchor_gap"]
        self.last_pose = np.full((4, 4), np.nan)
        self.generator = _fusion_steps(
            self.anchors[:0],
            self.slam[:0],
            self.anchor_valid[:0],
            self.slam_valid[:0],
            self.areas[:0],
            **self.config["fusion"],
        )
        next(self.generator)

    @property
    def anchor_due(self):
        return self.frame - self.last_attempt >= self.config["anchor_gap"]

    def step(self, slam, candidates=None):
        slam = np.asarray(slam, dtype=float)
        if slam.shape != (4, 4):
            raise ValueError("SLAM pose must be a 4x4 matrix")
        if not valid_pose(slam) and not np.isnan(slam).all():
            raise ValueError("SLAM pose must be SE(3) or entirely NaN")
        if candidates is not None and not self.anchor_due:
            raise ValueError("Candidate call violates the shared minimum anchor gap of 6")
        frame = self.frame
        if frame == len(self.anchors):
            for name in ("anchors", "slam"):
                value = getattr(self, name)
                setattr(self, name, np.concatenate([value, np.full_like(value, np.nan)]))
            for name in ("anchor_valid", "slam_valid"):
                value = getattr(self, name)
                setattr(self, name, np.concatenate([value, np.zeros_like(value)]))
            self.areas = np.ones(len(self.anchors))
        self.slam[frame] = slam
        self.slam_valid[frame] = valid_pose(slam)
        selection = None
        if candidates is not None:
            selected, selection = self.selector.step(*candidates, slam)
            self.last_attempt = frame
            if selected is not None:
                self.anchors[frame] = selected
                self.anchor_valid[frame] = True
        stop = frame + 1
        pose, diagnostics = self.generator.send(
            (
                self.anchors[:stop],
                self.slam[:stop],
                self.anchor_valid[:stop],
                self.slam_valid[:stop],
                self.areas[:stop],
            )
        )
        observed = bool(valid_pose(pose))
        if observed:
            self.last_pose = pose.copy()
        self.frame = stop
        diagnostics.update(
            selection=selection, held=not observed, valid=bool(valid_pose(self.last_pose))
        )
        return self.last_pose.copy(), diagnostics

    def close(self):
        self.generator.close()
