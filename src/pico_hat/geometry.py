"""Pose validation and rotation distances."""

import numpy as np


def valid_pose(poses):
    poses = np.asarray(poses)
    if poses.shape[-2:] != (4, 4):
        raise ValueError("Poses must have shape (..., 4, 4)")
    rotation = poses[..., :3, :3]
    # Invalid rows are allowed for missing observations, but never enter a solver.
    safe = np.where(np.isfinite(rotation), rotation, 0.0)
    return (
        np.isfinite(poses).all(axis=(-2, -1))
        & (np.max(np.abs(poses[..., 3, :] - [0, 0, 0, 1]), axis=-1) < 1e-5)
        & (np.max(np.abs(np.swapaxes(safe, -1, -2) @ safe - np.eye(3)), axis=(-2, -1)) < 1e-3)
        & (np.abs(np.linalg.det(safe) - 1) < 1e-3)
    )


def rotation_distance(first, second):
    return float(np.degrees(np.arccos(np.clip((np.trace(first @ second.T) - 1) / 2, -1, 1))))
