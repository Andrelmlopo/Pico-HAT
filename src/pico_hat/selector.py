"""Forward hypothesis selection with the running confidence gate."""

import numpy as np

from .geometry import rotation_distance, valid_pose


class Selector:
    def __init__(self, config):
        self.settings = config["selector"]
        self.guard = config["candidate_guard"]
        self.count = config["candidate_count"]
        self.scores = None
        self.previous = None
        self.tight_counts = []
        self.overrides = []

    def step(self, poses, scores, slam):
        poses, scores = np.array(poses, dtype=float), np.array(scores, dtype=float)
        if poses.shape != (self.count, 4, 4) or scores.shape != (self.count,):
            raise ValueError(f"Expected {self.count} candidate poses and scores")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Candidate scores must be finite inlier ratios in [0, 1]")
        norm = np.linalg.norm(poses[:, :3, 3], axis=-1)
        good = (
            valid_pose(poses)
            & (norm >= self.guard["minimum_translation_m"])
            & (norm <= self.guard["maximum_translation_m"])
            & (poses[:, 2, 3] > 0)
            & (scores > 0)
        )
        if good.sum() >= 3:
            depth = poses[:, 2, 3]
            median = np.median(depth[good])
            ratio = self.guard["depth_ratio"]
            good &= (depth >= median / ratio) & (depth <= median * ratio)
        if not good.any():
            return None, {"accepted": False, "valid_candidates": 0}
        poses[~good] = np.eye(4)
        scores[~good] = -1e9
        rotations = poses[:, :3, :3]
        unary = scores * self.settings["unary_scale"]
        if self.scores is None:
            self.scores = unary.copy()
        else:
            previous_slam, previous_rotations = self.previous
            has_motion = valid_pose(slam) and valid_pose(previous_slam)
            predicted = (
                (slam[:3, :3].T @ previous_slam[:3, :3]) @ previous_rotations
                if has_motion
                else None
            )
            self.scores = np.array(
                [
                    np.max(
                        self.scores
                        - self.settings["motion_weight"]
                        * (
                            np.array([rotation_distance(rotations[k], r) for r in predicted])
                            if has_motion
                            else np.zeros(self.count)
                        )
                    )
                    + unary[k]
                    for k in range(self.count)
                ]
            )
        choice, top = int(self.scores.argmax()), int(scores.argmax())
        trace = np.einsum("kij,ij->k", rotations, rotations[top])
        tight = trace > 1 + 2 * np.cos(np.deg2rad(self.settings["tight_degrees"]))
        self.tight_counts.append(int((tight & good).sum()))
        self.overrides.append(choice != top)
        accepted = (
            len(self.tight_counts) >= self.settings["warmup"]
            and np.median(self.tight_counts) >= self.settings["tight_minimum"]
            and np.mean(self.overrides) <= self.settings["maximum_override_fraction"]
        )
        self.previous = (slam.copy(), rotations.copy())
        index = choice if accepted else top
        return poses[index].copy(), {
            "accepted": bool(accepted),
            "candidate": index,
            "valid_candidates": int(good.sum()),
        }
