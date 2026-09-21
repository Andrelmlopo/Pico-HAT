"""DROID-SLAM frontend with frozen per-frame camera-to-world output."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class DroidMotion:
    def __init__(self, source, checkpoint, sequence, buffer=512):
        import cv2
        import torch

        sys.path[:0] = [str(Path(source).resolve() / "droid_slam"), str(Path(source).resolve())]
        from droid import Droid

        from .droid_readout import build_args, causal_pose

        self.torch, self.cv2 = torch, cv2
        self.readout = causal_pose
        self.sequence = sequence
        scale = np.sqrt(sequence.slam_resolution / (sequence.width * sequence.height))
        self.width, self.height = int(sequence.width * scale), int(sequence.height * scale)
        shape = [self.height - self.height % 8, self.width - self.width % 8]
        if min(shape) < 8:
            raise ValueError("SLAM resolution is too small for this aspect ratio")
        args = build_args(Path(checkpoint), shape)
        if type(buffer) is not int or buffer < 16:
            raise ValueError("droid_buffer must be an integer of at least 16")
        args.buffer = buffer
        self.model = Droid(args)
        K = sequence.K
        self.intrinsics = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        self.intrinsics[0::2] *= self.width / sequence.width
        self.intrinsics[1::2] *= self.height / sequence.height

    def step(self, index):
        torch, cv2 = self.torch, self.cv2
        if self.model.video.counter.value + 2 >= self.model.args.buffer:
            raise RuntimeError(
                "DROID keyframe buffer is full. Increase droid_buffer in runtime.json."
            )
        image = self.sequence.rgb(index)[..., ::-1].copy()
        if self.sequence.slam_input == "masked":
            mask = self.sequence.localization(index)
            if mask is None:
                return np.full((4, 4), np.nan)
            image[~mask] = 0
        image = cv2.resize(image, (self.width, self.height))
        image = image[: self.height - self.height % 8, : self.width - self.width % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)[None]
        with torch.no_grad():
            self.model.track(index, image, intrinsics=self.intrinsics)
            if not self.model.frontend.is_initialized:
                return np.full((4, 4), np.nan)
            estimate = self.readout(self.model, index, image, self.intrinsics, 3)
            if estimate is None:
                return np.full((4, 4), np.nan)
            row = estimate.inv().data.cpu().numpy()[0]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
        pose[:3, 3] = row[:3]
        return pose
