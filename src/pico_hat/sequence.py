"""RGB sequence manifests, camera calibration, and target localization."""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def natural_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


class Sequence:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.config = json.loads(self.path.read_text())
        root = self.path.parent
        allowed = {"rgb", "masks", "boxes", "camera", "slam_input", "slam_resolution"}
        unknown = set(self.config) - allowed
        if unknown:
            raise ValueError(f"Unknown sequence fields: {sorted(unknown)}")
        rgb = (root / self.config["rgb"]).resolve()
        self.frames = sorted(
            (
                p
                for p in rgb.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            ),
            key=natural_key,
        )
        if not self.frames:
            raise ValueError(f"No RGB frames in {rgb}")
        if len({p.stem for p in self.frames}) != len(self.frames):
            raise ValueError("RGB frames must have unique stems for mask matching")
        if ("masks" in self.config) == ("boxes" in self.config):
            raise ValueError("Provide exactly one of masks or boxes")
        self.mask_root = (root / self.config["masks"]).resolve() if "masks" in self.config else None
        if self.mask_root is not None and not self.mask_root.is_dir():
            raise ValueError(f"Mask directory does not exist: {self.mask_root}")
        self.boxes = (
            json.loads((root / self.config["boxes"]).read_text())
            if "boxes" in self.config
            else None
        )
        if self.boxes is not None:
            if not isinstance(self.boxes, dict):
                raise ValueError("boxes must be a JSON object keyed by RGB filename")
            unknown = set(self.boxes) - {p.name for p in self.frames}
            if unknown:
                raise ValueError(f"Boxes refer to unknown RGB files: {sorted(unknown)[:5]}")
        camera = self.config["camera"]
        if set(camera) != {"K", "width", "height"}:
            raise ValueError("camera requires K, width and height; undistort images beforehand")
        self.K = np.asarray(camera["K"], dtype=float)
        self.width, self.height = camera["width"], camera["height"]
        if any(type(v) is not int or v < 8 for v in (self.width, self.height)):
            raise ValueError("Camera width and height must be integers of at least 8 pixels")
        if (
            self.K.shape != (3, 3)
            or not np.isfinite(self.K).all()
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
            or not np.allclose(self.K[2], [0, 0, 1])
            or self.K[0, 1] != 0
            or self.K[1, 0] != 0
        ):
            raise ValueError("K must be a finite, zero-skew pinhole camera matrix")
        self.slam_input = self.config.get("slam_input", "masked" if self.mask_root else "rgb")
        if self.slam_input not in {"masked", "rgb"}:
            raise ValueError("slam_input must be masked or rgb")
        self.slam_resolution = self.config.get("slam_resolution", 196608)
        if type(self.slam_resolution) is not int or self.slam_resolution < 4096:
            raise ValueError("slam_resolution must be an integer pixel area of at least 4096")

    def __len__(self):
        return len(self.frames)

    def localization(self, index):
        path = self.frames[index]
        if self.mask_root is not None:
            mask_path = self.mask_root / (path.stem + ".png")
            if not mask_path.exists():
                return None
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            if mask.ndim == 3 and mask.shape[2] == 3:
                if np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(
                    mask[..., 0], mask[..., 2]
                ):
                    mask = mask[..., 0]
            if mask.ndim != 2 or mask.shape != (self.height, self.width):
                raise ValueError(f"{mask_path}: expected a single-channel mask at RGB resolution")
            mask = mask != 0
        else:
            box = self.boxes.get(path.name)
            if box is None:
                return None
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError(f"{path.name}: expected a finite [x1, y1, x2, y2] box")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
                raise ValueError(f"{path.name}: box is empty or outside the image")
            mask = np.zeros((self.height, self.width), dtype=bool)
            mask[int(np.floor(y1)) : int(np.ceil(y2)), int(np.floor(x1)) : int(np.ceil(x2))] = True
        return mask if mask.sum() >= 8 else None

    def rgb(self, index):
        path = self.frames[index]
        with Image.open(path) as image:
            if image.size != (self.width, self.height):
                raise ValueError(f"{path}: image size differs from camera calibration")
            return np.asarray(image.convert("RGB")).copy()

    def validate(self):
        available = 0
        for index in range(len(self)):
            self.rgb(index)
            available += self.localization(index) is not None
        if not available:
            raise ValueError("The sequence has no valid target masks or boxes")
        return {
            "frames": len(self),
            "detections": available,
            "width": self.width,
            "height": self.height,
            "slam_input": self.slam_input,
        }
