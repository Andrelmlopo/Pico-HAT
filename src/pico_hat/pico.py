"""Adapter for the native PicoPose stages 1--3 and PnP recovery."""

import json
import sys
from pathlib import Path

import numpy as np


class PicoPose:
    def __init__(self, source, checkpoint, templates):
        import cv2
        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, str(Path(source).resolve()))
        from model.picopose import Net
        from utils.data_utils import get_bbox, get_point_cloud_from_depth
        from utils.pose_recovery import pose_recovery_ransac_pnp
        from utils.torch_utils import init_points2d_numpy

        self.torch, self.cv2 = torch, cv2
        self.get_bbox = get_bbox
        self.recover = pose_recovery_ransac_pnp
        self.grid2d = init_points2d_numpy(224, patch_size=224 / 64)
        config = OmegaConf.load(Path(source) / "config/base.yaml").model
        config.stage1.pretrained = False
        self.net = Net(config)
        from .checkpoints import load_pico_state

        state = load_pico_state(checkpoint)
        self.net.load_state_dict(state, strict=True)
        self.net.cuda().eval()
        metadata = json.loads((Path(templates) / "metadata.json").read_text())
        if metadata.get("format") != "pico-hat.templates.v1":
            raise ValueError("Use templates produced by pico-hat prepare")
        with np.load(Path(templates) / "templates.npz", allow_pickle=False) as archive:
            images, depths, poses, K = (archive[k] for k in ("rgb", "depth", "poses", "K"))
        if (
            images.shape != (162, 480, 640, 3)
            or depths.shape != (162, 480, 640)
            or poses.shape != (162, 4, 4)
            or K.shape != (3, 3)
        ):
            raise ValueError("Invalid template array dimensions")
        data = {k: [] for k in ("tem_rgb", "tem_mask", "tem_pts3d", "tem_M", "tem_pose", "tem_K")}
        for image, depth, pose in zip(images, depths, poses):
            mask = (depth > 0).astype(np.float32)
            bbox = get_bbox(mask)
            rgb, cropped, M = self.preprocess(image, mask, bbox)
            points = get_point_cloud_from_depth(depth, K, list(bbox))
            points = cv2.resize(points, (64, 64), interpolation=cv2.INTER_NEAREST)
            for key, value in zip(data, (rgb, cropped, points, M, pose, K)):
                data[key].append(value)
        self.templates = {
            k: torch.as_tensor(np.array(v), dtype=torch.float32, device="cuda")[None]
            for k, v in data.items()
        }
        with torch.no_grad():
            self.features = torch.cat(
                [
                    self.net.feature_extractor(
                        self.templates["tem_rgb"][0][i : i + 16].contiguous()
                    )[-1]
                    for i in range(0, 162, 16)
                ]
            )[None]

    def preprocess(self, image, mask, bbox):
        y1, y2, x1, x2 = bbox
        if y2 <= y1 or x2 <= x1:
            raise ValueError("The localized target is too small to crop")
        cv2 = self.cv2
        # Native PicoPose expects BGR channels with these normalization constants.
        rgb = image[..., ::-1][y1:y2, x1:x2, :3] / 255.0
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = (
            (rgb.transpose(2, 0, 1) - np.array([0.48145466, 0.4578275, 0.40821073])[:, None, None])
            / np.array([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        ).astype(np.float32)
        crop = cv2.resize(
            mask[y1:y2, x1:x2].astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
        M = np.array(
            [
                [224 / (x2 - x1), 0, -x1 * 224 / (x2 - x1)],
                [0, 224 / (y2 - y1), -y1 * 224 / (y2 - y1)],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return rgb, crop, M

    def predict(self, image, mask, K):
        torch = self.torch
        rgb, mask, M = self.preprocess(image, mask, self.get_bbox(mask))
        homogeneous = np.concatenate([self.grid2d, np.ones((64, 64, 1))], axis=2)
        points = np.linalg.inv(M) @ homogeneous.reshape(-1, 3).T
        points = (points[:2] / points[2:]).T.reshape(64, 64, 2)
        data = dict(self.templates, template_feature=self.features)
        for key, value in dict(
            real_rgb=rgb, real_mask=mask, real_M=M, real_K=K, real_pts2d=points, real_pose=np.eye(4)
        ).items():
            data[key] = torch.as_tensor(value, dtype=torch.float32, device="cuda")[None]
        with torch.no_grad():
            outputs = self.net(data, 5)
            poses, scores = [], []
            for item in outputs:
                R, t, score, ok = self.recover(
                    item["tar_pts_2d"][0],
                    item["src_pts_3d"][0],
                    data["real_K"][0],
                    item["tem_pose"][0],
                    item["pred_tar_pts"][0],
                    item["pred_src_pts"][0],
                )
                pose = np.eye(4)
                if ok:
                    pose[:3, :3], pose[:3, 3] = R, np.asarray(t).reshape(3)
                else:
                    pose = item["pred_poses"][0].cpu().numpy()
                poses.append(pose)
                scores.append(score)
        scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-scores)
        return np.asarray(poses, dtype=np.float32)[order].astype(float), scores[order]
