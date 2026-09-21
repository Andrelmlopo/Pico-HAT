# Third-party components

Pico-HAT's Python package contains the temporal method, input handling,
rendering integration and adapters. It does not bundle neural checkpoints,
datasets, CAD models or upstream model repositories.

- **PicoPose**, Lihua Liu, Jiehong Lin, Zhenxin Liu and Kui Jia.
  [Repository](https://github.com/foollh/PicoPose),
  [paper](https://arxiv.org/abs/2504.02617).
  The pinned upstream README displays an MIT license badge, but that revision
  has no root LICENSE file. Its DINO-derived files also carry their own notices.
  Obtain and use the source and weights under their upstream terms. No claim
  is made that this repository's license covers them.
- **DROID-SLAM**, Zachary Teed and Jia Deng.
  [Repository](https://github.com/princeton-vl/DROID-SLAM),
  [paper](https://arxiv.org/abs/2108.10869). BSD 3-Clause.
  The causal readout adapts the upstream trajectory-filling procedure.
  Its notice is retained in [licenses/DROID-SLAM.txt](licenses/DROID-SLAM.txt).
- **pyrender**, **trimesh**, **NumPy**, **SciPy**, **Pillow**, **OpenCV**,
  **PyTorch**, **xFormers**, **MMCV**, and their dependencies are installed
  separately. Their respective licenses continue to apply.

```bibtex
@article{liu2025picopose,
  title={PicoPose: Progressive Pixel-to-Pixel Correspondence Learning for Novel Object Pose Estimation},
  author={Liu, Lihua and Lin, Jiehong and Liu, Zhenxin and Jia, Kui},
  journal={arXiv preprint arXiv:2504.02617},
  year={2025}
}

@article{teed2021droid,
  title={DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
