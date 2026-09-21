"""Causal DROID readout from current and past keyframes only."""

# Adapted from DROID-SLAM's trajectory filler. See licenses/DROID-SLAM.txt.

from pathlib import Path
from types import SimpleNamespace

import torch


@torch.no_grad()
def causal_pose(droid, tstamp, image, intrinsics, n_iter=3):
    from factor_graph import FactorGraph
    from lietorch import SE3

    video, tf = (droid.video, droid.traj_filler)
    N = int(video.counter.value)
    if N == 0:
        return None
    ts = video.tstamp[:N]
    Ps = SE3(video.poses[:N])
    if float(ts[N - 1]) == float(tstamp):
        return SE3(video.poses[N - 1 : N].clone())
    if N >= 2:
        dt = ts[N - 1] - ts[N - 2] + 0.001
        v = (Ps[N - 1 : N] * Ps[N - 2 : N - 1].inv()).log() / dt
        G = SE3.exp(v * (float(tstamp) - ts[N - 1])) * Ps[N - 1 : N]
    else:
        G = Ps[N - 1 : N]
    if n_iter <= 0:
        return G
    tt = torch.as_tensor([float(tstamp)], device="cuda")
    images = image[None].cuda()
    inputs = images[:, :, [2, 1, 0]] / 255.0
    inputs = inputs.sub_(tf.MEAN).div_(tf.STDV)
    with torch.autocast(device_type="cuda", enabled=True):
        fmap = tf._PoseTrajectoryFiller__feature_encoder(inputs)
    video.counter.value += 1
    try:
        video[N : N + 1] = (tt, images[:, 0], G.data, 1, None, intrinsics[None].cuda() / 8.0, fmap)
        graph = FactorGraph(video, tf.update)
        dst = torch.arange(N, N + 1, device="cuda")
        graph.add_factors(torch.as_tensor([N - 1], device="cuda"), dst)
        graph.add_factors(torch.as_tensor([max(N - 2, 0)], device="cuda"), dst)
        for _ in range(n_iter):
            graph.update(N, N + 1, motion_only=True)
        return SE3(video.poses[N : N + 1].clone())
    finally:
        video.counter.value = N


def build_args(weights: Path, image_size):
    return SimpleNamespace(
        weights=str(weights),
        buffer=512,
        image_size=image_size,
        disable_vis=True,
        beta=0.3,
        filter_thresh=2.4,
        warmup=8,
        keyframe_thresh=4.0,
        frontend_thresh=16.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
        stereo=False,
        asynchronous=False,
        frontend_device="cuda",
        backend_device="cuda",
    )
