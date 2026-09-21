"""Causal Sim(3) alignment and SE(3) fusion, using the HAT paper residuals (arXiv:2609.21597)."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = s * R
    A[:3, 3] = t
    return A


def params_to_sim3(params: np.ndarray) -> np.ndarray:
    log_s, omega, t = (params[0], params[1:4], params[4:7])
    s = np.exp(log_s)
    R = Rotation.from_rotvec(omega).as_matrix()
    return sim3_matrix(s, R, t)


def sim3_to_params(A: np.ndarray) -> np.ndarray:
    sR = A[:3, :3]
    t = A[:3, 3]
    s = np.linalg.norm(sR[:, 0])
    R = sR / s
    omega = Rotation.from_matrix(R).as_rotvec()
    return np.array([np.log(s), omega[0], omega[1], omega[2], t[0], t[1], t[2]])


def _rotation_log_single(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = (R[0, 0], R[0, 1], R[0, 2])
    m10, m11, m12 = (R[1, 0], R[1, 1], R[1, 2])
    m20, m21, m22 = (R[2, 0], R[2, 1], R[2, 2])
    trace = m00 + m11 + m22
    if trace >= m00 and trace >= m11 and (trace >= m22):
        s = np.sqrt(max(1.0 + trace, 0.0)) * 2.0
        x, y, z, w = ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    elif m00 >= m11 and m00 >= m22:
        s = np.sqrt(max(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        x, y, z, w = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 >= m22:
        s = np.sqrt(max(1.0 - m00 + m11 - m22, 0.0)) * 2.0
        x, y, z, w = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = np.sqrt(max(1.0 - m00 - m11 + m22, 0.0)) * 2.0
        x, y, z, w = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    nrm = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = (x / nrm, y / nrm, z / nrm, w / nrm)
    if w < 0.0:
        x, y, z, w = (-x, -y, -z, -w)
    sin_half = np.sqrt(x * x + y * y + z * z)
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle < 0.001:
        scale = 2.0 + angle**2 / 12.0 + 7.0 * angle**4 / 2880.0
    else:
        scale = angle / np.sin(angle / 2.0)
    return np.array([x * scale, y * scale, z * scale])


def rotation_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return _rotation_log_single(R)
    m00, m01, m02 = (R[..., 0, 0], R[..., 0, 1], R[..., 0, 2])
    m10, m11, m12 = (R[..., 1, 0], R[..., 1, 1], R[..., 1, 2])
    m20, m21, m22 = (R[..., 2, 0], R[..., 2, 1], R[..., 2, 2])
    trace = m00 + m11 + m22
    cand = np.stack([trace, m00, m11, m22], axis=-1)
    branch = np.argmax(cand, axis=-1)
    q = np.empty(R.shape[:-2] + (4,))
    b0 = branch == 0
    s = np.sqrt(np.maximum(1.0 + trace, 0.0)) * 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q[b0] = np.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], axis=-1)[b0]
        b1 = branch == 1
        s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        q[b1] = np.stack(
            [0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], axis=-1
        )[b1]
        b2 = branch == 2
        s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2.0
        q[b2] = np.stack(
            [(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], axis=-1
        )[b2]
        b3 = branch == 3
        s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2.0
        q[b3] = np.stack(
            [(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], axis=-1
        )[b3]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    sin_half = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, q[..., 3])
    small = angle < 0.001
    scale = np.where(
        small,
        2.0 + angle**2 / 12.0 + 7.0 * angle**4 / 2880.0,
        angle / np.where(small, 1.0, np.sin(angle / 2.0)),
    )
    return v * scale[..., None]


def se3_to_6vec(T: np.ndarray) -> np.ndarray:
    omega = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([omega, T[:3, 3]])


def vec6_to_se3(v: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:6]
    return T


def fuse_one_pose_global(params: np.ndarray, slam_pose: np.ndarray) -> np.ndarray:
    log_s = params[0]
    s = np.exp(log_s)
    R_A = Rotation.from_rotvec(params[1:4]).as_matrix()
    t_A = params[4:7]
    R_slam = slam_pose[:3, :3]
    t_slam = slam_pose[:3, 3]
    R_fused = R_A @ R_slam
    t_fused = s * R_A @ t_slam + t_A
    T = np.eye(4)
    T[:3, :3] = R_fused.T
    T[:3, 3] = -R_fused.T @ t_fused
    return T


def fuse_poses_global_batch(params: np.ndarray, slam_poses: np.ndarray) -> tuple:
    s = np.exp(params[0])
    R_A = Rotation.from_rotvec(params[1:4]).as_matrix()
    t_A = params[4:7]
    R_fused = R_A @ slam_poses[:, :3, :3]
    t_fused = s * (slam_poses[:, :3, 3] @ R_A.T) + t_A
    R_out = np.swapaxes(R_fused, 1, 2)
    t_out = -np.einsum("nij,nj->ni", R_out, t_fused)
    return (R_out, t_out)


def sim3_residual_batch(
    params: np.ndarray, anchor_poses: np.ndarray, slam_poses: np.ndarray, rot_only: bool = False
) -> np.ndarray:
    R_out, t_out = fuse_poses_global_batch(params, slam_poses)
    R_err = anchor_poses[:, :3, :3] @ np.swapaxes(R_out, 1, 2)
    e_rot = rotation_log(R_err)
    if rot_only:
        return e_rot
    e_trans = anchor_poses[:, :3, 3] - t_out
    return np.concatenate([e_rot, e_trans], axis=1)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple:
    n = src.shape[1]
    mu_s = src.mean(axis=1, keepdims=True)
    mu_d = dst.mean(axis=1, keepdims=True)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = (src_c**2).sum() / n
    if var_s < 1e-12:
        return (1.0, np.eye(3), (mu_d - mu_s).squeeze())
    W = dst_c @ src_c.T / n
    U, sigma, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (sigma * S.diagonal()).sum() / var_s
    t = (mu_d - s * R @ mu_s).squeeze()
    return (s, R, t)


def stage1_global_sim3(
    anchor_poses,
    slam_poses,
    valid_both,
    bbox_weights,
    outlier_threshold,
    max_iter,
    init_params=None,
    n_rounds=3,
    rot_only=False,
):
    idx_both = np.where(valid_both)[0]
    if init_params is not None:
        params = np.asarray(init_params, dtype=float).copy()
    else:
        slam_t = slam_poses[idx_both, :3, 3].T
        anchor_t = anchor_poses[idx_both, :3, 3].T
        s_init, R_init, t_init = umeyama_alignment(slam_t, anchor_t)
        if s_init <= 1e-12:
            # A stationary anchor cannot initialize scale from translation.
            # Keep a finite unit-scale start for the joint rotation/translation fit.
            s_init = 1.0
            t_init = anchor_t.mean(axis=1) - R_init @ slam_t.mean(axis=1)
        params = sim3_to_params(sim3_matrix(s_init, R_init, t_init))

    def residuals(p, idx, weights):
        return (
            sim3_residual_batch(p, anchor_poses[idx], slam_poses[idx], rot_only) * weights[:, None]
        ).ravel()

    inlier = np.ones(len(idx_both), dtype=bool)
    for _ in range(n_rounds):
        cur_idx = idx_both[inlier]
        w = bbox_weights[cur_idx]
        w = w / w.mean()
        result = least_squares(
            residuals,
            params,
            args=(cur_idx, w),
            method="lm",
            max_nfev=max_iter * 7,
            ftol=1e-10,
            xtol=1e-10,
        )
        params = result.x
        norms = np.linalg.norm(
            sim3_residual_batch(params, anchor_poses[idx_both], slam_poses[idx_both], rot_only),
            axis=1,
        )
        thr = outlier_threshold * max(np.median(norms[inlier]), 1e-06)
        new_inlier = norms < thr
        if new_inlier.sum() == inlier.sum():
            break
        if new_inlier.sum() < 3:
            break
        inlier = new_inlier
    return (params_to_sim3(params), params, inlier.sum())


def running_unary_weights(areas: np.ndarray, power: float) -> np.ndarray:
    n = len(areas)
    a_norm = areas / np.maximum(np.cumsum(areas) / np.arange(1, n + 1), 1e-12)
    w_raw = np.power(a_norm, power)
    return w_raw / np.maximum(np.cumsum(w_raw) / np.arange(1, n + 1), 1e-12)


def _fusion_steps(
    anchor_poses,
    slam_poses,
    valid_anchor,
    valid_slam,
    areas,
    slam_weight,
    rot_scale,
    max_iter,
    lag,
    bbox_power,
    bbox_power_trans,
    robust_loss,
    robust_fscale,
    disagree_deg,
    disagree_downweight,
    warmup,
    sim3_every,
    outlier_threshold,
    stage1_iter,
    warm_poses=None,
    warm_until=0,
    slam_weight_trans=None,
    stage1_rot_only=False,
    reanchor_after=0,
):
    n = len(anchor_poses)
    valid_both = valid_anchor & valid_slam
    if bbox_power_trans is None:
        bbox_power_trans = bbox_power
    w_rot_all = running_unary_weights(areas, bbox_power)
    w_tr_all = running_unary_weights(areas, bbox_power_trans)
    if slam_weight_trans is None:
        slam_weight_trans = slam_weight
    disagree_rad = np.deg2rad(disagree_deg) if disagree_deg >= 0.0 else None
    X_out = np.full((n, 4, 4), np.nan)
    free = {}
    params_t, n_joint, last_fit = (None, 0, -(10**9))
    fit_from, gate_run, reanchors = (0, 0, [])
    gate_prev = None

    def metric(i):
        if params_t is None or not valid_slam[i]:
            return None
        return fuse_one_pose_global(params_t, slam_poses[i])

    def solve(free_frames, anchor):
        slot = {f: k for k, f in enumerate(free_frames)}
        p0 = np.concatenate([se3_to_6vec(free[f]) for f in free_frames])
        met = {f: metric(f) for f in free_frames}
        if anchor is not None:
            met[anchor[0]] = metric(anchor[0])
        unary = []
        for f in free_frames:
            if not valid_anchor[f]:
                continue
            wr, wt = (w_rot_all[f], w_tr_all[f])
            if disagree_rad is not None and met[f] is not None:
                R_d = anchor_poses[f, :3, :3] @ met[f][:3, :3].T
                if np.linalg.norm(rotation_log(R_d)) > disagree_rad:
                    wr, wt = (wr * disagree_downweight, wt * disagree_downweight)
            unary.append((slot[f], anchor_poses[f], wr, wt))
        binary = []
        chain = ([anchor[0]] if anchor is not None else []) + list(free_frames)
        for a, b in zip(chain[:-1], chain[1:]):
            if met.get(a) is None or met.get(b) is None:
                continue
            dT = met[b] @ se3_inverse(met[a])
            binary.append((slot.get(a), slot[b], dT))
        if not unary and (not binary):
            return

        def residuals(p):
            Xs = [vec6_to_se3(p[6 * k : 6 * k + 6]) for k in range(len(free_frames))]
            out = []
            for k, M, wr, wt in unary:
                R_err = M[:3, :3] @ Xs[k][:3, :3].T
                out.append(
                    np.concatenate(
                        [wr * rotation_log(R_err) * rot_scale, wt * (M[:3, 3] - Xs[k][:3, 3])]
                    )
                )
            for ki, kj, dT in binary:
                Xi = anchor[1] if ki is None else Xs[ki]
                err = dT @ se3_inverse(Xs[kj] @ se3_inverse(Xi))
                out.append(
                    np.concatenate(
                        [
                            slam_weight * rotation_log(err[:3, :3]) * rot_scale,
                            slam_weight_trans * err[:3, 3],
                        ]
                    )
                )
            return np.concatenate(out)

        kw = dict(method="trf", max_nfev=max_iter, ftol=1e-08, xtol=1e-08, gtol=1e-10)
        if robust_loss != "linear":
            kw["loss"] = robust_loss
            kw["f_scale"] = robust_fscale
        res = least_squares(residuals, p0, **kw)
        for f, k in slot.items():
            free[f] = vec6_to_se3(res.x[6 * k : 6 * k + 6])

    payload = yield None
    while payload is not None:
        anchor_poses, slam_poses, valid_anchor, valid_slam, areas = payload
        n = len(anchor_poses)
        t = n - 1
        valid_both = valid_anchor & valid_slam
        w_rot_all = running_unary_weights(areas, bbox_power)
        w_tr_all = running_unary_weights(areas, bbox_power_trans)
        if len(X_out) < n:
            X_out = np.concatenate([X_out, np.full((max(n, len(X_out)), 4, 4), np.nan)])
        if valid_both[t]:
            n_joint += 1
        if n_joint >= 3 and (params_t is None or t - last_fit >= sim3_every):
            mask = valid_both.copy()
            mask[t + 1 :] = False
            mask[:fit_from] = False
            _, params_t, _ = stage1_global_sim3(
                anchor_poses,
                slam_poses,
                mask,
                w_rot_all,
                outlier_threshold,
                stage1_iter,
                init_params=params_t,
                n_rounds=3 if params_t is None else 1,
                rot_only=stage1_rot_only,
            )
            last_fit = t
        if t < warm_until:
            if warm_poses is not None and (not np.any(np.isnan(warm_poses[t]))):
                X_out[t] = warm_poses[t]
            elif valid_anchor[t]:
                X_out[t] = anchor_poses[t]
            continue
        if reanchor_after > 0 and valid_anchor[t] and (disagree_rad is not None):
            m_t = metric(t)
            if (
                m_t is None
                or np.linalg.norm(rotation_log(anchor_poses[t, :3, :3] @ m_t[:3, :3].T))
                <= disagree_rad
            ):
                gate_run, gate_prev = (0, None)
            else:
                coherent = False
                if gate_prev is not None:
                    m_p = metric(gate_prev[0])
                    if m_p is not None:
                        pred = m_t[:3, :3] @ m_p[:3, :3].T @ gate_prev[1][:3, :3]
                        coherent = (
                            np.linalg.norm(rotation_log(anchor_poses[t, :3, :3] @ pred.T))
                            <= disagree_rad
                        )
                gate_run = gate_run + 1 if coherent else 1
                gate_prev = (t, anchor_poses[t].copy())
            if gate_run >= reanchor_after:
                free.clear()
                params_t, n_joint, last_fit = (None, 0, -(10**9))
                fit_from, gate_run, gate_prev = (t, 0, None)
                reanchors.append(int(t))
        prev = max(free) if free else t - 1 if t > 0 else None
        prev_pose = free[prev] if prev in free else X_out[prev] if prev is not None else None
        if prev_pose is not None and np.any(np.isnan(prev_pose)):
            prev_pose = None
        init = None
        if params_t is not None and n_joint >= warmup:
            m_t, m_p = (metric(t), metric(prev) if prev is not None else None)
            if m_t is not None and m_p is not None and (prev_pose is not None):
                init = m_t @ se3_inverse(m_p) @ prev_pose
            elif valid_anchor[t]:
                init = anchor_poses[t].copy()
            elif m_t is not None:
                init = m_t
        elif valid_anchor[t]:
            init = anchor_poses[t].copy()
        if init is not None:
            free[t] = init
            if params_t is not None and n_joint >= warmup:
                anchor_f = min(free) - 1
                anchor = None
                if anchor_f >= 0 and (not np.any(np.isnan(X_out[anchor_f]))):
                    anchor = (anchor_f, X_out[anchor_f])
                solve(sorted(free), anchor)
        e = t - lag
        if e >= 0:
            if e in free:
                X_out[e] = free.pop(e)
            elif valid_anchor[e]:
                X_out[e] = anchor_poses[e]
        payload = yield (
            X_out[t].copy(),
            dict(
                ready=params_t is not None and n_joint >= warmup,
                joint_observations=int(n_joint),
                reanchors=list(reanchors),
            ),
        )
