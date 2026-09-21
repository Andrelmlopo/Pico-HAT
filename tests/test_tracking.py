import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pico_hat import Tracker, shared_config
from pico_hat.fusion import rotation_log, se3_inverse
from pico_hat.geometry import valid_pose


def observations(n=48):
    motion, candidates = [], {}
    for i in range(n):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0.003 * i, 0.008 * i, -0.002 * i]).as_matrix()
        pose[:3, 3] = [0.004 * i, 0.03 * np.sin(i / 6), 2 + 0.002 * i]
        motion.append(se3_inverse(pose))
        if i % 6 == 0:
            bank = np.repeat(pose[None], 5, axis=0)
            bank[3:, :3, :3] = bank[3:, :3, :3] @ Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
            candidates[i] = (bank, np.array([0.9, 0.88, 0.87, 0.3, 0.2]))
    return np.array(motion), candidates


def execute(motion, candidates, capacity=2):
    tracker = Tracker(capacity)
    try:
        return np.array([tracker.step(s, candidates.get(i))[0] for i, s in enumerate(motion)])
    finally:
        tracker.close()


def test_pose_direction_and_metric_scale():
    motion, candidates = observations()
    result = execute(motion, candidates)
    expected = np.array([se3_inverse(x) for x in motion])
    assert valid_pose(result).all()
    np.testing.assert_allclose(result[18:], expected[18:], atol=1e-5)


def test_outputs_are_causal_and_independent_of_preallocation():
    motion, candidates = observations()
    full = execute(motion, candidates)
    for stop in (1, 13, 19, 33):
        np.testing.assert_array_equal(execute(motion[:stop], candidates, 128), full[:stop])
    # Changing observations in the future must not affect the emitted prefix.
    modified = {**candidates, 36: (candidates[36][0][::-1], candidates[36][1])}
    np.testing.assert_array_equal(execute(motion, modified)[:36], full[:36])


def test_missing_initial_detection_and_failed_opportunity():
    missing = np.full((4, 4), np.nan)
    tracker = Tracker(1)
    try:
        pose, status = tracker.step(missing)
        assert np.isnan(pose).all() and not status["valid"]
        _, bank = observations(1)
        pose, status = tracker.step(missing, bank[0])
        assert status["valid"]
        original = pose.copy()
        pose[:] = 0  # Returned arrays must not alias internal state.
        for _ in range(5):
            pose, status = tracker.step(missing)
            np.testing.assert_array_equal(pose, original)
            assert status["held"]
        pose, status = tracker.step(missing, (bank[0][0], np.zeros(5)))
        np.testing.assert_array_equal(pose, original)
        assert not tracker.anchor_due
        assert status["selection"]["valid_candidates"] == 0
    finally:
        tracker.close()


def test_invalid_hypotheses_cannot_vote_or_be_selected():
    motion, bank = observations(1)
    poses, scores = bank[0]
    poses[0, 2, 3] = -1
    poses[1, :3, :3] = 0
    poses[3, 2, 3] = 900
    tracker = Tracker()
    result, status = tracker.step(motion[0], (poses, scores))
    tracker.close()
    assert status["selection"]["valid_candidates"] == 2
    np.testing.assert_array_equal(result, poses[2])


def test_bad_scores_and_early_anchors_are_rejected():
    motion, bank = observations(2)
    tracker = Tracker()
    with pytest.raises(ValueError, match="inlier ratios"):
        tracker.step(motion[0], (bank[0][0], np.ones(5) * 100))
    tracker.step(motion[0], bank[0])
    with pytest.raises(ValueError, match="gap"):
        tracker.step(motion[1], bank[0])
    tracker.close()


def test_rotation_log_at_zero_pi_and_random_rotations():
    rotations = Rotation.from_rotvec([[0, 0, 0], [np.pi, 0, 0], [0, -np.pi, 0], [0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        rotation_log(rotations.as_matrix()), rotations.as_rotvec(), atol=1e-12
    )
    for matrix, expected in zip(rotations.as_matrix(), rotations.as_rotvec()):
        np.testing.assert_allclose(rotation_log(matrix), expected, atol=1e-12)


def test_shared_config_is_returned_by_value():
    config = shared_config()
    config["fusion"]["slam_weight"] = 999
    assert shared_config()["fusion"]["slam_weight"] == 0.1


def test_stationary_anchors_with_nonstationary_slam_do_not_initialize_zero_scale():
    motion, bank = observations(30)
    constant = bank[0]
    result = execute(motion, {i: constant for i in bank})
    assert valid_pose(result).all()
