import json

import numpy as np

from pico_hat.cli import main


def test_config_runs_from_an_unrelated_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["pico-hat", "config"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["anchor_gap"] == 6


def test_replay_preserves_missing_initial_frames_and_refuses_overwrite(tmp_path, monkeypatch):
    motion = np.full((8, 4, 4), np.nan)
    candidates = np.repeat(np.eye(4)[None, None], 5, axis=1)
    candidates[:, :, 2, 3] = 2
    observations = tmp_path / "observations.npz"
    np.savez(
        observations,
        slam=motion,
        frames=np.array([1]),
        candidates=candidates,
        scores=np.ones((1, 5)),
    )
    output = tmp_path / "exact-name.npz"
    monkeypatch.setattr("sys.argv", ["pico-hat", "replay", str(observations), "--out", str(output)])
    assert main() == 0
    with np.load(output) as archive:
        assert np.isnan(archive["poses"][0]).all()
        assert np.isfinite(archive["poses"][1:]).all()
    original = output.read_bytes()
    assert main() == 1
    assert output.read_bytes() == original
