"""Optional EGL renderer check: PICO_HAT_RENDER_TESTS=1 pytest tests/test_rendering.py."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PICO_HAT_RENDER_TESTS") != "1", reason="Requires EGL and renderer dependencies"
)


def test_obj_texture_units_and_original_object_frame(tmp_path):
    import trimesh
    from PIL import Image

    from pico_hat.templates import render_templates

    # An off-center cuboid makes silent centering or axis changes observable.
    mesh = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
    mesh.apply_translation([0.4, -0.2, 0.15])
    uv = (mesh.vertices[:, :2] - mesh.vertices[:, :2].min(0)) / np.ptp(mesh.vertices[:, :2], axis=0)
    texture = np.full((16, 16, 3), [240, 70, 20], np.uint8)
    texture[:, 8:] = [20, 200, 100]
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.fromarray(texture))
    mesh.export(tmp_path / "model.obj")
    # The renderer needs only the native grid. A deterministic grid suffices
    # for this independent geometry test, without fetching neural sources.
    source = tmp_path / "source/rendering/src/lib3d/predefined_poses"
    source.mkdir(parents=True)
    poses = np.repeat(np.eye(4)[None], 162, axis=0)
    np.save(source / "obj_poses_level1.npy", poses)
    render_templates(tmp_path / "model.obj", "m", tmp_path / "source", tmp_path / "bank_m")
    with np.load(tmp_path / "bank_m/templates.npz") as archive:
        depth, K, pose, rgb = (
            archive[k][0] if k != "K" else archive[k] for k in ("depth", "K", "poses", "rgb")
        )
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs]
    camera = np.stack(((xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z), axis=1)
    model = (camera - pose[:3, 3]) @ pose[:3, :3]
    assert np.max(np.abs(model[:, 2] - 0.10)) < 0.001
    # Rasterization can include partially covered boundary pixels.
    pixel_tolerance = 2 * float(z.max()) / min(K[0, 0], K[1, 1])
    assert model[:, 0].min() > 0.25 - pixel_tolerance
    assert model[:, 0].max() < 0.55 + pixel_tolerance
    assert model[:, 1].min() > -0.30 - pixel_tolerance
    assert model[:, 1].max() < -0.10 + pixel_tolerance
    assert np.std(rgb[ys, xs, 0]) > 20
    mesh.apply_scale(1000)
    mesh.export(tmp_path / "model_mm.obj")
    render_templates(tmp_path / "model_mm.obj", "mm", tmp_path / "source", tmp_path / "bank_mm")
    with np.load(tmp_path / "bank_mm/templates.npz") as archive:
        np.testing.assert_allclose(archive["poses"][0], pose, atol=1e-8)
        np.testing.assert_allclose(archive["depth"][0], depth, atol=1e-6)
