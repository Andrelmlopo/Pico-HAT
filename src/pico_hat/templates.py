"""Render a textured mesh in its original object coordinate system."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

TEMPLATE_K = np.array([[572.4114, 0, 320], [0, 573.57043, 240], [0, 0, 1]], dtype=float)


def render_templates(mesh_path, units, pico_root, output):
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            "Run prepare in the Python 3.9 PicoPose environment; pyrender's OpenGL binding does not support Python 3.12+."
        )
    # EGL supports headless rendering; OSMesa can be selected by the caller.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    mesh_path, pico_root, output = Path(mesh_path), Path(pico_root), Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    asset = trimesh.load(mesh_path, force="scene", process=False)
    if not asset.geometry:
        raise ValueError("The mesh contains no geometry")
    # pyrender interprets integer colors as 0--255, floating colors as 0--1.
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
    vertices = []
    for name in asset.graph.nodes_geometry:
        transform, geometry = asset.graph[name]
        mesh = asset.geometry[geometry].copy()
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("Expected a triangle mesh, not a point cloud")
        vertices.append(np.asarray(mesh.vertices))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    points = np.concatenate(vertices)
    if not np.isfinite(points).all():
        raise ValueError("Mesh vertices must be finite")
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max())
    if radius <= 0:
        raise ValueError("The mesh has zero extent")
    distance = 2 * radius * TEMPLATE_K[1, 1] / 180
    grid_path = pico_root / "rendering/src/lib3d/predefined_poses/obj_poses_level1.npy"
    grid = np.load(grid_path, allow_pickle=False)
    if grid.shape != (162, 4, 4):
        raise ValueError(f"Expected the native 162-view grid at {grid_path}")
    camera = pyrender.IntrinsicsCamera(
        fx=TEMPLATE_K[0, 0],
        fy=TEMPLATE_K[1, 1],
        cx=320,
        cy=240,
        znear=max(radius * 1e-3, 1e-6),
        zfar=distance + 4 * radius,
    )
    camera_node = scene.add(camera)
    renderer = pyrender.OffscreenRenderer(640, 480)
    rgb, depth, poses = [], [], []
    try:
        for item in grid:
            pose = np.eye(4)
            pose[:3, :3] = item[:3, :3]
            # Center the view by translating the camera, preserving the CAD origin.
            pose[:3, 3] = [0, 0, distance] - pose[:3, :3] @ center
            scene.set_pose(camera_node, np.linalg.inv(pose) @ np.diag([1, -1, -1, 1]))
            color, z = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            if np.count_nonzero(z > 0) < 8:
                raise RuntimeError("A template is empty; check mesh geometry and materials")
            rgb.append(color[..., :3])
            depth.append(z.astype(np.float32))
            poses.append(pose)
    finally:
        renderer.delete()
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "templates.npz",
        rgb=np.array(rgb),
        depth=np.array(depth),
        poses=np.array(poses),
        K=TEMPLATE_K,
    )
    metadata = {
        "format": "pico-hat.templates.v1",
        "mesh": mesh_path.name,
        "mesh_units": units,
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "view_grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "views": 162,
        "translation_units": "m",
        "pose_convention": "object-to-camera",
        "object_frame": "source scene with node transforms applied; no recentering or axis conversion",
        "renderer": "pyrender",
        "radius_m": radius,
        "camera_distance_m": distance,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
