import json

import numpy as np
import pytest
from PIL import Image

from pico_hat.sequence import Sequence


@pytest.fixture
def manifest(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "masks").mkdir()
    for name in ("10.png", "2.png", "1.png"):
        Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(tmp_path / "rgb" / name)
    data = dict(
        rgb="rgb",
        masks="masks",
        camera=dict(width=32, height=24, K=[[50, 0, 16], [0, 50, 12], [0, 0, 1]]),
    )
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(data))
    return path, data


def test_natural_order_mask_and_missing_detection(manifest):
    path, _ = manifest
    mask = np.zeros((24, 32), np.uint8)
    mask[3:12, 4:15] = 255
    Image.fromarray(mask).save(path.parent / "masks/2.png")
    sequence = Sequence(path)
    assert [p.name for p in sequence.frames] == ["1.png", "2.png", "10.png"]
    assert sequence.localization(0) is None
    assert sequence.localization(1).sum() == 99
    assert sequence.validate()["detections"] == 1


def test_boxes_are_xyxy_and_float_edges_enclose_pixels(manifest):
    path, data = manifest
    del data["masks"]
    data["boxes"] = "boxes.json"
    path.write_text(json.dumps(data))
    (path.parent / "boxes.json").write_text(json.dumps({"1.png": [2.2, 3.1, 10.1, 14.8]}))
    sequence = Sequence(path)
    assert sequence.slam_input == "rgb"
    assert sequence.localization(0).sum() == 9 * 12
    assert sequence.validate()["detections"] == 1


@pytest.mark.parametrize(
    "field,value", [("width", 99), ("K", [[0, 0, 16], [0, 50, 12], [0, 0, 1]])]
)
def test_camera_errors_are_caught(manifest, field, value):
    path, data = manifest
    data["camera"][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Sequence(path).validate()


def test_box_typo_and_out_of_bounds_are_errors(manifest):
    path, data = manifest
    del data["masks"]
    data["boxes"] = "boxes.json"
    path.write_text(json.dumps(data))
    boxes = path.parent / "boxes.json"
    boxes.write_text(json.dumps({"bad.png": [1, 2, 10, 12]}))
    with pytest.raises(ValueError, match="unknown RGB"):
        Sequence(path)
    boxes.write_text(json.dumps({"1.png": [1, 2, 100, 12]}))
    with pytest.raises(ValueError, match="outside"):
        Sequence(path).validate()


def test_masks_cannot_have_wrong_resolution(manifest):
    path, _ = manifest
    Image.fromarray(np.ones((12, 16), np.uint8)).save(path.parent / "masks/1.png")
    with pytest.raises(ValueError, match="resolution"):
        Sequence(path).validate()
