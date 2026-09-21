# Pico-HAT

Hypothesis-Anchored Tracking for monocular RGB sequences, using PicoPose
for absolute pose hypotheses and DROID-SLAM for relative motion.

This repository implements Pico-HAT from
[HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation](https://arxiv.org/pdf/2609.21597)
by André Lopo, Atabak Dehban and Rodrigo Ventura.

The default is the **shared causal Pico-HAT configuration from the paper**.
Each frame uses only observations available at that frame. Previously emitted
poses stay fixed. The pipeline keeps PicoPose stages 1–3, forward hypothesis
selection, the running confidence gate, Sim(3) alignment, SE(3) fusion,
disagreement handling, reanchoring, and hold filling.

## Inputs

- An ordered RGB image sequence of one rigid target.
- Calibrated camera intrinsics for the supplied image resolution.
- A target mask or a bounding box for each available detection.
- A CAD triangle mesh with known scale. OBJ with MTL and textures, GLB/glTF,
  PLY, and STL are supported by the mesh loader. Textures or vertex colors
  provide appearance cues. Geometry-only models can be rendered, but pose
  accuracy depends on how well their appearance matches the images.

Images must be undistorted. Keep OBJ material and texture files beside the
model, with their relative paths intact. Convert STEP, IGES, or proprietary
CAD formats to a triangle mesh before use.

## Installation

See [INSTALL.md](docs/INSTALL.md) for the pinned source revisions, separate
GPU environments, CUDA extensions, and upstream checkpoints. The temporal
package can also be installed without GPU dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install .
```

## Run a sequence

Prepare templates once per object in the PicoPose environment:

```bash
.venv-pico/bin/pico-hat prepare \
  --mesh data/object/model.obj --units mm \
  --pico-source third_party/PicoPose --out templates/object
```

Create `data/sequence/sequence.json` using
[the mask example](configs/sequence_masks.json) or
[the bounding-box example](configs/sequence_boxes.json).
Replace the example camera calibration with your own. Paths are relative
to the JSON file. RGB images are naturally sorted, so `2.png` precedes `10.png`.

```text
data/sequence/
  sequence.json
  rgb/000001.png
  rgb/000002.png
  masks/000001.png
  masks/000002.png
```

Masks are single-channel PNGs with the same stem and resolution as the RGB
image. Zero means background and any nonzero value means the target.
For boxes, provide a JSON object keyed by the complete RGB filename:

```json
{
  "000001.png": [120, 80, 420, 360],
  "000002.png": [122, 81, 422, 361],
  "000003.png": null
}
```

Boxes use pixel coordinates `[x1, y1, x2, y2]`, with exclusive upper edges.
A missing entry, missing mask, empty mask, or `null` box means no detection.
It does not remove the RGB frame. The next eligible detection can produce an
anchor once six frames have elapsed since the previous attempted anchor.

Set the interpreter, source and checkpoint paths in
[configs/runtime.json](configs/runtime.json), then run:

```bash
.venv/bin/pico-hat validate data/sequence/sequence.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/pico-hat run \
  --sequence data/sequence/sequence.json \
  --runtime configs/runtime.json \
  --templates templates/object --out outputs/sequence
```

`poses.jsonl` is written as frames are processed. `poses.npz` contains an
`N × 4 × 4` array and the corresponding image names. Poses transform model
coordinates to the OpenCV camera frame: **x right, y down, z forward**, with
translations in **metres**. The model origin and axes are preserved through
template rendering. For glTF scenes, node transforms are applied as stored.

Frames before the first valid pose have `null` in JSON and NaNs in NumPy.
Later gaps hold the last valid pose and are marked `held` in the JSON output.
Logs, the resolved configuration and raw observations accompany every run.
Existing output directories are never overwritten.

## Shared settings

| Parameter | Value |
|---|---:|
| Minimum anchor gap | 6 frames |
| Candidates per anchor | 5 |
| Unary scale | 100 |
| Motion transition weight | 0.3 |
| Fusion motion weight | 0.1 |
| Disagreement gate | 30° |
| Selection / fusion warm-up | 3 / 4 observations |
| Huber scale | 0.3 |
| Alignment update interval | 10 frames |
| Coherent disagreements before reanchoring | 4 anchors |

The complete settings are packaged in the [shared configuration](src/pico_hat/shared.json)
and printed by `pico-hat config`. No dataset-specific temporal overrides are
applied. [Method notes](docs/METHOD.md) describe input policies, coordinate
conventions, and the distinction between this shared causal method and the
separate offline studies.

## Replay and Python API

```bash
.venv/bin/pico-hat replay outputs/sequence/observations.npz \
  --out outputs/replayed.npz
```

```python
from pico_hat import Tracker

tracker = Tracker()
# slam_T_wc: current frozen camera-to-world motion estimate, shape (4, 4).
# candidates: (object-to-camera poses [5, 4, 4], inlier scores [5]), or None.
pose, status = tracker.step(slam_T_wc, candidates)
tracker.close()
```

Call `step` once per RGB frame. Use `tracker.anchor_due` to decide whether
to query PicoPose. The API does not perform detection or load neural models.

## Validation and attribution

[VALIDATION.md](docs/VALIDATION.md) records release checks and their scope.
CPU tests run in CI. GPU inference requires the upstream weights and compiled
DROID extensions.

This implementation builds on [PicoPose](https://github.com/foollh/PicoPose)
and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).
Please cite those methods when using their models. Their sources and weights
are obtained separately and retain their own terms. See
[THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If you use this implementation, please cite the [HAT paper](https://arxiv.org/pdf/2609.21597)
and the upstream methods listed above.

```bibtex
@article{lopo2026hat,
  title={HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation},
  author={Lopo, Andr{\'e} and Dehban, Atabak and Ventura, Rodrigo},
  journal={arXiv preprint arXiv:2609.21597},
  year={2026},
  doi={10.48550/arXiv.2609.21597},
  url={https://arxiv.org/pdf/2609.21597}
}
```
