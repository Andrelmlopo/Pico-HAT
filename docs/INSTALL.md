# Installation

GPU inference supports Linux with an NVIDIA CUDA GPU. The release was exercised
on an A100 with CUDA 11.8 wheels. Smaller GPUs have not been profiled. Both
networks stay resident, and DROID map memory grows with its keyframe buffer.
The DROID extensions require a CUDA toolkit, a compatible C++ compiler, and
`nvcc` on `PATH`. A driver alone is not enough to compile them.

Run these commands from the repository root. The environments are separate
because PicoPose uses PyTorch 2.0 and NumPy 1.26, while the validated DROID
installation uses PyTorch 2.7 and NumPy 2.2.

## Source repositories

```bash
python3 scripts/fetch_sources.py
```

This fetches pinned revisions without copying third-party source into the
Pico-HAT Git history:

| Source | Revision |
|---|---|
| PicoPose | `543f8fe0dc8fe014a602113977fc034bd9b4578d` |
| DROID-SLAM | `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99` |

An existing checkout must match the revision and have no tracked changes.
The helper does not reset or overwrite a checkout. DROID submodules are fetched
at the revisions recorded by that commit.

## CPU coordinator

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install .
.venv/bin/pico-hat config
```

## PicoPose and rendering

```bash
python3.9 -m venv .venv-pico
.venv-pico/bin/python -m pip install --upgrade 'pip<25' 'setuptools<81' wheel
.venv-pico/bin/python -m pip install \
  torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-pico/bin/python -m pip install xformers==0.0.18
.venv-pico/bin/python -m pip install -r requirements-pico.txt
.venv-pico/bin/python -m pip install --no-deps .
.venv-pico/bin/python -m pip check
```

The adapter uses native PicoPose inference. `mmcv-lite` supplies `ConvModule`
for its RAFT decoder without compiling MMCV's unused CUDA operators. Training,
BOP dataset preparation and PyTorch Lightning are not needed. DINO weights
come from the full PicoPose checkpoint, so initialization does not download
a second model.
Use the regular OpenCV wheel in this environment because MMCV declares that
distribution as a dependency. Installing both regular and headless OpenCV
would place two distributions over the same `cv2` module.

Headless rendering defaults to EGL. Install the system EGL/OpenGL runtime and
NVIDIA driver libraries. On machines with software OpenGL configured, set
`PYOPENGL_PLATFORM=osmesa` explicitly. OBJ textures are loaded through their
MTL file. Keep the material and image files at the relative paths referenced
by the OBJ/MTL. A binary glTF (`.glb`) is a convenient self-contained alternative.
Run `prepare` in this Python 3.9 environment. The PyOpenGL version required by
pyrender is incompatible with Python 3.12 and newer. CPU tracking and replay
are tested separately on newer Python versions.

## DROID-SLAM

```bash
python3.10 -m venv .venv-droid
.venv-droid/bin/python -m pip install 'setuptools<81' wheel ninja
.venv-droid/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-droid/bin/python -m pip install -r requirements-droid.txt
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/lietorch
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/pytorch_scatter
.venv-droid/bin/python -m pip install --no-build-isolation ./third_party/DROID-SLAM
.venv-droid/bin/python -m pip install --no-deps .
.venv-droid/bin/python -m pip check
```

For example, if the toolkit is installed at `/usr/local/cuda-11.8`, set
`CUDA_HOME` to that path and add its `bin` directory to `PATH` before building.
Use `MAX_JOBS=4` to limit parallel compilation on machines with limited RAM.
The runtime requires `droid_backends`, `lietorch` and `torch_scatter` to import
successfully. It does not need DROID's visualization packages.

## Checkpoints

Download the authors' checkpoints and keep them outside Git:

- [PicoPose checkpoint](https://drive.google.com/file/d/1hDDr0o4pEEHKi4QOUQ4zU0I-Ts5H2bI1/view)
  → `weights/picopose.ckpt`.
- [DROID-SLAM checkpoint](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view)
  → `weights/droid.pth`.

These are the links provided by the
[PicoPose authors](https://github.com/foollh/PicoPose) and
[DROID-SLAM authors](https://github.com/princeton-vl/DROID-SLAM).
Weights retain their upstream terms. Checkpoint loading is limited to the
model formats expected by the pinned adapters.

The validated files have these SHA-256 digests:

```text
9d1173d3db55e690e468edde453b06d47402d7b993748e4d2fe8f34a203014c0  picopose.ckpt
46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526  droid.pth
```

PicoPose's published checkpoint includes training metadata. The loader accepts
that format only when its digest matches the file above. Other checkpoints
must be tensor-only state dictionaries accepted by PyTorch's restricted loader.

The example `configs/runtime.json` resolves paths relative to its own directory.
Set `CUDA_VISIBLE_DEVICES` when launching the coordinator to choose the physical
GPU. Both workers use the same visible GPU. They start once and are stopped on
normal completion, exceptions, or keyboard interruption.

## Tests

```bash
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
```

The ordinary test suite is CPU-only. A full inference smoke test also needs
the upstream checkpoints, CUDA extensions, a model mesh, and an RGB sequence
with camera calibration and masks or boxes. The validation report distinguishes
those checks from a fresh installation of every GPU dependency.
