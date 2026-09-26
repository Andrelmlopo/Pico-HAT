# Release validation

The machine-readable receipt is [validation.json](validation.json).
Checks were performed on 21 September 2026.

## Temporal equivalence

The packaged tracker reproduced all saved outputs bit for bit on eight
recorded streams: SPARK RT500, YCB-Video scene 48/object 6, five SwissCube
streams, and one SHIRT stream. This covers 5,420 input frames, including
missing observations and hold-filled output. The receipt records input
SHA-256 digests. No neural inference was used for this comparison.

## Tests and packaging

- 17 CPU tests pass on Python 3.13, NumPy 2.2.6 and SciPy 1.15.3.
- The same 17 tests pass against an installed wheel on Python 3.10,
  NumPy 1.26.4 and SciPy 1.13.1, from outside the source tree.
- The optional EGL test passes for a textured OBJ/MTL model. It checks
  visible texture colors, the original off-centre object frame, projection
  geometry and equivalent metre/millimetre input models.
- Ruff checks, formatting checks, source/wheel builds and package dependency
  checks pass. GPU model dependencies are kept out of the CPU package.
- The pinned PicoPose and DROID repositories, including DROID submodules,
  were fetched into new directories using the published source-fetch helper.

CPU tests cover pose direction, metric scale, near-zero and 180° rotations,
prefix causality, buffer growth, missing initial detections, failed anchors,
invalid hypotheses, score validation, mask/box validation, image ordering,
virtual-environment interpreter paths, replay and overwrite protection.

## GPU integration

A complete 300-frame SPARK RT500 run used a textured GLB model, newly rendered
templates, binary masks, a newly installed PicoPose environment and freshly
fetched upstream sources. It returned 300 valid poses and 50 anchor calls.
DROID produced 280 valid motion observations. Fusion was initialized on
210 frames, and 75 frames used hold filling during startup or recovery.
Replaying its saved observations produced exactly the same 300 output poses.

The bounding-box run, using rectangular masks for the DROID input, also
returned 300 valid poses and 50 anchors. DROID produced 267 valid motion
observations, fusion was initialized on 246 frames, and 45 frames used hold
filling. Its 300 poses also replay bit for bit.
Shorter 48-frame mask and box runs also exercised worker startup, model loading,
inference and shutdown during integration.

The fresh PicoPose environment uses Python 3.9, PyTorch 2.0.0+cu118,
xFormers 0.0.18, NumPy 1.26.4, SciPy 1.13.1 and MMCV-lite 2.0.0. Its
dependency check passes. The DROID environment uses Python 3.10 and
PyTorch 2.7.1+cu118. Its existing compiled CUDA extensions were reused at
the pinned source revision, and its dependency check passes. Building those
extensions on another CUDA/compiler combination was not tested.

## Scope

These checks validate the release workflow and temporal extraction. They are
not a new four-dataset accuracy benchmark, a throughput measurement, or a
guarantee for every GPU, camera, object, and dependency combination. The new
standalone renderer is not the renderer used to produce every historical
benchmark template bank. Published benchmark scores are not assigned to the
newly rendered templates.

Issues corrected during preparation include zero-scale alignment initialization,
loss of virtual-environment identity when resolving interpreter symlinks,
duplicate OpenCV distributions, an unavailable xFormers wheel index, incompatible
NumPy/mesh-loader versions, Python/OpenGL compatibility, and integer ambient
colors making template textures too dark. Regression checks cover the affected
input, geometry and runtime behavior.

## Fresh installation and subset checks, 26 September 2026

A new checkout and new Python environments were tested on an 8 GB RTX 3070
Laptop GPU. PicoPose used Python 3.9/PyTorch 2.0.0+cu118; MegaPose and DROID
used Python 3.10/PyTorch 2.7.1+cu118. All three DROID extensions were built
from pinned sources with CUDA 11.8 and GCC 11.5. Checkpoint hashes and repeated
source-helper checks passed. See the [dated receipt](release_validation_20260926.json).

The 17 CPU tests, optional EGL renderer test, installed-wheel tests, dependency
checks, lint, formatting and package builds pass. No new Pico-HAT code defect
was reproduced in this validation.

Both HAT methods ran through SpaceRGBbenchmark on 300 dense YCB-V inputs
(scene 48/object 6; five scored targets), 36 SwissCube inputs (seq_000400),
and 36 inputs from each of SHIRT's four streams. Each method emitted valid
SE(3) poses for all 480 inputs. Saved observations reproduce every pose exactly
through the API and standalone replay CLI. Five causal prefixes per stream
match exactly (60 checks across both methods).

These checks have material limits:

- SPARK 2024 was not rerun because the restricted dataset and matching inputs
  were unavailable. Earlier validation receipts are separate evidence.
- SwissCube's declared 393216-pixel DROID resolution exhausted this GPU.
  The successful development run explicitly uses 196608 pixels and buffer 64;
  this does not validate the published default on 8 GB. The frozen schedule
  and published resolution remain unchanged.
- SHIRT used a local Tango CAD whose correspondence to the historical calibrated
  origin is unverified. Its accuracy scores are not release evidence.
- YCB-V produced 138 valid DROID motion observations per method but too few
  usable anchors for fusion: 295/300 outputs were held. Pico-HAT also held
  35/36 outputs on SHIRT roe2_synthetic without initializing fusion. Valid
  poses and exact replay do not establish useful tracking accuracy.
- This is not a full-partition accuracy or repeated-FPS experiment. Metadata
  matches all 41 YCB-V streams, 100 SwissCube sequences (8,522 targets) and
  four SHIRT streams (9,484 targets), but neural inference used subsets only.
