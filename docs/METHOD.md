# Method and conventions

The shared configuration follows the Pico-HAT column of the parameter table
in the [HAT paper](https://arxiv.org/pdf/2609.21597): gap 6, five raw PicoPose hypotheses, unary scale 100,
motion transition weight 0.3, fusion weight 0.1, disagreement threshold 30°,
fusion warm-up 4 and Huber scale 0.3. No MegaPose refiner is used.

At each eligible detection, PicoPose's native template retrieval, affine
regression and correspondence refinement produce five pose hypotheses.
PnP inlier ratios supply their scores. The shared geometric guard rejects
invalid rotations, nonpositive depth, zero inliers and translations outside
0.001–100 m. With at least three valid candidates, depths must lie within
a factor of three of their median. This range is part of the published
configuration and may be unsuitable for a different physical scene scale.

Forward dynamic programming combines scaled inlier scores with geodesic
rotation differences in degrees. For camera-to-world DROID rotations,
the object rotation increment is `R_wc(current).T @ R_wc(previous)`.
The running confidence gate checks at least three valid opportunities,
a median of two candidates within 15° of the top candidate, and an override
fraction of at most 0.7. If it rejects temporal selection, the best valid
raw hypothesis supplies that anchor.

Sim(3) alignment and SE(3) fusion follow the original numerical residuals.
Only the current pose is free. Alignment is fitted from the observed prefix
and refreshed every ten frames when enough joint observations exist.
The fusion warm-up requires four joint anchor/motion observations. A 30°
disagreement downweights an anchor by 0.05. Four mutually coherent
disagreeing anchors reset the alignment for subsequent output. Missing
output holds the latest valid pose.

DROID runs its frontend and local optimization. The adapter never calls
the offline `terminate()` path. Non-keyframes use the most recent two
keyframes and three motion-only updates. The returned trajectory is frozen
at each frame, even if DROID subsequently revises its internal map.

## Localization and motion inputs

This package accepts supplied detections. Detector generation and
Mask2Former/CNOS installation are outside the runner. A mask identifies
one target. Bounding boxes create rectangular target masks for PicoPose.
Masks from multiple objects must be separated before use.

By default, DROID sees the masked RGB image when pixel masks are provided,
and the full RGB image when only boxes are provided. `slam_input` can set
either policy explicitly. When masking is requested and the target is
missing, DROID receives no update for that frame and the motion observation
is marked missing. Strong background motion in unmasked videos may not
represent the target's motion.

`slam_resolution` is the resize pixel area. Intrinsics are scaled before
cropping the bottom/right edges to multiples of eight. The default is
196608. The [HAT paper](https://arxiv.org/pdf/2609.21597) used native dataset input policies, including a larger
393216 area for SwissCube. These are input preparation choices rather than
changes to the shared temporal settings.

## CAD and output frame

Template depth and translations are stored in metres. The renderer loads
scene-node transforms and unit scaling, then moves the camera to frame the
object. It does not shift the object's origin or apply dataset axis fixes.
OpenGL camera axes are converted explicitly to the OpenCV camera convention.
Output obeys `point_camera = R @ point_model_metres + t`.

The release uses a standalone pyrender template generator with the native
162-view PicoPose grid. The paper's precomputed template banks used other
rendering tools and dataset-specific frame preparation. New templates are
therefore not expected to reproduce all published frontend predictions or
benchmark scores. The temporal code is separately checked against saved
paper observations. Exact benchmark reproduction also requires the same
images, detections, template banks, camera calibration and checkpoints.

This release implements the shared **causal** method. The first-half tuned
offline Pico-HAT study, later thesis localization rules, and SLAM-free
ablations are distinct configurations and are not selected by this runner.

## Degenerate inputs

Before an initial valid anchor, no absolute pose is available. A stationary
anchor with a moving SLAM trajectory cannot initialize translation scale
through Umeyama alignment. In that case the implementation supplies a
finite unit-scale initial guess to the joint fit. This avoids the original
zero-scale division without changing nondegenerate replay results.

The DROID map has a finite keyframe buffer. Exhausting it raises an explicit
error. Increase `droid_buffer` for longer sequences if GPU memory permits.
This implementation does not claim bounded-memory operation on indefinite
streams.
