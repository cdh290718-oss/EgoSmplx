# Early MANO and visible shoulder/arm observations (opt-in experiment)

Baseline: current rigid-head soft reference, original 13 candidate body points,
original downstream wrist/contour/MANO-guided/original-seam pipeline.

A disables early_observations. B adds native-image MANO wrist, four MCPs, and
16 wrist-loop projections in stage two. C additionally adds shoulder, upper-arm,
and forearm visible exterior constraints. No extra Sapiens body keypoints,
manual observations, metric MANO depth, free vertex deformation or shape changes.
Downstream optimizers remain unchanged for this ablation.

Hand assignment uses automatic Sapiens wrists with rigid-result fallback and is
frozen before stage two. Detector score >= .5, at least 16 in-image joints and
finite in-image wrist loop required for early hand constraints. This is a quality
gate, not a calibrated guarantee of joint accuracy. Weights are wrist 1, palm .15,
loop .5, normalized smooth-L1 beta .003 in image coordinates, averaged per hand.

Contours use the union of hand/arm/torso/upper-clothing classes. High-confidence
person/background boundaries inside automatic shoulder-elbow-wrist regions are
eligible; image cuts and immediate hand vicinity are excluded. Shoulder radius
uses shoulder span so clothing width is not assumed equal to short projected
upper-arm length; shoulder candidates must be closer along the arm to shoulder
than elbow. Upper clothing only provides weak containment (.15), not surface
matching. Skin boundaries additionally use nearest visible model contour samples.
This is not a certified object occlusion mask. Softmax is not calibrated accuracy.

Model contour uses central-camera facing-change edges, filtered against the
quarter-resolution exterior silhouette of the full projected body, refreshed
every 25 iterations. Visibility is approximate (not an exact depth-buffer renderer).
At most 96 model edge midpoints and 64 target pixels per region are used.
Contour group means are weighted .5. Missing model contour for an observed region
costs .05 instead of silently zeroing its term. Discrete correspondence updates
make the optimization nonconvex; no global convergence guarantee is asserted.

All priors, 19 active body joints, original bounds, rigid-head weight, optimizer
steps and original final seam are held fixed. These experiments do not also test
freezing downstream non-arm joints. A full rerun checks the disabled route against
the previous two-center artifacts. Set early_observations.enabled=false to revert.
Checkpoint: checkpoint/pre-early-mano-upper-contour-20261013 (687bd87).
