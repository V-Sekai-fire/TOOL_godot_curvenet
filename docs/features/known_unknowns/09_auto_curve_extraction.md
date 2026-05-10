# 09 — Auto profile-curve extraction

## Why (current pain)

Profile curves are 100% artist-authored. Demo workflows on
non-character meshes (test rigs, captured scans, asset library) need
to bootstrap curves automatically — no artist time. Without auto
extraction, every demo is bottlenecked on someone hand-placing
curves. Low/research priority gap.

## Gall-minimum slice

- **In scope**: geodesic level sets driven by the heat method (todo
  10). For each automatically detected feature point (curvature
  extrema), compute geodesic distance from it; level sets at sampled
  isovalues become candidate profile curves.
- **Deferred**: ML extraction (PIE-Net etc.) — banned by the
  no-third-party rule. Skeleton-driven extraction (Tagliasacchi 2012,
  Coverage Axis 2022) — explored but ruled out per the chat
  trajectory in favour of heat-method level sets, which stay on the
  manifold.
- **Why this slice**: leverages the heat method we'd have to write
  anyway (todo 10). The level-set extraction itself is a small wrapper
  on top.

## Files to touch

- `lean/Curvenet/AutoExtract.lean` (new) — spec the level-set extractor
  on a tiny mesh; native_decide proofs.
- `src/curvenet/auto_extract.h` (new) — C++ mirror.
- `src/curvenet_editor_plugin.cpp` — "Auto-extract curves" button in
  the deformer's inspector that runs the extractor and populates
  `profile_curves`.
- `tests/test_auto_extract.cpp` (new) — RC properties: extracted
  curves are smooth (consecutive knots ε-close), level sets are closed
  loops on closed meshes.

## Approach

- Detect feature points: per-vertex Gaussian / mean curvature, threshold
  to top-K extrema. Cheap and good enough for v0.
- For each feature point, run the heat method to get a geodesic
  distance field over the mesh.
- Sample isovalues at fixed intervals; trace the level set on each
  triangle (canonical marching-triangles / linear interpolation).
- Convert each extracted polyline into a `Curve3D` with positions only
  (no tangents/widths/tilts authored — defaults take over).
- Editor button creates the curves resource and assigns to the
  deformer's `profile_curves`. Artist can then refine.

## Verification

- **Lean**: `lake build` clean. `Curvenet.AutoExtract` adds ~6 proofs
  on small instances.
- **C++**: ~5 RC properties on closed-loop level sets + smoothness.
- **GDExtension**: `scons -j8` clean. Editor button visible on
  selection.
- **Manual**: load a test mesh (sphere, torus), click "Auto-extract
  curves", verify smooth closed-loop curves appear in the viewport.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: 10 (heat method).

## Estimated cost

- LOC: ~600 across Lean (~150) + C++ extract (~250) + editor plugin
  (~100) + tests (~100).
- Effort: large.
- Risk: medium-high. Curvature thresholds, isovalue sampling, and
  level-set tracing each have tuning knobs that affect output quality.
  Plan for an interactive sliders pass before declaring "done".
