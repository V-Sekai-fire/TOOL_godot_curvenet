# 05 — Side toggle UI for asymmetric pull at intersections

## Why (current pain)

`IntersectionFrames::per_side_scaled_frames` already returns both `+`
and `−` sides per outgoing segment, and `cut_mesh::CutVertexKind`
already carries a `side` boolean — but no editor surface. The
deformer always pulls from the right side at intersection knots,
making asymmetric character articulation (e.g., armpit pull only
deforms one side) impossible. Medium-priority gap.

## Gall-minimum slice

- **In scope**: per-knot side flag (only meaningful at intersection
  knots), exposed two ways:
  - inspector property `knot_sides : TypedArray<PackedInt32Array>`
    (parallel to `knot_widths` — 0 = left, 1 = right; default 1)
  - 3D click target on the curve at intersection knots that flips
    the side
- **Deferred**: separate left/right widths (would need
  `knot_widths_l` and `knot_widths_w` instead of the current scalar
  `knot_widths`).
- **Why this slice**: the algorithm side is fully wired — this is
  pure UX surface for an existing capability.

## Files to touch

- `src/curvenet_deformer_3d.h` — `knot_sides` field +
  setter/getter, granular `set_knot_side(curve_id, knot_idx, side)`.
- `src/curvenet_deformer_3d.cpp` — bind step reads `knot_sides`
  per knot, passes to the per-side scaled frame builder; bind_methods
  adds the property and granular accessors.
- `src/curvenet_gizmo_plugin.cpp` — at each intersection knot, draw
  a small clickable arrow on the chosen side; click → call
  `set_knot_side` to flip.
- `tests/test_surface_projection.cpp` (or new `test_side_toggle.cpp`)
  — RC properties: side default to 1, granular setter auto-extends
  array, side flip changes the per-side frame output.

## Approach

- The deformer's current bind uses the same side for every knot
  implicitly. Wire `knot_sides[c][k]` to feed
  `cut_mesh::CutVertexKind::sample_kind(curve_id, sample_idx, side)`
  during sample promotion.
- Gizmo: at every knot classified as `intersection` (already detected
  by `curvenet_builder::classify`), draw two small arrows in the
  binormal directions, one filled (current side) and one outlined
  (other side). Click the outlined one → flip.
- For a primary handle hit-test on click, use the gizmo's
  primary-handle channel with id encoding similar to tilt/width but
  for the click-only target. Or, since side toggle isn't really a
  "drag" operation, hook the editor plugin's input event directly.
- Undo/redo: the `set_knot_side` setter is one boolean — single
  do/undo pair per click.

## Verification

- **Lean**: no change (`IntersectionFrames` already proves per-side
  correctness).
- **C++**: ~3 RC props on `knot_sides` array bookkeeping (auto-extend,
  default value, granular setter agrees with array setter).
- **GDExtension**: `scons -j8` clean.
- **Manual**: in editor, place a "+"-pattern intersection (two
  crossing curves), drag a knot at the intersection, verify the
  surface follows asymmetrically based on the chosen side.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: 02 conceptually (without intersection cracks the
  side toggle has no observable effect on a single-curve mesh — but
  the multi-curve intersection case already works via the existing
  `IntersectionFrames` machinery).

## Estimated cost

- LOC: ~200 across deformer (~80) + gizmo (~80) + tests (~40).
- Effort: small-medium.
- Risk: low. Algorithm is in place; this is plumbing.
