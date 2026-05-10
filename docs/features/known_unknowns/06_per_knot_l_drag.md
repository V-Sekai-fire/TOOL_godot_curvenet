# 06 — Per-knot length scale `l` drag

## Why (current pain)

DeGoes22 §3 has two width parameters: `l` (length scale along
tangent) and `w` (cross-section width along binormal). Today only `w`
has a 3D drag handle (commit `9a8ee1f`); `l` is auto-derived from
segment length. Artists can't independently stretch/compress a knot's
length influence — it's coupled to the segment's Euclidean length.
Medium-priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#medium-priority).

## Gall-minimum slice

- **In scope**: third drag-handle kind for `l`, mirroring the existing
  tilt + width pattern. Stored as `knot_lengths` per-curve
  PackedFloat32Array on the deformer, defaults to 1.0. Composed into
  F_h at runtime as a uniform scale along the tangent.
- **Deferred**: animated `l` curves over time (todo 12 territory).
- **Why this slice**: trivial extension of the proven tilt/width
  drag-handle plumbing. ~120 LOC. Completes the §3 §3 frame UX.

## Files to touch

- `src/curvenet_deformer_3d.h` — `knot_lengths` field +
  setter/getter, granular `set_knot_length` /
  `get_knot_length`. `RestCache.rest_curve_lengths`.
- `src/curvenet_deformer_3d.cpp` — bind reads rest lengths; runtime
  composes `s_l = l_posed / l_rest` as a tangent-direction scale into
  F_h before the perpendicular scale and tilt rotation.
- `src/curvenet_gizmo_plugin.cpp` — third drag-handle kind in the
  secondary id space, positioned along the rotated +tangent at
  distance `L_HANDLE_RADIUS · l`. Drag-along-tangent → new `l`.
- `tests/test_direct_delta_mush.cpp` — extend with property: F_h
  scales tangent component by `s_l` when widths and tilts are flat.

## Approach

- Storage and granular accessor: copy/paste from the `knot_widths`
  pattern. The granular setter auto-extends the array and defaults
  to 1.0.
- Runtime: in the per-handle F_h construction (commit `189c320`'s
  width path), add a tangent-direction scale:
  `F_l = s_l · t⊗tᵀ + (I - t⊗tᵀ)`
  i.e., scale by s_l along t, identity perpendicular. Compose
  `F = F_l · F_perp · R_tilt · F_segment_isolated`.
- Gizmo handle: place at `origin + t · L_HANDLE_RADIUS · l`. Drag
  raycast plane = perpendicular to view (existing pattern). Project
  hit onto tangent, distance from origin / L_HANDLE_RADIUS = new l,
  clamped to [0.05, 50.0].
- Handle id space extends:
  ```
  [0, nT)              tangent
  [nT, nT+nK)          tilt
  [nT+nK, nT+2*nK)     width
  [nT+2*nK, nT+3*nK)   length            ← new
  ```
- `_get_handle_name` adds "length N" branch.

## Verification

- **Lean**: no change. `ScaledFrames.lean`'s `(l/l̆) · R(t̆ → t)` is
  the same uniform-tangent scale we're applying.
- **C++**: 1-2 new RC properties on the F_l composition.
- **GDExtension**: `scons -j8` clean. New material `length_handles`
  for visual distinctness.
- **Manual**: in editor, drag a knot's length handle along the
  tangent; verify the surface stretches/compresses along that
  direction without affecting cross-section width or tilt.

## Blocks / blocked-by

- **Blocks**: nothing.
- **Blocked-by**: nothing.

## Estimated cost

- LOC: ~120 across deformer (~50) + gizmo (~50) + tests (~20).
- Effort: small.
- Risk: low. The drag-handle mechanism is fully proven by tilt +
  width.
