# 06 — Scaled frames + intersection frames

## What it is

DeGoes22 §3 geometric algebra. Per-segment / per-knot scaled-orthonormal
frame `B_i S_i` (basis × diagonal scale), the deformation gradient
`F = (B_i S_i)·(B̆_i S̆_i)⁻¹` between rest and posed frames, and at
intersection knots the per-side `(+, −)` frames that handle asymmetric
pull. The math layer underneath every per-handle transform.

## Status

shipping default — invoked by the runtime DDM matvec
([09](09_direct_delta_mush.md)) to compute per-handle `F_h` and by the
bind step's per-side frame derivation.

## Files

### Lean spec
- `lean/Curvenet/ScaledFrames.lean` — 19 native_decide proofs
  (Mat3 ops, deformation gradient, M·M⁻¹ round-trip, inverse
  involutivity)
- `lean/Curvenet/IntersectionFrames.lean` — 14 native_decide proofs
  (corner vectors, per-side normals, per-side widths, n_+ = -n_-,
  scaledFrame homogeneity)
- `lean/Curvenet/CurveInterp.lean` — 5 native_decide proofs (per-side
  interpolation along curves)
- `lean/Curvenet/SegmentGradient.lean` — 4 native_decide proofs
  (isolated / intersection dispatcher)

### C++ implementation
- `src/curvenet/scaled_frames.h` — `Mat3`, `mat3_mul`,
  `mat3_inv`, `deformation_gradient`, `isolated_segment_gradient`,
  `smallest_rotation`, `skew`
- `src/curvenet/intersection_frames.h` — `corner_vectors`,
  `corner_normals`, `per_side_normals`, `per_side_widths`,
  `scaled_frame`, `per_side_scaled_frames`
- `src/curvenet/curve_interp.h`
- `src/curvenet/segment_gradient.h`

### Tests
- `tests/test_scaled_frames.cpp`

## API surface

C++-internal. Programmer surface used by the deformer's runtime:
- `scaled_frames::isolated_segment_gradient(rest_p, rest_q,
  posed_p, posed_q) → Mat3` — the workhorse for tangent-direction
  rotation + length scale
- `scaled_frames::deformation_gradient(rest_BS, posed_BS) → Mat3` —
  full intersection-aware path
- `intersection_frames::scaled_frame(t, n, l, w) → Mat3` — builds
  `B S` from frame components

## How it works

- **Isolated segment** (no intersections): given two endpoints
  rest+posed, build `F = (l_posed/l_rest) · R(t̆ → t)` via
  Rodrigues' smallest-rotation formula. Captures uniform scale +
  tangent rotation, not tilt or asymmetric width.
- **Intersection knot**: per-side frames derived from outgoing
  segment tangents at the knot. The `+` side uses the corner-normal
  rotated CCW from the segment tangent; the `−` side reflects across
  the segment. T-junction degenerates handled by the
  `(c_{i+1} + c_{i-1})/|·|` fallback.
- **Full path** (used at runtime when widths/tilts engage):
  `F = (B S) · (B̆ S̆)⁻¹` — a one-liner via `deformation_gradient`.

## Cross-references

- Lean specs are the canonical reference for the algebra
- Composed with tilt rotation (Rodrigues around posed tangent) and
  perpendicular-to-tangent width scaling in
  [09 DDM](09_direct_delta_mush.md)'s runtime branch
- Per-side machinery enables future asymmetric pull
  ([known unknowns 05](../known_unknowns/05_side_toggle_ui.md))
