# 02 — Curvenet graph

## What it is

Builds the deduplicated, classified knot graph from the artist's
authored profile curves. ε-merges shared endpoints across curves so
nearby positions become a single knot, then classifies each merged
knot per DeGoes22 §3 — anchor (deg 1) / regular (deg 2) /
intersection (deg ≥ 3).

## Status

shipping default — runs on every bind to turn `Curve3D[]` into the
graph the surface projector consumes.

## Files

### Lean spec
- `lean/Curvenet/CurvenetBuilder.lean` — 10 native_decide proofs

### C++ implementation
- `src/curvenet/curvenet_builder.h` — `KnotKind`, `KnotRef`,
  `CurvenetGraph`, `build()`, `classify()`, `outgoing_tangents()`

### Tests
- `tests/test_curvenet_builder.cpp`

## API surface

Programmer-facing:
- `curvenet_builder::build(curves, eps) → CurvenetGraph`
- `curvenet_builder::classify(graph) → vector<KnotKind>`
- `curvenet_builder::outgoing_tangents(graph) → vector<vector<Vec3>>`

Artist-facing: indirectly via `CurveNetDeformer3D::profile_curves` —
the deformer passes them through `build()` automatically.

## How it works

- Walk every curve's points; for each point, search existing
  knot_positions for one within ε. Reuse if found, else allocate.
- Track incidence: each merged knot records which (curve_id,
  knot_idx) entries collapsed onto it.
- `classify` counts incident segment endpoints per merged knot:
  anchor (1), regular (2), intersection (≥ 3). Closed loops count
  the self-touching sample.
- ε defaults to 1e-6 in `apply_deformation`'s call site —
  conservative; meant to merge intentional shared endpoints, not
  perturb-distinct-near knots.

## Cross-references

- Feeds [04 surface projection](04_surface_projection.md)
- Visualization in [11 editor visualization](11_editor_visualization.md)
  (knot kind markers)
- Drives the per-side fallback path through
  [06 intersection frames](06_scaled_frames.md)
