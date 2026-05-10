# 01 — Edge + face surface projection

## Why (current pain)

`surface_projection.h` only emits `ProjectionKind::vertex` — every
curvenet knot snaps to its closest mesh vertex. On coarse meshes this
produces visible "stair-step" curves: the artist drags a smooth
Bézier, but the deformer sees a polyline that jumps between widely
spaced vertices. Listed as the #1 high-priority gap.

## Gall-minimum slice

- **In scope**: closest-point-on-triangle helper that returns
  barycentric coordinates plus `ProjectionKind` (`vertex` /
  `edge_intersection` / `face_interior`); extend
  `promote_vertex_samples` → `promote_samples` that dispatches by kind
  and inserts new vertices for edge/face cases.
- **Deferred**: actual curve-segment tracing through edges/faces
  (knots become isolated samples, not connected cracks) — that's todo
  02.
- **Why this slice**: gets knots onto arbitrary surface points without
  yet committing to the more invasive cracking semantics. Cuts the
  stair-stepping today; cracks come later.

## Files to touch

- `lean/Curvenet/SurfaceProjection.lean` — extend with
  `closestPointOnTriangle` + a `projectToSurface` producer that emits
  the three kinds.
- `src/curvenet/surface_projection.h` — C++ mirror; new
  `closest_point_on_triangle` + `project_to_surface` +
  `promote_samples`.
- `src/curvenet_deformer_3d.cpp` — bind step calls
  `project_to_surface` instead of `project_to_vertices`, and
  `promote_samples` instead of `promote_vertex_samples`.
- `tests/test_surface_projection.cpp` — add edge-case props
  (knot at edge midpoint, knot at face centroid, barycentric sum-to-1
  invariant, kind dispatch correctness).

## Approach

- Implement closest-point-on-triangle as the canonical Ericson
  algorithm: project to plane, check barycentric, clamp to edges if
  outside. Returns `(point, barycentric, kind)`. Pure function, easy
  `native_decide` proof on small instances.
- `project_to_surface` iterates every triangle, picks the global
  minimum-distance result. O(nf) per knot — fine for the bind step's
  one-shot cost.
- For `edge_intersection`: call
  `cut_algorithm::subdivide_edge` (already specced with `native_decide`
  proofs in `lean/Curvenet/CutAlgorithm.lean` — 45 proofs).
  Append the new vertex's position to `rest_cache.positions`. Promote
  the new vertex via the existing sample-kind machinery.
- For `face_interior`: call `cut_algorithm::split_face`. Three new
  triangles replace the original; positions / `tri_indices` get
  appended. Promote the new vertex.
- Keep the deformer's hash logic conservative — any change to knot
  count or projection results invalidates the cache.

## Verification

- **Lean**: `lake build` clean, `Curvenet.SurfaceProjection` proof
  count goes from 10 to ~16.
- **C++**: `tests/test_surface_projection.cpp` adds ~6 RC properties
  covering edge midpoint / face centroid / mixed knots. Total RC
  count: 92 → ~98.
- **GDExtension**: `scons -j8` clean.
- **Manual**: in editor, place a knot off-vertex on a coarse mesh.
  Drag and verify the deformer follows the smooth curve, not the
  stair-stepped vertex sequence.

## Blocks / blocked-by

- **Blocks**: 02 (curve tracing needs the edge/face projection kinds
  to know where to cut).
- **Blocked-by**: none.

## Estimated cost

- LOC: ~300 across Lean (~80) + C++ headers (~120) + deformer (~50) +
  tests (~50).
- Effort: medium.
- Risk: low — closest-point-on-triangle is canonical, both
  `cut_algorithm` primitives are already specced and proved.
