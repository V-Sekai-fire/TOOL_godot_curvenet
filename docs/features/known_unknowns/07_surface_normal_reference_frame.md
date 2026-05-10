# 07 — Surface-normal reference frame for tilt

## Why (current pain)

The per-knot frame's reference normal is derived from
**world-Y-up** (or world-Z when the tangent is too close to Y) via
Gram-Schmidt — both in the gizmo visualization
(`compute_knot_frame` in `curvenet_gizmo_plugin.cpp`) and in the
deformer's runtime tilt path. On a curved surface this means
`tilt = 0` corresponds to "rotate around tangent until the binormal
aligns with world Y", *not* "the rest frame matches the surface".

Result: an artist setting tilt = 0 expects the frame to lie flat on
the surface; instead it's tied to global axes. Tilting a knot on a
curved torso doesn't behave like tilting a bone — it behaves like
tilting against the world. Medium-priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#medium-priority).

## Gall-minimum slice

- **In scope**: bake mesh vertex normals into `RestCache` at the bind
  step, then derive the per-knot reference (b0, n0) at the projected
  vertex from the **surface normal at the projection** instead of
  world-up. Fall back to world-up only when the projection has no
  associated mesh vertex (shouldn't happen in normal flow).
- **Deferred**: artist-controllable rest normal override (rare in
  practice — surfaces don't usually need a non-default rest frame).
- **Why this slice**: tiny pure correctness fix. Doesn't change the
  authoring API; the artist sees better tilt semantics for free.

## Files to touch

- `src/curvenet_deformer_3d.h` — `RestCache.rest_vertex_normals :
  std::vector<Vec3>` (one per mesh vertex).
- `src/curvenet_deformer_3d.cpp` — bind step accumulates per-vertex
  normals from incident face normals (same code already present in
  the `emit_mesh` lambda — lift to a pure helper). At runtime, look
  up the projected vertex's normal as the reference for the rest
  (b0, n0). Apply Δtilt around the *posed* tangent (using a
  similarly-derived posed normal — likely just the rest normal if
  the surface doesn't deform much, or a transported version under F_h
  for full fidelity).
- `src/curvenet_gizmo_plugin.cpp` — `compute_knot_frame` consults
  the deformer's rest normals if a deformer pointer is available;
  fall back to world-up otherwise.

## Approach

- Vertex-normal accumulation is already implemented inline in the
  deformer's mesh emit step (face normal accumulation, normalize at
  the end). Refactor to a `compute_vertex_normals(positions,
  tri_indices)` helper used by both bind and emit.
- At bind, store these in `rest_cache.rest_vertex_normals` aligned
  with `rest_cache.positions` indexing.
- For each handle, the rest reference normal is
  `rest_vertex_normals[col_input_handle's projected vertex]`. Build
  rest (b0, n0) as `b0 = (n_ref - t·(n_ref·t)).normalized()`,
  `n0 = t × b0`.
- For posed-side, transport the rest normal under the bind-time
  segment rotation: `n_ref_posed = R(t_rest → t_posed) · n_ref_rest`.
  This keeps the frame surface-aligned even as the curve bends.
- Gizmo's `compute_knot_frame` either takes the rest normal from the
  deformer's cache (when called during _redraw with a deformer
  pointer) or falls back to world-up Gram-Schmidt for the bare-curve
  preview case.

## Verification

- **Lean**: no change.
- **C++**: 2 new RC properties — vertex-normal accumulation matches
  hand-computed result on a 2-tri strip; tilt = 0 produces a frame
  whose binormal lies in the surface tangent plane.
- **GDExtension**: `scons -j8` clean.
- **Manual**: in editor, on a curved mesh (sphere or torso), set a
  knot's tilt to 0 and verify the green/blue axes lie tangent to the
  surface (not aligned with world Y).

## Blocks / blocked-by

- **Blocks**: nothing strictly, but improves quality of tilt-related
  features (06).
- **Blocked-by**: nothing.

## Estimated cost

- LOC: ~80 across deformer (~50) + gizmo (~30).
- Effort: small.
- Risk: low. Vertex-normal computation is canonical; the only
  subtlety is the posed-normal transport, which has a clean
  one-line derivation from the existing `smallest_rotation` helper.
