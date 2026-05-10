# 02 — Curve-segment tracing through faces

## Why (current pain)

Even after todo 01 lands, knots are *isolated* sample-promoted
vertices. The curve segments *between* knots aren't represented in
the mesh — the cut-mesh has discrete Dirichlet boundary points, not
the connected cracks DeGoes22 §4.1 assumes. Result: deformation
"leaks" across the curve at runtime instead of pulling either side
independently. #2 high-priority gap in
[`docs/FEATURES.md`](../docs/FEATURES.md#high-priority).

## Gall-minimum slice

- **In scope**: a tracer that, given two adjacent projected knots and
  the cut-mesh between them, walks the geodesic from knot to knot and
  inserts cracks at each crossed edge / face via
  `cut_algorithm::insert_crack_at_endpoint`.
- **Deferred**: multi-curve intersection cracking (when two curves
  cross, the intersection geometry is non-trivial); robust geodesic
  via heat method (todo 10) — until then, fall back to Euclidean
  straight segments through the surface and accept slight
  inaccuracy on highly curved meshes.
- **Why this slice**: completes the §4.1 cut-mesh story for *single*
  curves on the surface. Multi-curve intersections happen at
  intersection knots which are already a separate code path.

## Files to touch

- `lean/Curvenet/CurveTracer.lean` (new) — spec the per-segment
  tracer that takes (start_kind, end_kind, mesh) and emits a sequence
  of cracks. Native_decide proofs on a 2-triangle strip + a 4-vertex
  quad.
- `src/curvenet/curve_tracer.h` (new) — C++ mirror.
- `src/curvenet_deformer_3d.cpp` — bind step calls the tracer after
  `promote_samples` for each curve segment.
- `tests/test_curve_tracer.cpp` (new) — RC props: trace through one
  edge, trace through a face, manifold preserved after cracks.

## Approach

- Tracer input: ordered list of `ProjectedKnot` for one curve, the
  current cut-mesh.
- For each adjacent pair (k_i, k_{i+1}):
  - If both endpoints share a triangle: insert one crack along the
    Euclidean segment within that triangle.
  - Else: walk halfedges, find the next triangle the segment enters,
    insert a crack at the entry edge, recurse on the remainder.
  - Use `cut_algorithm::insert_crack_at_endpoint` (already specced)
    for the actual halfedge surgery; the tracer is just the
    walker.
- After all cracks are inserted, the cut-mesh's manifold check
  (`cut_mesh::partition_of_unity`, already proved) should still
  pass — modulo the intentional crack at the curve, which is the
  whole point.
- The deformer's `Lh_csr` rebuild after tracing automatically picks
  up the new halfedge structure since assembly walks face loops.

## Verification

- **Lean**: `lake build` clean. `Curvenet.CurveTracer` adds ~8
  proofs (2-tri strip trace, quad trace, manifold preserved).
- **C++**: `tests/test_curve_tracer.cpp` adds ~5 RC properties.
- **GDExtension**: `scons -j8` clean.
- **Manual**: in editor, draw a curve across a coarse mesh, drag one
  end's handle, verify the *other* side of the curve doesn't deform
  in lockstep — the cut decouples them.

## Blocks / blocked-by

- **Blocks**: 05 (side toggle has no effect without intersection
  cracks, and a curve crack is the simplest intersection-of-one).
- **Blocked-by**: 01.

## Estimated cost

- LOC: ~400 across Lean (~120) + C++ (~150) + deformer (~50) + tests
  (~80).
- Effort: medium-large.
- Risk: medium. The tracer's halfedge walking has many edge cases
  (degenerate intersections, knots on shared edges, curve segments
  that exit + re-enter a triangle). Plan for diagnostic dumps and
  build up examples gradually.
