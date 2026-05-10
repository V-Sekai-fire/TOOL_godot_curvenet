# 04 — Surface projection

## What it is

Projects each curvenet knot onto the source mesh by snapping to its
closest mesh vertex (vertex-only). Promotes the chosen vertex to a
sample in the cut-mesh, allocates per-sample column indices, and
records the mapping back to the input (curve_id, knot_idx) tuple.

## Status

shipping default — this is how curves get attached to the surface
today. **Vertex-only**; edge and face projection kinds are reserved
in the type but not yet emitted by the producer (see
[known unknowns 01](../known_unknowns/01_edge_face_projection.md)).

## Files

### Lean spec
- `lean/Curvenet/SurfaceProjection.lean` — 10 native_decide proofs

### C++ implementation
- `src/curvenet/surface_projection.h` — `ProjectionKind`,
  `ProjectedKnot`, `project_to_vertices`, `promote_vertex_samples`

### Tests
- `tests/test_surface_projection.cpp`

## API surface

Programmer-facing:
- `surface_projection::project_to_vertices(knots, mesh_positions)
  → vector<ProjectedKnot>` — closest-vertex per knot
- `surface_projection::promote_vertex_samples(cut_mesh, projections)
  → vector<int>` — per-knot column index, with same-vertex knots
  collapsing to one column (first wins)

## How it works

- Brute-force closest-vertex: O(n_knots × n_verts) per bind.
  Acceptable because bind is one-shot and even 81 k × small-curvenet is
  sub-second.
- Promotion writes `cut_mesh::CutVertexKind::sample_kind(col, 0,
  side)` at the chosen vertex slot. Subsequent halfedges with target
  at that vertex now contribute to C, not V.
- Multiple knots projecting to the same vertex deduplicate into a
  single column — preserves the assumption that "one sample = one
  Dirichlet boundary value".

## Cross-references

- Consumes [02 curvenet graph](02_curvenet_graph.md)
- Modifies [03 cut-mesh](03_cut_mesh.md)
- Stair-steps on coarse meshes — see
  [known unknowns 01](../known_unknowns/01_edge_face_projection.md) for edge+face
  generalization
