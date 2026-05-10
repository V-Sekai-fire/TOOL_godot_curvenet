# 03 — Cut-mesh + cut-algorithm

## What it is

Halfedge mesh extended with sample-promoted vertices (curvenet knots
that landed on the surface) and a set of surgical primitives that
modify the mesh in place — subdivide an edge, split a face at an
interior point, insert a crack at a halfedge endpoint. The basis
for DeGoes22 §4.1 cut-mesh construction.

## Status

shipping default for the sample-promotion path. The surgery
primitives (`subdivide_edge`, `split_face`, `insert_crack_at_endpoint`)
are specced and proved but not yet invoked by the bind step — see
[todo 01](../../todo/01_edge_face_projection.md) and
[todo 02](../../todo/02_curve_segment_tracing.md).

## Files

### Lean spec
- `lean/Curvenet/CutMesh.lean` — 12 native_decide proofs
- `lean/Curvenet/CutAlgorithm.lean` — **45** native_decide proofs
  (largest single proof set in the project)

### C++ implementation
- `src/curvenet/cut_mesh.h` — `CutVertexKind` (mesh_vertex /
  sample / edge_intersection), `CutMesh`, `v_column_of`,
  `c_column_of`, `partition_of_unity`
- `src/curvenet/cut_algorithm.h` — `subdivide_edge`, `split_face`,
  `insert_crack_at_endpoint`

### Tests
- `tests/test_cut_algorithm.cpp`

## API surface

Programmer-facing:
- `cut_mesh::CutMesh` aggregates a `HalfedgeMesh` plus per-vertex
  `CutVertexKind` and per-halfedge `segment_of_halfedge` annotation.
- `cut_algorithm::subdivide_edge(mesh, halfedge_id, t)` — inserts
  a new vertex on the edge, splits surrounding faces.
- `cut_algorithm::split_face(mesh, face_id, bary)` — adds a vertex
  inside a face, fans 3 new triangles from it.

## How it works

- A "sample-promoted" vertex carries the curvenet knot identity (its
  curve_id + knot_idx + side flag). At assembly time, halfedges whose
  target is a sample contribute to the C matrix (Dirichlet boundary);
  others contribute to V.
- The partition-of-unity invariant: every halfedge contributes to
  exactly one of V or C, never both, never neither.
  `cut_mesh::partition_of_unity` checks this and is exercised by
  every cut-mesh test.
- The 45 proofs in `CutAlgorithm.lean` cover correctness of each
  surgical primitive on small fixtures (single tri, two-tri strip,
  star, fan).

## Cross-references

- Built from [01 halfedge mesh](01_halfedge_mesh.md)
- Driven by [04 surface projection](04_surface_projection.md) (which
  decides which vertices to promote)
- Feeds [05 cot-Laplacian](05_cot_laplacian.md)'s face-loop
  traversal
- Surgery primitives unlock [todo 01](../../todo/01_edge_face_projection.md)
  and [todo 02](../../todo/02_curve_segment_tracing.md)
