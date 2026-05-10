# 01 — Halfedge mesh

## What it is

Halfedge data structure with triangle-soup builder, manifold checks,
and twin / face-loop invariants. Foundational topology layer for the
cut-mesh and Laplacian assembly.

## Status

shipping default — used by every cut-mesh build at bind time.

## Files

### Lean spec
- `lean/Curvenet/Halfedge.lean` — 9 native_decide proofs
- `lean/Curvenet/HalfedgeBuilder.lean` — 14 native_decide proofs

### C++ implementation
- `src/curvenet/halfedge.h` — `Halfedge`, `HalfedgeMesh`,
  `is_manifold` checks
- `src/curvenet/halfedge_builder.h` — `from_triangles(vertex_count,
  tri_indices) → HalfedgeMesh`

### Tests
- `tests/test_halfedge.cpp`
- `tests/test_halfedge_builder.cpp`

## API surface

C++-internal. Programmer surface:
- `halfedge_builder::from_triangles(count, indices)` — builds the
  mesh from a flat triangle index list

## How it works

- Each interior triangle yields 3 halfedges (`3f`, `3f+1`, `3f+2` in
  CCW face-loop order).
- Twins are paired by directional `(source, target)` edge map.
- Boundary halfedges are added one per orphan interior; linked into
  outer-face loops using `next` pointers.
- Manifold invariants checked: indices in range, twin involutive,
  twins point to opposite source/target, face loops close.

## Cross-references

- Drives [03 cut-mesh](03_cut_mesh.md)
- Used by [05 cot-Laplacian](05_cot_laplacian.md) for face traversal
- Mirrors `src/curvenet/halfedge_builder.h` from triangle-soup input
  in `CurveNetDeformer3D::apply_deformation`'s bind block
