# 05 — Cot-Laplacian

## What it is

DeGoes22 §4.2's `L_h` matrix: the cut-mesh-wide Laplacian as the sum
of per-cut-face polygon Laplacians at halfedge-corner indices. The
`V^T L_h V` system is the shared LHS of both stages of the §6 solve.
Sharp & Crane 2020 intrinsic mollification gives finite cot weights
even on character meshes with degenerate triangles (the embedding
path produces ±∞ on real input).

## Status

shipping default — every bind builds the sparse `Lh_csr` and
`LhsM_csr = V^T L_h V` against the projected cut-mesh.

## Files

### Lean spec
- `lean/Curvenet/PolygonLaplacian.lean` — 7 native_decide proofs
- `lean/Curvenet/RobustLaplacian.lean` — 14 native_decide proofs
  (Sharp-Crane mollified)
- `lean/Curvenet/CutMeshLaplacian.lean` — 12 native_decide proofs
  (V^T L V symmetry, null-space, two-tri strip, cut-mesh case)

### C++ implementation
- `src/curvenet/polygon_laplacian.h`
- `src/curvenet/robust_laplacian.h`
- `src/curvenet/cut_mesh_laplacian.h` —
  `assemble_lh_csr_robust`, `assemble_vt_lh_v_csr_robust`,
  `default_mollify_delta`

### Tests
- `tests/test_polygon_laplacian.cpp`
- (RobustLaplacian invariants exercised through the deformer's
  end-to-end tests)

## API surface

C++-internal. The deformer calls
`assemble_vt_lh_v_csr_robust(cut_mesh, positions, delta)` once per
bind to populate `rest_cache.LhsM_csr`.

## How it works

- For each interior cut-face, fan-triangulate via halfedge `next`
  walking. For each fan triangle, derive edge lengths, mollify with
  `delta = 1e-5 × mean_edge_length`, build the 3×3 cot-Laplacian
  from those mollified lengths, scatter into the global `nh × nh`
  L_h.
- V is implicit — a sparse 0/1 matrix mapping mesh-vertex unknowns
  to halfedges. We never form V; `apply_vt` and `assemble_v` walk
  halfedges directly.
- `V^T L_h V` is the **nv × nv** system both §6 stages solve. SPD
  apart from the rank deficit at sample-promoted vertex rows
  (degenerate; the dense / CG path returns 0 for those slots and the
  caller overlays the constraint value).
- Symmetric and has the constant vector in its null space — both
  proved at small instances in `CutMeshLaplacian.lean`.

## Cross-references

- Built on [03 cut-mesh](03_cut_mesh.md)'s face-loop traversal
- Solved by [07 solver kernels](07_solver_kernels.md)
- Drives [08 §6 solve](08_degoes22_solve.md) and the bind harvest
  for [09 DDM](09_direct_delta_mush.md)
- Sharp & Crane 2020 — "A Laplacian for Nonmanifold Triangle Meshes"
