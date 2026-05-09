# DeGoes22 deformer — performance baseline

Wall-clock numbers for both solver paths under
`src/curvenet_deformer_3d.cpp::apply_deformation`. Run via:

```sh
make -C tests bench    # builds tests/bench_deform.cpp at -O3 + runs it
```

The benchmark builds an N×N triangulated unit-plane grid, promotes the
four corner vertices to curvenet samples, and times 30 frames of the
§4.3 two-stage solve with identity `Fc` and a slowly-drifting `Xc` per
frame. Two paths are timed: a dense LU factor-once path, and a sparse
CSR `Lₕ` path with Jacobi-preconditioned conjugate gradient and
warm-started initial guess (each frame seeds CG with the previous
frame's iterate). The runtime uses the sparse path; the dense path is
kept for regression checking.

## Dense path

| N×N  | `nv`  | `nh`  | bind (ms) | frame avg (ms) | frames/s |
|------|-------|-------|-----------|----------------|----------|
|  10² |   100 |   522 |        41 |           4.35 |    230   |
|  15² |   225 |  1232 |       466 |          25.54 |     39   |
|  20² |   400 |  2242 |      2797 |          87.44 |     11   |
|  25² |   625 |  3552 |     11962 |         215.98 |      5   |
|  30² |   900 |  5162 |     38579 |         458.06 |      2   |

Both columns scale cubically with `nv`. The bind cost is dominated by
`mat_mul` on the dense `nh × nh` Lₕ, which is why the dense table caps
at 900 verts: larger grids exhaust memory before reporting a number.

## Sparse path (current runtime)

| N×N  | `nv`   | `nh`   | bind (ms) | frame avg (ms) | frames/s |
|------|--------|--------|-----------|----------------|----------|
|  10² |    100 |    522 |       0.3 |           0.07 |  13881   |
|  15² |    225 |   1232 |       0.9 |           0.24 |   4096   |
|  20² |    400 |   2242 |       2.3 |           0.64 |   1568   |
|  25² |    625 |   3552 |       4.9 |           1.43 |    700   |
|  30² |    900 |   5162 |       9.7 |           2.77 |    361   |
|  40² |   1600 |   9282 |      29.0 |           8.09 |    124   |
|  50² |   2500 |  14602 |      68.1 |          18.98 |     53   |
|  70² |   4900 |  28842 |     256.3 |          69.05 |     15   |

At 900 verts the sparse + warm-start path is ~160× faster per frame
than dense. Bind memory is `O(nnz(Lₕ))` ≈ `O(nh)`, so the table extends
to 4900 verts without exhausting memory. Warm-starting from the
previous frame's iterate adds another 1.5–1.8× over cold-start CG on
smooth interactive drags, because the per-frame change in `Fv` and
`Xv` is small. CG iteration count grows with `√κ` of the LHS; the
Jacobi preconditioner is enough at these sizes but will need upgrading
for the 50K+ character meshes the runtime targets (see
`todos/08_gpu_compute_solver.md`).

## What this means for the target platforms

* Demo (10² grid, 100 verts): 8 kHz on the math, so the per-frame
  budget is dominated by the SurfaceTool emit on the Godot side.
* Small character (5K verts, ~70²): ~10 frames/s on the sparse CPU
  path. Interactive but below 90 Hz.
* Steam Deck / Quest 3 character (≥5K, often 20–50K): the CPU sparse
  path needs warm-start CG and a better preconditioner to keep up.
  100K real-time on these handhelds requires the GPU compute path
  in `todos/08_gpu_compute_solver.md`.

## Benchmark scope

Included in the per-frame timing:

* assembling `C·F_c` and `C·X_c` per-halfedge constraint matrices
* `Lₕ · (C·F_c)` and the `Vᵀ·…` aggregation (Eq. 6a RHS)
* solve against the cached LHS (LU back-sub for dense, CG for sparse,
  9 columns then 3)
* the bridge: `compute_fh`, `average_over_faces`, `compute_yh`

Excluded from per-frame:

* the SurfaceTool / ArrayMesh emit (Godot-side, hard to bench in
  isolation)
* the curvenet_builder ε-merge and surface_projection (run only at
  bind time in the runtime)

The `bind` column includes everything from `from_triangles` through
the LHS assembly (and LU factorisation for the dense row). Real
`apply_deformation` also runs the curvenet build and sample promotion
at bind, which adds a small constant.

## Next perf milestones

1. Incomplete Cholesky preconditioner on `LhsM_csr` to reduce
   iteration count further, at the cost of extra bind time.
2. GPU compute path per `todos/08_gpu_compute_solver.md` for the
   100K-vertex Steam Deck and Quest 3 target.

## When to re-run

Re-run `make -C tests bench` after any of:

* changes to `cut_mesh_laplacian::assemble_lh_csr` /
  `assemble_vt_lh_v_csr`
* changes to `sparse::cg` (preconditioner, tolerance, warm-start)
* changes to the per-frame solve in `curvenet_deformer_3d.cpp`
* the GPU compute path landing
