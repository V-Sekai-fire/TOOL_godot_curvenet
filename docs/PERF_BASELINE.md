# DeGoes22 deformer — CPU baseline

Wall-clock numbers for `src/curvenet_deformer_3d.cpp::apply_deformation`
on CPU. Run via:

```sh
make -C tests bench    # builds tests/bench_deform.cpp at -O3 + runs it
```

Tracks three workloads: synthetic plane (smooth, well-conditioned),
synthetic plane via dense LU (regression check for the historical
path), and a real character mesh (Mire Quest body, 5485 verts).

## Real character mesh — current ceiling

`Mire body` is loaded from `tests/mire_body_data.h` (baked from
`demo/character/MireQuest.blend` via Blender MCP). Source:
[V-Sekai-fire/mesh-mille-mire-feuille](https://github.com/V-Sekai-fire/mesh-mille-mire-feuille).
Curvenet is a single closed-loop with N hand-picked sample points
projected to the closest body vertices.

| label                       | nv   | nh    | nc | bind ms | frame ms | frames/s |
|-----------------------------|------|-------|----|---------|----------|----------|
| Mire body, 4-sample loop    | 5485 | 30856 | 4  |  293    | **5779** | **0.2**  |
| Mire body, 8-sample loop    | 5485 | 30856 | 8  |  290    | **5782** | **0.2**  |

**~5.8 seconds per frame.** This is the actual deployment-relevant
number; everything below extrapolates from synthetic Laplacians and
overstates real performance.

The synthetic 70² plane at similar nh = 28842 reports 69 ms/frame.
The 84× gap is **not** in the matrix size — it's in the conditioning
of the cot-Laplacian on irregular character topology (skinny triangles,
obtuse angles). CG iter count grows with √κ, and a 10⁴ jump in κ
buys a ~100× jump in iter count, which is consistent with the
measured frame time. See "Diagnosis path" below.

## Synthetic plane — sparse + warm-start (the runtime path)

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

These numbers describe a smooth, well-conditioned 2D Laplacian on a
unit-square plane with corner samples. They are **not** representative
of real character meshes — see the row above.

## Synthetic plane — dense LU (kept for regression checking)

| N×N  | `nv`  | `nh`  | bind (ms) | frame avg (ms) | frames/s |
|------|-------|-------|-----------|----------------|----------|
|  10² |   100 |   522 |        41 |           4.35 |    230   |
|  15² |   225 |  1232 |       466 |          25.54 |     39   |
|  20² |   400 |  2242 |      2797 |          87.44 |     11   |
|  25² |   625 |  3552 |     11962 |         215.98 |      5   |
|  30² |   900 |  5162 |     38579 |         458.06 |      2   |

Both columns scale cubically. Dense path stops at 900 verts on
memory; sparse takes over from there.

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

## Diagnosis path for the real-mesh result

Two questions to answer before reaching for a fix:

1. **Is CG hitting the iteration cap?** Current `cg_max_iter = nv * 2`,
   so 10970 iters per RHS column max. With 12 RHS per frame, hitting
   the cap on every column gives ~131k iters per frame, which is in
   the ballpark of the measured 5.8 s. If true: preconditioner problem.
2. **Are `LhsM_csr` diagonal entries reasonable?** Cot weights on
   obtuse triangles can be negative. Jacobi divides by `diag(A)`, so
   negative or near-zero diagonals make the preconditioner produce
   garbage and CG progress badly.

Cheapest diagnostic: thread an iteration count + final residual through
`sparse::cg_with_guess` and report mean iters / convergence ratio per
frame in the bench. Then decide between:

* Stronger preconditioner (incomplete Cholesky on `LhsM_csr`)
* Robust polygon Laplacian (Sharp-Crane 2020 / Bunge et al 2020,
  see `todos/07`) so the cot weights stay well-conditioned even on
  hairy meshes
* Mesh repair upstream (fTetWild manifold prepass per `todos/06`)
  so the deformer never sees the worst topology
* All of the above in some order

## Next steps

Pure CPU work, ranked by expected impact on the real-mesh row:

1. Diagnose CG iter count + diagonal sanity on the Mire body.
2. Robust polygon Laplacian per `todos/07_robust_laplacian_upgrade.md`.
3. Incomplete Cholesky preconditioner.
4. fTetWild manifold prepass per `todos/06_ftetwild_runtime_integration.md`.

GPU work is **not** on this list. The CPU path needs to work on real
character meshes before GPU optimisation has a load-bearing target.
GPU progress is tracked separately in
[docs/PERF_GPU.md](PERF_GPU.md).

## When to re-run

Re-run `make -C tests bench` after any of:

* changes to `cut_mesh_laplacian::assemble_lh_csr` /
  `assemble_vt_lh_v_csr`
* changes to `sparse::cg` (preconditioner, tolerance, warm-start)
* changes to the per-frame solve in `curvenet_deformer_3d.cpp`
* swapping the polygon Laplacian implementation
