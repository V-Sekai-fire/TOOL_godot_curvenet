# DeGoes22 deformer — performance baseline

Wall-clock numbers for the **dense LU factor-once** path in
`src/curvenet/dense_linalg.h` driven by the per-frame solve in
`src/curvenet_deformer_3d.cpp::apply_deformation`. Run via:

```sh
make -C tests bench    # builds tests/bench_deform.cpp at -O3 + runs it
```

The benchmark builds an N×N triangulated unit-plane grid, promotes the
four corner vertices to curvenet samples, factorises `VᵀLₕV` once, and
times 30 frames of the §4.3 two-stage solve with identity `Fc` and a
slowly-drifting `Xc` per frame.

## Baseline numbers (Apple Silicon M-series, dense path)

| N×N  | `nv`  | `nh`  | bind (ms) | frame avg (ms) | frames/s |
|------|-------|-------|-----------|----------------|----------|
|  10² |   100 |   522 |        57 |           4.28 |    234   |
|  15² |   225 |  1232 |       449 |          24.50 |     41   |
|  20² |   400 |  2242 |      2740 |          83.73 |     12   |
|  25² |   625 |  3552 |     11345 |         213.80 |      5   |
|  30² |   900 |  5162 |     37176 |         452.51 |      2   |

Both the bind and per-frame columns scale cubically with `nv`: each
doubling of vertex count multiplies the cost by roughly 8–9×. The bind
cost is dominated by `mat_mul` on the dense Lₕ (`nh × nh`), which goes
to ~30 GB at N×N = 30² and is the reason the bench caps at 900 verts —
larger grids exhaust memory before reporting a number.

## What this means for the target platforms

* **Demo (10² grid, 100 verts):** 234 FPS interactive — fits Steam Deck
  and Quest 3 with margin. The Coons-stub-bug-or-not the demo always
  ran fine here; this confirms the new pipeline does too.
* **Small character (~5K verts):** dense bind would need ~10⁶ ms ≈
  17 minutes, which is unworkable. Dense path is structurally dead at
  this scale.
* **Steam Deck / Quest 3 character (≥5K, often 20–50K):** dense path
  cannot even bind. Sparse `Lₕ` assembly (todos/08 phase 1) and
  iterative CG are mandatory, and 100K real-time still requires GPU
  compute (todos/08 phase 7+).

## Benchmark scope (what's included / excluded)

Included in the per-frame timing:
* assembling `C·F_c` and `C·X_c` per-halfedge constraint matrices
* `Lₕ · (C·F_c)` and `Vᵀ · Lₕ · (C·F_c)` dense matmuls (Eq. 6a RHS)
* `solve_multi_with_lu` against the cached factor (9 columns then 3)
* the bridge: `compute_fh`, `average_over_faces`, `compute_yh`

Excluded from per-frame:
* the SurfaceTool / ArrayMesh emit (Godot-side; hard to bench in
  isolation)
* the curvenet_builder ε-merge and surface_projection (run only at
  bind time in the runtime)

The `bind` column includes everything from `from_triangles` through
`factorize_lu`. Real apply_deformation also runs the curvenet build +
sample promotion at bind, which adds a small constant.

## Comparison points the next perf commits should hit

After the **sparse `Lₕ` assembly** lands (CSR `Lₕ` instead of dense
`nh × nh`), the bind memory drops from `O(nh²)` to `O(nnz(Lₕ)) ≈ O(nh)`
and the bind time becomes linear. Expected:

| N×N  | `nv`  | bind (ms) target | frame (ms) target |
|------|-------|------------------|-------------------|
|  30² |   900 |             ~50  |              ~50  |
|  70² |  4900 |            ~300  |             ~300  |
| 100² | 10000 |            ~600  |             ~600  |

After **GPU compute CG** (todos/08), 90 FPS at 100K vertices on
Steam Deck and Quest 3. Frame target: ≤ 5 ms.

## When to re-run

Re-run `make -C tests bench` after any of:
* swapping `dense::solve_multi_with_lu` → `sparse::cg`
* turning `cut_mesh_laplacian::assemble_lh` sparse
* adding warm-start CG state to RestCache
* the GPU compute path landing
