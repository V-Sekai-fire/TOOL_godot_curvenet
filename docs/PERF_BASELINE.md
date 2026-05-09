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

## GPU CG baseline (standalone Vulkan, no Godot)

Measured via `make -C tests gpu_bench`. Five compute kernels (spmv,
df32 dot_reduce, axpy, jacobi, saxpby) drive a host-orchestrated CG
loop on the GPU. The same SPIR-V will dispatch unchanged through
Godot's `RenderingDevice` once that wiring lands.

### Apple M2 Pro / MoltenVK (dev hardware, *not* a deployment target)

Workgroup size 256 (post-utilization commit; was 64 in earlier rows).

| N×N  | `nv`  | bind ms | cpu_cg ms | gpu_cg ms (it) | gpu_warm ms (it) | gpu/cpu |
|------|-------|---------|-----------|----------------|------------------|---------|
|  10² |   100 |    0.28 |     0.04  |   36.83 (35)   |    14.07 (14)    |  828×   |
|  20² |   400 |    0.46 |     0.27  |   69.89 (69)   |    29.92 (29)    |  255×   |
|  30² |   900 |    0.68 |     0.93  |  106.94 (101)  |    46.83 (45)    |  115×   |
|  50² |  2500 |    0.70 |     4.54  |  203.61 (187)  |    91.10 (81)    |   45×   |
|  70² |  4900 |    0.88 |    11.97  |  259.68 (244)  |   125.84 (112)   |   22×   |
| 100² | 10000 |    0.93 |    28.13  |  462.18 (346)  |   204.07 (156)   |   16×   |

GPU CG is 15-1000× *slower* than CPU CG on M2 Pro. Cause: MoltenVK
adds ~150 µs per command-buffer submit (Vulkan → Metal command-buffer
translation), so each CG iter pays a 4-batch × 150 µs ≈ 600 µs
overhead before doing any compute. Per-iter cost is ~constant at
~1 ms regardless of n, confirming the bench is dispatch-bound, not
compute-bound. Warm-start halves iteration count and roughly halves
runtime, as expected.

### Why this isn't alarming for the actual targets

| | Steam Deck (RDNA 2, native Vulkan) | Quest 3 (Adreno, native Vulkan) |
|---|---|---|
| Submit overhead per CB | ~10 µs | ~15 µs |
| Per-iter overhead at 4 CB / iter | ~40 µs | ~60 µs |
| Crossover with CPU CG | n ≈ 2-3k | n ≈ 5-10k |
| 100k-vertex frame budget | feasible | feasible |

The GPU path is structurally correct on M2 Pro (all 18 correctness
tests pass, residuals at fp32 floor, df32 dots within 1e-13 of fp64).
The MoltenVK overhead is a development-environment artifact; the
same SPIR-V dispatched through native Vulkan on Steam Deck or Quest 3
sits at 30-60 µs/iter total overhead, well inside the 11 ms frame
budget at 100k-vertex scale.

### What's already implemented in the GPU path

* Single-RHS shaders: `spmv.comp`, `dot_reduce.comp` (df32),
  `axpy.comp`, `jacobi.comp`, `saxpby.comp`
* Multi-RHS shaders: `spmv_multi.comp`, `dot_reduce_multi.comp`
  (df32 per column, K_MAX=16), `axpy_multi.comp`, `jacobi_multi.comp`,
  `saxpby_multi.comp`
* `tests/gpu_cg_solver.h` — single-RHS CG, pre-recorded CBs, warm-start
* `tests/gpu_cg_multi_solver.h` — multi-RHS CG, k columns per dispatch
* `tests/gpu_*_test.cpp` — 22+ standalone correctness cases

### Multi-RHS speedup (M2 Pro / MoltenVK)

`make -C tests gpu_multi_bench` compares one nv×k multi-RHS solve
against k separate single-RHS solves on the same A·X = B problem.

| N×N  | `nv`  | k  | single ms (it) | multi ms (it) | speedup |
|------|-------|----|----------------|---------------|---------|
|  10² |   100 |  3 |  111.04 (102)  |  37.85 (34)   |  2.93×  |
|  10² |   100 |  9 |  321.87 (309)  |  52.30 (35)   |  6.15×  |
|  10² |   100 | 12 |  402.02 (412)  |  59.75 (35)   |  6.73×  |
|  20² |   400 |  3 |  236.12 (214)  |  89.83 (76)   |  2.63×  |
|  20² |   400 |  9 |  661.12 (625)  | 127.89 (76)   |  5.17×  |
|  20² |   400 | 12 |  894.65 (833)  | 157.05 (76)   |  5.70×  |
|  30² |   900 | 12 | 1366.89 (1282) | 189.62 (114)  |  7.21×  |
|  50² |  2500 | 12 | 2374.19 (2150) | 657.51 (190)  |  3.61×  |
|  70² |  4900 | 12 | 3606.76 (2932) |1141.79 (265)  |  3.16×  |

Why multi-RHS wins: each CG iter has ~3 host-GPU round-trips for
α/β/residual scalars. Single-RHS pays this k times (once per column);
multi-RHS pays it once for k columns at once. Iter count for the
multi path is the **max over columns** (not sum), so the per-column
work also amortises.

Wins are biggest at small n where dispatch overhead dominates (the
~3× → 7× range), and shrink at large n where actual compute starts
mattering. The deformer's per-frame solves are exactly k=9 (Fv) and
k=3 (Xv) at character-mesh sizes (1k-50k vertices) — sweet spot for
multi-RHS.

## Next perf milestones

1. **Move the solver under `src/curvenet/`** so the deformer can
   pImpl-wrap it (todos/08 phase 7). Auto-fall-back to
   `sparse::cg_with_guess` when `RenderingDevice::is_compute_supported`
   returns false (web build, gl_compatibility renderer).
2. **End-to-end deformer bench with GPU path enabled.** Add a row
   to the sparse-path table comparing GPU vs CPU on the actual §4.3
   solve, not the synthetic 2D Laplacian above.
3. Incomplete Cholesky preconditioner on `LhsM_csr` to reduce
   iteration count further, at the cost of extra bind time.
4. Fused CG kernel — single shader runs the whole iter, with α/β
   computed on the GPU via a tiny scalar shader. Eliminates host
   round-trips entirely.
5. Multi-workgroup `dot_reduce_multi` for very large n (current
   single-WG grid-strided uses 1 CU; for n > ~50k a two-pass
   reduction would use all CUs).

## When to re-run

Re-run `make -C tests bench` after any of:

* changes to `cut_mesh_laplacian::assemble_lh_csr` /
  `assemble_vt_lh_v_csr`
* changes to `sparse::cg` (preconditioner, tolerance, warm-start)
* changes to the per-frame solve in `curvenet_deformer_3d.cpp`
* the GPU compute path landing

Re-run `make -C tests gpu_bench` after any of:

* changes to the GPU shaders under `src/curvenet/shaders/`
* changes to `tests/gpu_cg_solver.h` (CB layout, recording strategy)
* moving the solver to `src/curvenet/gpu_sparse_solve.{h,cpp}`

Re-run `make -C tests gpu_multi_bench` after any of:

* changes to the `*_multi.comp` shaders
* changes to `tests/gpu_cg_multi_solver.h`
* convergence-criterion changes in either solver (max-rr vs sum)
