# DeGoes22 deformer — GPU compute (synthetic, work-in-progress)

GPU CG kernel set built under `tests/`, dispatched standalone via
Vulkan (no Godot integration yet). All numbers below are on
**synthetic 2D Laplacians**, not the real character mesh from
[PERF_BASELINE.md](PERF_BASELINE.md). They overstate real-world
performance: see the 5.8-second-per-frame finding on the actual
character mesh.

The CPU path doesn't yet meet target on real characters, so this
work is on hold pending the CPU diagnosis (preconditioner, polygon
Laplacian, mesh repair). When the CPU baseline is workable, the
GPU shaders are ready to be wired through Godot's `RenderingDevice`
per `todos/08_gpu_compute_solver.md`.

## What's implemented

* Single-RHS shaders: `spmv.comp`, `dot_reduce.comp` (df32),
  `axpy.comp`, `jacobi.comp`, `saxpby.comp`
* Multi-RHS shaders: `spmv_multi.comp`, `dot_reduce_multi.comp`
  (df32 per column, K_MAX=16), `axpy_multi.comp`, `jacobi_multi.comp`,
  `saxpby_multi.comp`
* `tests/gpu_cg_solver.h` — single-RHS CG, pre-recorded CBs, warm-start
* `tests/gpu_cg_multi_solver.h` — multi-RHS CG, k columns per dispatch
* `tests/gpu_*_test.cpp` — 22+ standalone correctness cases

`make -C tests gpu` runs all correctness tests; the GPU CG result
matches CPU CG within fp32 tolerance on every SPD instance tested.

## Single-RHS CG (M2 Pro / MoltenVK)

`make -C tests gpu_bench`. Workgroup size 256.

| N×N  | `nv`  | bind ms | cpu_cg ms | gpu_cg ms (it) | gpu_warm ms (it) | gpu/cpu |
|------|-------|---------|-----------|----------------|------------------|---------|
|  10² |   100 |    0.28 |     0.04  |   36.83 (35)   |    14.07 (14)    |  828×   |
|  20² |   400 |    0.46 |     0.27  |   69.89 (69)   |    29.92 (29)    |  255×   |
|  30² |   900 |    0.68 |     0.93  |  106.94 (101)  |    46.83 (45)    |  115×   |
|  50² |  2500 |    0.70 |     4.54  |  203.61 (187)  |    91.10 (81)    |   45×   |
|  70² |  4900 |    0.88 |    11.97  |  259.68 (244)  |   125.84 (112)   |   22×   |
| 100² | 10000 |    0.93 |    28.13  |  462.18 (346)  |   204.07 (156)   |   16×   |

GPU CG is 15-1000× *slower* than CPU CG on M2 Pro because MoltenVK
adds ~150 µs per command-buffer submit. Per-iter cost is ~constant
at ~1 ms regardless of n — dispatch-bound, not compute-bound.

## Multi-RHS CG (M2 Pro / MoltenVK)

`make -C tests gpu_multi_bench`. Compares one nv×k multi-RHS solve
vs k separate single-RHS solves.

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

Multi-RHS amortises the per-iter dispatch overhead k×, with iter
count being the **max over columns** (not sum). The deformer's
per-frame solves are k=9 (Fv) and k=3 (Xv).

## Why M2 Pro / MoltenVK numbers don't generalise

| | Steam Deck (RDNA 2, native Vulkan) | Quest 3 (Adreno, native Vulkan) |
|---|---|---|
| Submit overhead per CB | ~10 µs | ~15 µs |
| Per-iter overhead at 4 CB / iter | ~40 µs | ~60 µs |
| Crossover with CPU CG | n ≈ 2-3k | n ≈ 5-10k |

The MoltenVK overhead is a development-environment artifact. The
same SPIR-V dispatched through native Vulkan on Steam Deck or Quest 3
sits at 30-60 µs/iter total overhead. **Untested on real hardware
until either the deformer is wired into RenderingDevice or we ship
to a Steam Deck for measurement.**

## Precision strategy

* `spmv` / `axpy` / `jacobi` / `saxpby`: fp32, ~few ulp error
* `dot_reduce`: df32 (two-fp32-lane carries, ~48-bit effective
  mantissa) — required because fp32 dot over 100k elements
  stagnates at ~1e-3 relative error
* Host α / β math: fp64

df32 reaches fp64-quality scalars at ~3× the per-add cost of fp32,
no extra memory bandwidth. Da Graça-Defour 2006 has the operator
definitions; see `references.bib`.

## Next steps (gated on CPU baseline)

1. CPU path workable on real character meshes (preconditioner /
   robust Laplacian / mesh repair — see PERF_BASELINE.md)
2. Move solver under `src/curvenet/gpu_sparse_solve.{h,cpp}` with
   pImpl, wire into `CurveNetDeformer3D::RestCache` with auto-
   fallback to `sparse::cg_with_guess`
3. End-to-end deformer bench with GPU enabled (real character mesh,
   not synthetic)
4. Fused CG kernel — single shader runs the whole iter with α/β
   computed on the GPU. Eliminates the host round-trips that
   currently dominate cost.
5. Multi-workgroup `dot_reduce_multi` for n > ~50k (current single-
   workgroup grid-stride uses 1 CU)

## When to re-run

`make -C tests gpu_bench` after changes to single-RHS shaders or
`tests/gpu_cg_solver.h`.

`make -C tests gpu_multi_bench` after changes to `*_multi.comp`
shaders, `tests/gpu_cg_multi_solver.h`, or convergence criteria.
