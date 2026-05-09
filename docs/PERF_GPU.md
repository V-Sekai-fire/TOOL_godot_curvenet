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

## fp32 precision is not enough on the deformer's actual matrix

Validated 2026-05 by running `gpu_cg_solver` (the same kernel set
that passes synthetic 2D-Laplacian tests at residuals ~1e-7 to 1e-6)
against the deformer's actual `LhsM` matrix from real character
meshes. Results (`make -C tests bench_5k_gpu`,
`bench_70k_gpu`):

| Mesh         | nv    | CPU iters / resid | GPU iters / resid | GPU outcome |
|--------------|-------|-------------------|-------------------|-------------|
| Mire 5k      | 5485  | 1,401 / 4.3e-9    | 10,970 (cap) / **0.073**    | did not converge |
| Mire 81k     | 81613 | 20,668 / 1.2e-8   | 163,226 (cap) / **2.1e3**   | did not converge |

GPU CG hits its iter cap on both. Cause: even with df32 dots, the
per-vertex `x` is stored as fp32. Over ~1k+ iters the fp32 accumulation
errors compound and the residual can't drop below the fp32 precision
floor on a matrix this conditioned (κ ≈ 4×10⁸ at 81k). The synthetic
2D Laplacian tests passed because those matrices have κ ≈ 10⁴-10⁵
and need ~30-300 iters — comfortably inside fp32's 7-digit budget.

What this means: the current GPU CG infrastructure is not directly
applicable to the deformer's matrices at any scale. Three viable
next paths:

1. **Mixed-precision iterative refinement** (Buttari 2008 /
   Carson-Higham 2018, in references.bib). Outer fp64 loop on CPU
   that wraps the GPU fp32 CG. Each refinement iter computes a fp64
   residual on CPU, then solves the correction step on GPU in fp32,
   adding the result back. ~3-5 outer iters drives final residual
   below fp64 floor regardless of inner fp32 stagnation.
2. **Per-meshlet GPU CG.** Each meshlet's local matrix is small
   (~256×256) and well-conditioned (κ ≤ 10² typically); fp32 CG
   converges in 30-100 iters within fp32's precision budget. The
   meshlet decomposition path becomes load-bearing here — not just
   for parallelism, but for precision.
3. **fp64 GPU CG.** Native fp64 throughput is 1/16 (RDNA 2) to 1/64
   (Adreno) of fp32, so this kills the GPU advantage on the target
   hardware. Off the table for Steam Deck / Quest 3.



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
