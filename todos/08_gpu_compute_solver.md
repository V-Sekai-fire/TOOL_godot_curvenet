# GPU compute shader CG — Steam Deck + Quest 3 path to 100k real-time

**Goal:** the deformer runs the per-frame `VᵀLₕV · x = rhs` solve on the
GPU so a 100k-vertex character mesh stays in the 11 ms / 90 FPS budget
on both Steam Deck (RDNA 2, Vulkan) and Quest 3 (Adreno, Vulkan-Mobile).

**Why GPU is the only realistic path on these platforms:** Steam Deck
and Quest 3 are handheld-class CPUs (4 Zen 2 cores at ~3 GHz throttled,
ARMv9 mobile cores at ~3 GHz throttled). A planar-mesh sparse Cholesky
factor at 100k has Θ(n^1.5) ≈ 30M non-zeros; back-sub is Θ(n^1.5) ops
per RHS, 12 RHS = 360M ops/frame. At a *very generous* 10 GFLOP/s
sustained on these CPUs that's 36 ms/frame ≈ 28 FPS on Steam Deck and
worse on Quest 3 once thermals kick in. The same compute on the GPU
(~1.5–2 TFLOP/s) lands in 1–3 ms.

## Why not CPU sparse Cholesky on these specific platforms?

| | Steam Deck (Zen 2 throttled) | Quest 3 (Snapdragon XR2 Gen 2) |
|---|---|---|
| Sustained per-core FLOP/s | ~10 GFLOP/s | ~6 GFLOP/s |
| Hand-rolled supernodal Cholesky back-sub at 100k, 12 RHS | ~36 ms | ~60 ms |
| Reach 90 FPS at 100k? | No (~28 FPS) | No (~15 FPS) |
| Reach 60 FPS at 100k? | Marginal | No |

CHOLMOD-quality SIMD-tuned code would be ~5× faster but is out of reach
without months of dedicated optimisation work. Even with that, Quest 3
thermals make sustained 90 FPS at 100k uncertain.

## Architecture

```
existing                                    new
─────────                                   ──────────
src/curvenet/sparse_linalg.h               src/curvenet/gpu_sparse_solve.{h,cpp}
  SparseMatrixCSR                            GPUSparseSolver  (RD wrapper)
  spmv, cg                                   bind buffers, dispatch shaders
  (CPU fallback)                             cache pipeline + buffers across frames
                                            
                                           src/curvenet/shaders/spmv.glsl
                                             y = A · x  (1 thread per row, 1 RHS column)
                                           src/curvenet/shaders/dot_reduce.glsl
                                             scalar = Σ aᵢ bᵢ  (workgroup reduction)
                                             df32 accumulator: each lane keeps a
                                             (hi, lo) fp32 pair so the partial
                                             sums don't lose precision over 100K
                                             terms. Reduces to a single df32
                                             scalar; the CPU host reads back the
                                             pair and folds it to fp64 for α/β.
                                           src/curvenet/shaders/axpy.glsl
                                             y ← y + α x   (in place)
                                           src/curvenet/shaders/jacobi.glsl
                                             z ← b ⊘ diag(A)
```

The CG iteration loop itself stays on the CPU — it issues 4–5 dispatches
per iteration (spmv → axpy → axpy → dot → dot → jacobi → saxpby) and
reads the residual scalar back to decide convergence. Dispatches are
~10 µs each on these GPUs; with 20 warm-start iterations × 5 dispatches
= 100 dispatches per RHS column ≈ 1 ms. 12 RHS sequentially = 12 ms
(too slow). **Multi-RHS batching is mandatory** — see below.

### Multi-RHS batching

Stack the 12 RHS columns into a single nh × 12 matrix B. The spmv
shader processes them with one dispatch (12 mul-adds per row instead
of 1, hits the same memory once). The vector ops (axpy, dot) work
column-wise. **One dispatch per iteration, 12 columns at once.**
20 warm-start iterations × 5 dispatches = 100 dispatches per
*frame*, not per RHS. ~1 ms total at 10 µs per dispatch.

### Persistent buffers between frames

`VᵀLₕV` is built once at bind time and lives on the GPU. `x_v` from the
previous frame stays on the GPU as the warm start for the next solve —
no upload/download on the hot path. Per frame we only write the new
`Fc` / `Xc` (12 floats × nc samples ≈ a few KB) and read back the final
`x_v` (nv × 3 ≈ 1.2 MB at 100k verts → ~5 ms over the PCIe link on
Steam Deck, but Quest 3 unified memory makes this free).

For Quest 3 the entire deformed mesh array can be written to a Godot
ArrayMesh's vertex buffer directly via RD, skipping the CPU readback
entirely. That's a Phase 7+ win.

## Renderer switch

Current `demo/project.godot`:
```
[rendering]
renderer/rendering_method="gl_compatibility"
```

`gl_compatibility` has **no compute support**. Required change:

```
[rendering]
renderer/rendering_method="mobile"
renderer/rendering_method.mobile="mobile"
renderer/rendering_method.web="gl_compatibility"
```

Mobile (Forward+ Mobile) is the right call because:
* it works on Steam Deck (Vulkan)
* it's what Quest 3 wants natively (Quest's Vulkan requires the Mobile renderer)
* desktop testing on Mac/Windows/Linux works fine
* falls back to gl_compatibility on web/old-iOS builds where the
  CPU sparse CG covers (slower but functional)

Side effects of the switch we'll need to verify:
* VertexHandles3D's wireframe rendering still works (it uses
  `EditorNode3DGizmo::add_lines` — should be renderer-agnostic)
* CurveNetGizmo handle materials still draw (same)
* The demo's StandardMaterial3D may pick up different defaults
  (PBR vs unshaded); minor visual tuning expected

## Phases (ordered, each lands in its own commit)

1. **Renderer switch + RD scaffolding.** Change demo/project.godot;
   add a minimal "hello-world compute" shader (vector add) wired
   through `RenderingDevice` to confirm it runs on both Steam Deck and
   Quest 3. ~1–2 days.

2. **GPU sparse mat-vec.** Upload SparseMatrixCSR to 3 storage buffers,
   write spmv.glsl, dispatch, read back result. RapidCheck property:
   GPU spmv result == CPU spmv result within float eps. ~2 days.

3. **GPU vector ops.** dot_reduce.glsl, axpy.glsl, jacobi.glsl. Each
   tested against CPU. dot_reduce is the trickiest (workgroup reduction
   + final atomic-add or scalar buffer). ~2 days.

4. **GPU CG iteration loop on CPU.** Wire mat-vec + vector ops into the
   CG step. CPU loop reads residual norm to decide convergence. Test:
   GPU CG result matches CPU CG within tol on a few small instances.
   ~2–3 days.

5. **Multi-RHS batching.** Stack 12 RHS columns, dispatch 12-wide spmv.
   ~1 day.

6. **Persistent warm start.** x_v stays on GPU between frames; only
   Fc/Xc upload per frame. ~1 day.

7. **Wire into deformer.** RestCache gains a `gpu_solver_state` (pImpl
   to hide RD types from the header). At bind time, build GPU
   pipeline + upload VᵀLₕV. apply_deformation dispatches to GPU.
   CPU sparse CG remains as runtime fallback when RD compute is
   unavailable (web build, old hardware). ~2–3 days.

8. **Performance tuning.** Fused CG shader (one dispatch per iter
   instead of 5). Async readback for x_v on Steam Deck. Direct write
   to Godot ArrayMesh vertex buffer on Quest 3 (skip CPU roundtrip).
   Open-ended.

Total: ~2 weeks of focused work for Phases 1–7, then Phase 8 as
performance demands.

## Lean correspondence

The algorithm is the same — preconditioned CG over a sparse SPD matrix.
`lean/Curvenet/SparseLinAlg.lean` already proves the iteration
correctness via `native_decide` on small instances. The GPU shader
implements the same iteration; correctness flows through the existing
proof, with a property test asserting GPU and CPU outputs match within
float epsilon on RapidCheck-generated SPD inputs (n ≤ 16).

No new Lean work is required for the algorithm itself. We may want to
add a Lean-level "this matrix is SPD" predicate so the precondition
for CG is explicit, but that's a refinement, not a blocker.

## Risks

| Risk | Mitigation |
|---|---|
| `gl_compatibility` users (web build) lose support | Keep CPU sparse CG path; auto-select via RD::is_compute_supported |
| Float32 precision at 100k with high-κ Laplacian | Per-kernel split: fp32 in spmv/axpy/jacobi (each ~7 ε), df32 (two-fp32 pair, ~48-bit mantissa, Da Graça 2006) only in the dot reduction where the n·ε error of a 100K-element fp32 sum stagnates CG |
| Quest 3 Android Vulkan quirks | Test Phase 1 hello-world on a real Quest 3 before going deeper |
| Renderer-switch visual regression on demo | Tune StandardMaterial3D + light energy; this is a small visual tweak, not a blocker |
| Dispatch overhead dominates at small mesh sizes | At ≤5k verts the CPU path is faster anyway; runtime threshold for which solver to use |

## Acceptance criteria

* `apply_deformation` on a 100k-vertex character mesh completes in
  ≤2 ms on Steam Deck (Forward+ Mobile)
* same in ≤3 ms on Quest 3 in standalone mode
* warm-start CG reaches the residual tolerance in ≤20 iterations on
  continuous-drag interaction
* CPU fallback still produces identical results within float eps for
  meshes of size ≤16 (RapidCheck property in
  `tests/test_gpu_sparse_solve.cpp`)
* the demo scene drags handles in the editor at 60 FPS minimum on
  M-series Mac (development), confirming the integration path is
  correct before deploying to handheld

## Out of scope for this milestone

* GPU sparse Cholesky factorisation (much harder, marginal additional
  win over batched CG with warm starts)
* Multigrid preconditioner (would help convergence at 100k+ but adds
  a coarsening pass; deferred)
* Mesh streaming / LOD-based deformer (handles meshes > 100k by
  decimating; orthogonal to the solver)
